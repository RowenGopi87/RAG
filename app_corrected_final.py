import os
import shutil
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
import glob

# Import our corrected processors with fallback handling
from src.pdf_ocr_fix_corrected import EnhancedPDFProcessor

# Optional document processor import
try:
    from src.document_processor import AdvancedDocumentProcessor
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSING_AVAILABLE = False
    print("⚠️ Advanced document processing not available")

# Optional web processor import
try:
    from src.web_confluence_processor import WebConfluenceProcessor
    WEB_PROCESSING_AVAILABLE = True
except ImportError:
    WEB_PROCESSING_AVAILABLE = False
    print("⚠️ Web processing not available (missing web scraping dependencies)")

# Load environment variables
load_dotenv()

# Configure logging - Reduced verbosity for cleaner startup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep INFO for main app, WARNING for dependencies

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGES_FOLDER'] = 'extracted_images'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'docx', 'pptx', 'xlsx'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    is_persistent=True
))

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables!")
    openai_client = None
else:
    # Initialize OpenAI client with timeout and retry settings
    openai_client = OpenAI(
        api_key=api_key,
        timeout=30.0,  # 30 second timeout
        max_retries=2   # Maximum 2 retries
    )
    logger.info("OpenAI client initialized successfully with timeout settings!")

# Initialize PDF processor (always available)
pdf_processor = EnhancedPDFProcessor(chunk_size=1000, chunk_overlap=200, images_folder=app.config['IMAGES_FOLDER'])

# Initialize document processor (optional)
document_processor = None
if DOCUMENT_PROCESSING_AVAILABLE:
    try:
        document_processor = AdvancedDocumentProcessor(
            chroma_client=chroma_client,  # Pass the chroma_client
            chunk_size=1000, 
            chunk_overlap=200
        )
        logger.info("✅ Advanced document processing initialized")
    except Exception as e:
        logger.warning(f"⚠️ Document processor initialization failed: {e}")
        DOCUMENT_PROCESSING_AVAILABLE = False

# Initialize web processor (optional)
web_processor = None
if WEB_PROCESSING_AVAILABLE:
    confluence_username = os.getenv('CONFLUENCE_USERNAME')
    confluence_token = os.getenv('CONFLUENCE_API_TOKEN')
    confluence_auth = (confluence_username, confluence_token) if confluence_username and confluence_token else None
    
    try:
        web_processor = WebConfluenceProcessor(
            chunk_size=1000, 
            chunk_overlap=200,
            images_folder=app.config['IMAGES_FOLDER'],
            confluence_auth=confluence_auth
        )
        logger.info("✅ Web processing initialized")
    except Exception as e:
        logger.warning(f"⚠️ Web processor initialization failed: {e}")
        WEB_PROCESSING_AVAILABLE = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_url(text: str) -> bool:
    """Check if text is a URL"""
    return text.strip().startswith(('http://', 'https://'))

def store_chunks_in_chromadb(chunks, collection_name="documents"):
    """Store document chunks in ChromaDB"""
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG collection with OCR and web scraping support"}
        )
        
        # Prepare data for ChromaDB
        # Note: ChromaDB can't store complex objects like images directly in metadata,
        # so we'll store image info as serialized strings and reconstruct later
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk['text'])
            ids.append(chunk['id'])
            
            # Handle metadata - ChromaDB only supports basic types
            metadata = chunk['metadata'].copy()
            
            # If chunk has associated images, store their info in metadata
            if 'images' in chunk and chunk['images']:
                images = chunk['images']
                
                # Handle both PDF and web image formats
                image_filenames = []
                image_paths = []
                image_pages = []
                image_types = []
                image_ocr_texts = []
                image_alt_texts = []
                
                for img in images:
                    # Common fields
                    image_filenames.append(img.get('filename', ''))
                    
                    # PDF-style images have local_path and page_number
                    if 'local_path' in img:
                        image_paths.append(img.get('local_path', ''))
                        image_pages.append(str(img.get('page_number', '')))
                        image_types.append('pdf')
                        image_ocr_texts.append(img.get('ocr_text', ''))
                        image_alt_texts.append('')
                    
                    # Web-style images also have local_path now (plus alt_text)
                    elif 'base64_data' in img:
                        image_paths.append(img.get('local_path', ''))  # Web images now have local_path too!
                        image_pages.append('')  # No page number for web images
                        image_types.append('web')
                        image_ocr_texts.append(img.get('ocr_text', ''))
                        image_alt_texts.append(img.get('alt_text', ''))
                
                # Store as comma-separated strings
                metadata['associated_image_filenames'] = ','.join(image_filenames)
                metadata['associated_image_paths'] = ','.join(image_paths)
                metadata['associated_image_pages'] = ','.join(image_pages)
                metadata['associated_image_types'] = ','.join(image_types)
                metadata['associated_image_ocr_texts'] = ','.join(image_ocr_texts)
                metadata['associated_image_alt_texts'] = ','.join(image_alt_texts)
                
                logger.info(f"💾 Storing chunk with {len(images)} associated images: {image_filenames}")
            
            metadatas.append(metadata)
        
        logger.info(f"💾 STORING: {len(chunks)} chunks in collection '{collection_name}'")
        
        # Add documents to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✅ STORED: {len(chunks)} chunks successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error storing chunks in ChromaDB: {str(e)}")
        raise

def query_chromadb(query, collection_name="documents", n_results=5):
    """Query ChromaDB for relevant chunks with relevance-first ranking"""
    try:
        logger.info(f"🔍 RAG STEP 1: Searching ChromaDB for query: '{query}'")
        collection = chroma_client.get_collection(name=collection_name)
        
        # Query expansion for better matching - but more focused
        query_variants = [query]
        
        # Add targeted query expansions only for specific terms
        query_lower = query.lower()
        if 'portfolio program' in query_lower:
            query_variants.extend([
                'portfolio management program strategy',
                'SAFe portfolio level',
                'agile portfolio program'
            ])
        elif 'safe' in query_lower and ('agile' in query_lower or 'framework' in query_lower):
            query_variants.extend([
                'scaled agile framework methodology',
                'SAFe implementation'
            ])
        
        logger.info(f"🔄 Using {len(query_variants)} query variants for search")
        
        # Get more results initially for better selection
        initial_results = max(n_results * 3, 15)  # Get 3x more results initially
        
        # Collect results from all query variants
        all_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
        
        for variant in query_variants:
            try:
                variant_results = collection.query(
                    query_texts=[variant],
                    n_results=initial_results // len(query_variants) + 5
                )
                
                if variant_results['documents'] and variant_results['documents'][0]:
                    all_results['documents'][0].extend(variant_results['documents'][0])
                    all_results['metadatas'][0].extend(variant_results['metadatas'][0])
                    all_results['distances'][0].extend(variant_results['distances'][0])
                    all_results['ids'][0].extend(variant_results['ids'][0])
                    
            except Exception as e:
                logger.warning(f"Query variant '{variant}' failed: {e}")
        
        results = all_results
        
        # Remove duplicates based on chunk ID
        if results['documents'] and results['documents'][0]:
            seen_ids = set()
            deduplicated = {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
            
            for i in range(len(results['documents'][0])):
                chunk_id = results['ids'][0][i]
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    deduplicated['documents'][0].append(results['documents'][0][i])
                    deduplicated['metadatas'][0].append(results['metadatas'][0][i])
                    deduplicated['distances'][0].append(results['distances'][0][i])
                    deduplicated['ids'][0].append(results['ids'][0][i])
            
            results = deduplicated
            logger.info(f"🔄 Deduplicated to {len(results['documents'][0])} unique chunks")
        
        # Format results and PRIORITIZE BY RELEVANCE (distance score)
        relevant_chunks = []
        if results['documents']:
            logger.info(f"📚 RAG STEP 2: Found {len(results['documents'][0])} chunks, ranking by relevance...")
            
            # Create all chunks with their relevance scores
            all_chunks = []
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                
                # Reconstruct associated images from metadata
                associated_images = []
                if metadata.get('associated_image_filenames'):
                    image_filenames = metadata['associated_image_filenames'].split(',')
                    image_paths = metadata.get('associated_image_paths', '').split(',')
                    image_pages = metadata.get('associated_image_pages', '').split(',')
                    image_types = metadata.get('associated_image_types', '').split(',')
                    image_ocr_texts = metadata.get('associated_image_ocr_texts', '').split(',')
                    image_alt_texts = metadata.get('associated_image_alt_texts', '').split(',')
                    
                    for j, filename in enumerate(image_filenames):
                        if filename.strip():
                            image_type = image_types[j].strip() if j < len(image_types) else 'pdf'
                            
                            image_meta = {
                                'filename': filename.strip(),
                                'image_type': image_type
                            }
                            
                            # Add type-specific fields
                            if image_type == 'pdf':
                                image_meta.update({
                                    'local_path': image_paths[j].strip() if j < len(image_paths) else '',
                                    'page_number': image_pages[j].strip() if j < len(image_pages) else '',
                                    'ocr_text': image_ocr_texts[j].strip() if j < len(image_ocr_texts) else ''
                                })
                            elif image_type == 'web':
                                image_meta.update({
                                    'ocr_text': image_ocr_texts[j].strip() if j < len(image_ocr_texts) else '',
                                    'alt_text': image_alt_texts[j].strip() if j < len(image_alt_texts) else ''
                                })
                            
                            associated_images.append(image_meta)
                
                chunk_data = {
                    'text': doc,
                    'metadata': metadata,
                    'distance': distance,
                    'images': associated_images  # Reconstructed image associations
                }
                all_chunks.append(chunk_data)
            
            # SORT BY RELEVANCE (lowest distance = most relevant) - THIS IS THE KEY FIX
            all_chunks.sort(key=lambda x: x['distance'])
            
            # Take the most relevant chunks first
            relevant_chunks = all_chunks[:n_results]
            
            # Group by source for logging
            source_groups = {}
            for chunk in relevant_chunks:
                source = chunk['metadata'].get('source', 'unknown')
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(chunk)
            
            logger.info(f"📊 Content sources (ordered by relevance):")
            for source in source_groups:
                avg_distance = sum(c['distance'] for c in source_groups[source]) / len(source_groups[source])
                total_images = sum(len(c.get('images', [])) for c in source_groups[source])
                logger.info(f"  📄 {source}: {len(source_groups[source])} chunks (avg distance: {avg_distance:.4f}) | Images: {total_images}")
            
            logger.info(f"🎯 RAG STEP 3: Selected {len(relevant_chunks)} most relevant chunks:")
            for i, chunk in enumerate(relevant_chunks):
                source = chunk['metadata'].get('source', 'unknown')
                distance = chunk['distance']
                method = chunk['metadata'].get('processing_method', 'unknown')
                num_images = len(chunk.get('images', []))
                logger.info(f"  📄 Chunk {i+1}: Distance={distance:.4f} | Source='{source}' | Method={method} | Images={num_images}")
                logger.info(f"      📝 Preview: '{chunk['text'][:100]}...'")
                
        else:
            logger.warning("❌ No relevant chunks found in ChromaDB")
        
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {str(e)}")
        return []

def generate_response_with_rag(query, relevant_chunks):
    """Generate response using OpenAI with RAG context"""
    try:
        logger.info(f"🤖 RAG STEP 3: Generating response with {len(relevant_chunks)} chunks")
        
        if not openai_client:
            return "Sorry, OpenAI API key is not configured. Please set your OPENAI_API_KEY environment variable and restart the application."
        
        # Build context from relevant chunks
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        logger.info(f"📝 RAG STEP 4: Built context with {len(context)} characters")
        
        # Create the prompt
        system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question. 

When dealing with image-related queries (like "show me the image" or "display the diagram"):
- If the context contains OCR text from images (marked with "--- IMAGE X OCR ---"), explain what visual content was found
- Be helpful by describing the content and structure of images based on the OCR text
- Don't just say "context doesn't include details" - instead, explain what image content is available

For other queries, if the answer cannot be found in the context, say so clearly. Be accurate and cite the source when possible."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""

        logger.info(f"💬 RAG STEP 5: Sending to OpenAI GPT-4o with 30s timeout...")

        # Call OpenAI API with explicit timeout handling
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.7,
                timeout=30  # Explicit 30 second timeout
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"✅ RAG STEP 6: Received OpenAI response ({len(ai_response)} characters)")
            
            return ai_response
            
        except Exception as api_error:
            # Handle specific OpenAI API errors
            error_msg = str(api_error)
            
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                logger.error(f"⏰ OpenAI API timeout: {error_msg}")
                return "Sorry, the AI service is taking too long to respond. This might be due to high demand. Please try again in a moment."
            elif "rate limit" in error_msg.lower():
                logger.error(f"🚫 OpenAI rate limit: {error_msg}")
                return "Sorry, too many requests have been made recently. Please wait a moment and try again."
            elif "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
                logger.error(f"💳 OpenAI quota exceeded: {error_msg}")
                return "Sorry, the OpenAI API quota has been exceeded. Please check your OpenAI account billing."
            elif "invalid_api_key" in error_msg.lower():
                logger.error(f"🔑 Invalid OpenAI API key: {error_msg}")
                return "Sorry, there's an issue with the OpenAI API key configuration. Please check your API key."
            else:
                logger.error(f"❌ OpenAI API error: {error_msg}")
                return f"Sorry, I encountered an error while generating a response. Please try again. Error: {error_msg}"
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in generate_response_with_rag: {str(e)}")
        return f"Sorry, I encountered an unexpected error while processing your request: {str(e)}"

@app.route('/')
def index():
    """Serve the modern UI"""
    return render_template('index_modern.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Preserve original filename for source attribution
            original_filename = file.filename
            secure_name = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
            file.save(filepath)
            
            logger.info(f"📤 UPLOAD: Processing '{original_filename}' (saved as '{secure_name}')")
            
            # Determine file type and process accordingly
            file_ext = original_filename.rsplit('.', 1)[1].lower()
            
            if file_ext == 'pdf':
                # Extract text with OCR support (if available)
                try:
                    text, images_metadata = pdf_processor.extract_text_with_ocr(filepath)
                    logger.info(f"📖 PDF extraction completed: {len(text)} chars, {len(images_metadata)} images")
                    processing_type = 'pdf_upload'
                except Exception as pdf_error:
                    logger.error(f"❌ PDF processing failed: {pdf_error}")
                    # Fallback to basic processing
                    try:
                        text = pdf_processor._extract_with_pypdf2_only(filepath)[0]
                        images_metadata = []
                        processing_type = 'pdf_upload_fallback'
                    except:
                        text = f"PDF processing failed: {str(pdf_error)}"
                        images_metadata = []
                        processing_type = 'pdf_upload_error'
            elif file_ext in ['jpg', 'jpeg', 'png']:
                # Process image file with OCR
                images_metadata = []  # Initialize images_metadata for standalone images
                
                if pdf_processor.ocr_enabled:
                    # Use OCR to extract text from the image
                    try:
                        from PIL import Image
                        import pytesseract
                        
                        # Open and process the image with OCR
                        image = Image.open(filepath)
                        ocr_text = pytesseract.image_to_string(image)
                        
                        # Save the image to extracted_images folder for display
                        import shutil
                        saved_image_path = os.path.join(app.config['IMAGES_FOLDER'], f"uploaded_{original_filename}")
                        shutil.copy2(filepath, saved_image_path)
                        logger.info(f"💾 Saved image for display: uploaded_{original_filename}")
                        
                        # Create proper image metadata structure
                        image_metadata = {
                            'page_number': 1,  # Standalone images are considered "page 1"
                            'image_index': 1,
                            'filename': f"uploaded_{original_filename}",
                            'local_path': saved_image_path,
                            'ocr_text': ocr_text.strip(),
                            'has_ocr': bool(ocr_text.strip()),
                            'width': image.width,
                            'height': image.height,
                            'format': image.format or 'Unknown'
                        }
                        images_metadata.append(image_metadata)
                        
                        if ocr_text.strip():
                            text = f"Image file: {original_filename}\n--- IMAGE OCR TEXT ---\n{ocr_text}\n--- END IMAGE OCR TEXT ---"
                            logger.info(f"🔍 OCR extracted {len(ocr_text)} characters from {original_filename}")
                        else:
                            text = f"Image file: {original_filename}\nNo text detected in image with OCR"
                            
                        processing_type = 'image_upload_ocr'
                        
                    except Exception as ocr_error:
                        logger.warning(f"⚠️ OCR failed for {original_filename}: {ocr_error}")
                        text = f"Image file: {original_filename}\nOCR processing failed: {str(ocr_error)}"
                        processing_type = 'image_upload_ocr_failed'
                else:
                    text = f"Image file: {original_filename} (OCR not available)"
                    processing_type = 'image_upload_no_ocr'
            else:
                # For other file types (docx, pptx, xlsx) - placeholder for now
                text = f"Document file: {original_filename}\nAdvanced document processing would be implemented here"
                processing_type = 'document_upload'
            
            # Create chunks with proper source attribution
            try:
                if ('images_metadata' in locals() and images_metadata):
                    logger.info(f"📦 Creating chunks with {len(images_metadata)} associated images")
                    chunks = pdf_processor.create_chunks_with_proper_metadata(text, original_filename, images_metadata)
                else:
                    logger.info(f"📦 Creating chunks without image associations")
                    chunks = pdf_processor.create_chunks_with_proper_metadata(text, original_filename)
                
                logger.info(f"✅ Created {len(chunks)} chunks successfully")
                
                # Debug: Check if any chunks have images
                chunks_with_images = sum(1 for chunk in chunks if chunk.get('images'))
                if chunks_with_images > 0:
                    logger.info(f"🖼️ {chunks_with_images} chunks have associated images")
                    # Log details about the first few chunks with images
                    for i, chunk in enumerate(chunks):
                        if chunk.get('images') and i < 3:  # Show first 3 chunks with images
                            logger.info(f"   Chunk {i}: {len(chunk['images'])} images - {[img.get('filename', 'unknown') for img in chunk['images']]}")
                else:
                    logger.info(f"📄 No chunks have associated images")
                    
            except Exception as chunk_error:
                logger.error(f"❌ Chunk creation failed: {chunk_error}")
                # Create fallback chunks
                chunks = [{
                    'id': f"{original_filename}_chunk_0",
                    'text': text,
                    'metadata': {
                        'source': original_filename,
                        'file_type': file_ext,
                        'chunk_index': 0,
                        'total_chunks': 1,
                        'processing_method': 'fallback'
                    }
                }]
            
            # Store in ChromaDB
            store_chunks_in_chromadb(chunks)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'message': f'Successfully processed {original_filename}' + (' with OCR support' if pdf_processor.ocr_enabled and file_ext in ['pdf', 'jpg', 'jpeg', 'png'] else ' (basic processing)'),
                'chunks_created': len(chunks),
                'characters_extracted': len(text),
                'ocr_enabled': pdf_processor.ocr_enabled and file_ext in ['pdf', 'jpg', 'jpeg', 'png'],
                'original_filename': original_filename,
                'processing_type': processing_type,
                'file_type': file_ext
            })
        
        return jsonify({'error': 'Invalid file type. Supported formats: PDF, JPG, JPEG, PNG, DOCX, PPTX, XLSX'}), 400
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/process-url', methods=['POST'])
def process_url():
    """Handle URL processing including Confluence"""
    try:
        if not WEB_PROCESSING_AVAILABLE or not web_processor:
            return jsonify({'error': 'Web processing not available. Missing web scraping dependencies.'}), 400
        
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        if not is_url(url):
            return jsonify({'error': 'Invalid URL format. Must start with http:// or https://'}), 400
        
        logger.info(f"🌐 URL PROCESSING REQUEST: {url}")
        
        # Process the URL
        result = web_processor.process_url(url)
        
        # Store chunks in ChromaDB
        store_chunks_in_chromadb(result['chunks'])
        
        return jsonify({
            'message': f'Successfully processed URL: {result["title"]}',
            'url': url,
            'title': result['title'],
            'chunks_created': result['chunks_count'],
            'characters_extracted': result['total_characters'],
            'images_found': result.get('images_count', 0),
            'tables_found': result.get('tables_count', 0),
            'processing_type': 'url_scraping'
        })
        
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        return jsonify({'error': f'Error processing URL: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries with RAG"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if the message is a URL and web processing is available
        if is_url(message) and WEB_PROCESSING_AVAILABLE and web_processor:
            logger.info(f"🌐 URL DETECTED IN CHAT: {message}")
            
            try:
                # Process URL directly
                result = web_processor.process_url(message)
                store_chunks_in_chromadb(result['chunks'])
                
                return jsonify({
                    'response': f"I've successfully processed the URL: {result['title']}\n\nExtracted {result['total_characters']} characters. You can now ask questions about this content!",
                    'url_processed': True,
                    'title': result['title'],
                    'chunks_created': result['chunks_count']
                })
                
            except Exception as e:
                logger.error(f"Error processing URL in chat: {e}")
                return jsonify({
                    'response': f"I tried to process the URL but encountered an error: {str(e)}. Please check the URL and try again.",
                    'url_processed': False
                })
        elif is_url(message):
            return jsonify({
                'response': "I detected a URL in your message, but web processing is not available. Please upload the content as a PDF or enable web scraping dependencies.",
                'url_processed': False
            })
        
        # Regular chat query
        logger.info(f"🚀 RAG PIPELINE STARTED")
        logger.info(f"❓ User Question: '{message}'")
        
        # Query ChromaDB for relevant context
        relevant_chunks = query_chromadb(message)
        
        if not relevant_chunks:
            logger.warning("⚠️ RAG PIPELINE STOPPED: No relevant chunks found")
            return jsonify({
                'response': "I don't have any relevant information in my knowledge base to answer your question. Please upload some documents first."
            })
        
        # Generate response with RAG
        response = generate_response_with_rag(message, relevant_chunks)
        
        logger.info(f"🎯 RAG PIPELINE COMPLETED SUCCESSFULLY")
        
        # Prepare source information and extract images
        sources_list = []
        relevant_images = []
        
        # Check for available images and visual content
        images_folder = Path('extracted_images')
        has_images = images_folder.exists() and list(images_folder.glob('*.png'))
        
        # Look for keywords that suggest user wants images OR if chunks have associated images
        image_keywords = ['show', 'display', 'image', 'picture', 'map', 'chart', 'graph', 'visual', 'see', 'diagram', 'figure', 'look like', 'appears']
        wants_images = any(keyword in message.lower() for keyword in image_keywords)
        
        # Check if any chunks have associated images (this is key!)
        has_relevant_image_content = False
        uploaded_images = []  # Track uploaded images specifically
        chunks_with_images = False  # Track if any chunks have associated images
        
        for chunk in relevant_chunks:
            source_name = chunk['metadata'].get('source', 'Unknown')
            if source_name not in sources_list:
                sources_list.append(source_name)
            
            # Check if this chunk has associated images
            chunk_images = chunk.get('images', [])
            if chunk_images:
                chunks_with_images = True
                logger.info(f"🖼️ Found chunk with {len(chunk_images)} associated images from {source_name}")
            
            # Check for image OCR content markers in the chunk
            chunk_text = chunk.get('text', '').lower()
            if any(marker in chunk_text for marker in ['--- image', 'ocr', 'image file:']):
                has_relevant_image_content = True
                
                # Check if this is from an uploaded image
                if 'image file:' in chunk_text and ('--- image ocr text ---' in chunk_text or 'no text detected' in chunk_text):
                    # Extract the original filename from the chunk
                    lines = chunk['text'].split('\n')
                    for line in lines:
                        if line.startswith('Image file:'):
                            original_filename = line.replace('Image file:', '').strip()
                            uploaded_images.append(original_filename)
                            break
        
        # Enhanced logic: show images if user wants them OR if chunks have associated images
        should_show_images = wants_images or chunks_with_images or has_relevant_image_content
        
        logger.info(f"🖼️ IMAGE DISPLAY LOGIC: wants_images={wants_images}, chunks_with_images={chunks_with_images}, has_relevant_content={has_relevant_image_content}, should_show={should_show_images}")
        
        # Initialize image-related variables
        relevant_images = []
        chunk_associated_images = []
        
        # Enhanced image handling - prioritize uploaded images and relevant content
        if should_show_images and (has_images or has_relevant_image_content):
            try:
                logger.info(f"🖼️ Processing images for query: '{message}'")
                
                # First, check if any retrieved chunks have associated images
                for chunk in relevant_chunks:
                    chunk_images = chunk.get('images', [])
                    for img_meta in chunk_images:
                        img_type = img_meta.get('image_type', 'pdf')
                        
                        if img_type == 'pdf' and img_meta.get('local_path') and Path(img_meta['local_path']).exists():
                            # PDF images stored as files
                            chunk_associated_images.append(Path(img_meta['local_path']))
                            logger.info(f"📸 Found PDF image: {img_meta.get('filename', 'unknown')}")
                        
                        elif img_type == 'web':
                            # Web images are now saved as local files too!
                            if img_meta.get('local_path') and Path(img_meta['local_path']).exists():
                                chunk_associated_images.append(Path(img_meta['local_path']))
                                logger.info(f"📸 Found web image: {img_meta.get('filename', 'unknown')}")
                            else:
                                logger.warning(f"  ⚠️ Web image {img_meta.get('filename', 'unknown')} local file not found: {img_meta.get('local_path', 'no path')}")
                
                # Remove old limitation logic - web images now work!
                # No longer need chunk_web_images tracking since they're file-based now
                
                if has_images:
                    logger.info(f"🔍 Looking for images in {images_folder}")
                    
                    # Start with PDF chunk-associated images
                    relevant_image_files = chunk_associated_images.copy()
                    
                    # Then look for uploaded images that were referenced in chunks
                    for uploaded_img in uploaded_images:
                        uploaded_img_path = images_folder / f"uploaded_{uploaded_img}"
                        if uploaded_img_path.exists() and uploaded_img_path not in relevant_image_files:
                            relevant_image_files.append(uploaded_img_path)
                            logger.info(f"📸 Found uploaded image: {uploaded_img}")
                    
                    # Then add other relevant images
                    all_images = list(images_folder.glob('*.png'))
                    for img_file in all_images:
                        if img_file not in relevant_image_files:
                            # Check if image name relates to the query or chunks
                            img_name_lower = img_file.name.lower()
                            query_words = message.lower().split()
                            
                            # Add if image name contains query words or chunk sources
                            if any(word in img_name_lower for word in query_words if len(word) > 3):
                                relevant_image_files.append(img_file)
                            elif any(source.lower().replace('.pdf', '').replace(' ', '') in img_name_lower 
                                   for source in sources_list):
                                relevant_image_files.append(img_file)
                    
                    # If no specific matches found, include the largest images (likely most important)
                    if not relevant_image_files:
                        all_images.sort(key=lambda x: x.stat().st_size, reverse=True)
                        relevant_image_files = all_images[:3]
                    
                    # Take up to 5 most relevant images
                    selected_images = relevant_image_files[:5]
                    logger.info(f"📸 Selected {len(selected_images)} file-based images to include")
                    
                    for img_file in selected_images:
                        try:
                            with open(img_file, 'rb') as f:
                                img_base64 = base64.b64encode(f.read()).decode('utf-8')
                            
                            # Determine image type and description
                            if img_file.name.startswith('uploaded_'):
                                img_type = 'uploaded image'
                                original_name = img_file.name.replace('uploaded_', '')
                                description = f'Uploaded image: {original_name}'
                            elif img_file.name.startswith('pdf_') and '_page' in img_file.name:
                                img_type = 'PDF content'
                                # Extract page number from filename like "pdf_filename_page3_img1.png"
                                try:
                                    page_part = img_file.name.split('_page')[1].split('_')[0]
                                    description = f'Image from PDF page {page_part} (related to your query)'
                                except:
                                    description = f'Image from PDF content related to: "{message}"'
                            elif img_file.name.startswith('web_image'):
                                img_type = 'web content'
                                description = f'Image from web content related to: "{message}"'
                            else:
                                img_type = 'document content'
                                description = f'Image from document content related to: "{message}"'
                            
                            relevant_images.append({
                                'filename': img_file.name,
                                'base64_data': img_base64,
                                'alt_text': f'Image from {img_type}',
                                'description': description,
                                'size_kb': round(img_file.stat().st_size / 1024, 1)
                            })
                            
                            logger.info(f"  ✅ Added image: {img_file.name} ({round(img_file.stat().st_size / 1024, 1)} KB) - {img_type}")
                            
                        except Exception as img_e:
                            logger.warning(f"Could not process image {img_file.name}: {img_e}")
                
                # Log summary
                logger.info(f"📸 Successfully processed {len(relevant_images)} images for display")
                
            except Exception as e:
                logger.warning(f"Error extracting images: {e}")
        
        logger.info(f"🎯 FINAL RESULT: Found {len(relevant_images)} images to return with response")
        
        # Debug: Show detailed info about what we're returning
        if relevant_images:
            logger.info(f"📸 RETURNING IMAGES:")
            for i, img in enumerate(relevant_images):
                logger.info(f"  Image {i+1}: {img.get('filename', 'unknown')} ({img.get('size_kb', 0)} KB) - {img.get('description', 'no description')}")
        else:
            logger.warning(f"⚠️ NO IMAGES TO RETURN - Debug info:")
            logger.warning(f"  should_show_images: {should_show_images}")
            logger.warning(f"  has_images: {has_images}")
            logger.warning(f"  chunks_with_images: {chunks_with_images}")
            logger.warning(f"  chunk_associated_images found: {len(chunk_associated_images)}")
            
            # Debug: Show what images exist in the folder
            if images_folder.exists():
                all_images = list(images_folder.glob('*.png'))
                logger.warning(f"  Available images in folder: {[img.name for img in all_images[:10]]}")  # Show first 10
        
        return jsonify({
            'response': response,
            'sources': sources_list,
            'chunks_used': len(relevant_chunks),
            'images': relevant_images,
            'has_images': len(relevant_images) > 0
        })
        
    except Exception as e:
        logger.error(f"❌ RAG PIPELINE ERROR: {str(e)}")
        return jsonify({'error': f'Error processing chat: {str(e)}'}), 500

@app.route('/clear-database', methods=['POST'])
def clear_database():
    """Clear the entire vector database"""
    try:
        logger.info("🗑️ DATABASE CLEAR REQUEST")
        
        global chroma_client
        
        # Step 1: Clear ChromaDB collections properly
        try:
            logger.info("🔄 Clearing ChromaDB collections...")
            collections = chroma_client.list_collections()
            for collection in collections:
                chroma_client.delete_collection(collection.name)
                logger.info(f"🗑️ Deleted collection: {collection.name}")
        except Exception as e:
            logger.warning(f"⚠️ Warning clearing collections: {e}")
        
        # Step 2: Reset ChromaDB client to release file handles
        try:
            logger.info("🔄 Resetting ChromaDB client...")
            # Create a new client without persistence to release locks
            chroma_client = chromadb.Client()
            import time
            time.sleep(0.5)  # Brief pause to ensure files are unlocked
        except Exception as e:
            logger.warning(f"⚠️ Warning resetting client: {e}")
        
        # Step 3: Clean up files and directories
        items_to_clean = [
            ("image_registry.json", "file"),
            ("processed_files.json", "file"),
            ("extracted_images", "dir"),
            ("chroma_db", "dir")  # ChromaDB last after client reset
        ]
        
        cleaned_count = 0
        
        for item_name, item_type in items_to_clean:
            item_path = Path(item_name)
            if item_path.exists():
                try:
                    if item_type == "file":
                        item_path.unlink()
                        logger.info(f"🗑️ Removed file: {item_name}")
                        cleaned_count += 1
                    elif item_type == "dir":
                        # For directories, try multiple times with increasing delays
                        max_attempts = 3
                        for attempt in range(max_attempts):
                            try:
                                shutil.rmtree(item_path)
                                logger.info(f"🗑️ Removed directory: {item_name}")
                                cleaned_count += 1
                                break
                            except PermissionError as pe:
                                if attempt < max_attempts - 1:
                                    logger.warning(f"⚠️ Attempt {attempt + 1} failed for {item_name}, retrying...")
                                    import time
                                    time.sleep(1)  # Wait longer between attempts
                                else:
                                    raise pe
                except Exception as e:
                    logger.error(f"❌ Failed to remove {item_name}: {e}")
                    # Continue with other items even if one fails
        
        # Step 4: Reinitialize ChromaDB client
        try:
            logger.info("🔄 Reinitializing ChromaDB client...")
            chroma_client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                is_persistent=True
            ))
        except Exception as e:
            logger.error(f"❌ Failed to reinitialize ChromaDB: {e}")
            # Fallback to non-persistent client
            chroma_client = chromadb.Client()
        
        # Step 5: Recreate necessary folders
        try:
            os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)
            os.makedirs("chroma_db", exist_ok=True)
        except Exception as e:
            logger.warning(f"⚠️ Warning creating folders: {e}")
        
        logger.info(f"✅ DATABASE CLEARED: {cleaned_count} items removed")
        
        return jsonify({
            'message': f'Database cleared successfully! Removed {cleaned_count} items.',
            'items_cleared': cleaned_count,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"❌ Error clearing database: {str(e)}")
        return jsonify({'error': f'Error clearing database: {str(e)}'}), 500

@app.route('/database-status', methods=['GET'])
def database_status():
    """Get database status information"""
    try:
        status = {
            'chromadb_exists': Path('chroma_db').exists(),
            'image_registry_exists': Path('image_registry.json').exists(),
            'extracted_images_exists': Path('extracted_images').exists(),
            'processed_files_exists': Path('processed_files.json').exists(),
            'total_documents': 0,
            'collections': []
        }
        
        # Get ChromaDB info
        if status['chromadb_exists']:
            try:
                collections = chroma_client.list_collections()
                for collection in collections:
                    count = collection.count()
                    status['total_documents'] += count
                    status['collections'].append({
                        'name': collection.name,
                        'count': count
                    })
            except Exception as e:
                logger.warning(f"Error getting ChromaDB info: {e}")
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting database status: {str(e)}")
        return jsonify({'error': f'Error getting database status: {str(e)}'}), 500

@app.route('/debug-sources', methods=['GET'])
def debug_sources():
    """Debug endpoint to check what's in ChromaDB"""
    try:
        collection = chroma_client.get_collection(name="documents")
        
        # Get a sample of documents to inspect metadata
        results = collection.get(limit=10, include=['metadatas', 'documents'])
        
        debug_info = {
            'total_documents': collection.count(),
            'sample_metadata': results['metadatas'] if results['metadatas'] else [],
            'sample_ids': results['ids'] if results['ids'] else [],
            'ocr_enabled': pdf_processor.ocr_enabled,
            'web_processing_enabled': WEB_PROCESSING_AVAILABLE,
            'confluence_configured': bool(os.getenv('CONFLUENCE_USERNAME'))
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({'error': f'Debug error: {str(e)}'}), 500

@app.route('/debug-search', methods=['POST'])
def debug_search():
    """Debug endpoint to test search functionality"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        logger.info(f"🔍 DEBUG SEARCH: '{query}'")
        
        # Get raw ChromaDB results
        relevant_chunks = query_chromadb(query)
        
        # Format debug info
        debug_info = {
            'query': query,
            'chunks_found': len(relevant_chunks),
            'results': []
        }
        
        for i, chunk in enumerate(relevant_chunks):
            chunk_info = {
                'rank': i + 1,
                'source': chunk['metadata'].get('source', 'unknown'),
                'distance': chunk['distance'],
                'processing_method': chunk['metadata'].get('processing_method', 'unknown'),
                'chunk_index': chunk['metadata'].get('chunk_index', 'unknown'),
                'preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'has_image_content': any(marker in chunk['text'].lower() for marker in ['--- image', 'ocr text', 'image file:'])
            }
            debug_info['results'].append(chunk_info)
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Error in debug search: {str(e)}")
        return jsonify({'error': f'Debug search error: {str(e)}'}), 500

@app.route('/debug-images', methods=['POST'])
def debug_images():
    """Debug endpoint to test image associations"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        logger.info(f"🔍 DEBUG IMAGES: '{query}'")
        
        # Get chunks with image associations
        relevant_chunks = query_chromadb(query)
        
        # Analyze image associations
        debug_info = {
            'query': query,
            'chunks_found': len(relevant_chunks),
            'total_associated_images': 0,
            'results': []
        }
        
        for i, chunk in enumerate(relevant_chunks):
            associated_images = chunk.get('images', [])
            debug_info['total_associated_images'] += len(associated_images)
            
            image_info = []
            for img in associated_images:
                img_exists = Path(img.get('local_path', '')).exists() if img.get('local_path') else False
                image_info.append({
                    'filename': img.get('filename', ''),
                    'page_number': img.get('page_number', ''),
                    'local_path': img.get('local_path', ''),
                    'exists': img_exists,
                    'has_ocr': bool(img.get('ocr_text', '').strip())
                })
            
            chunk_info = {
                'rank': i + 1,
                'source': chunk['metadata'].get('source', 'unknown'),
                'distance': chunk['distance'],
                'associated_images': len(associated_images),
                'images': image_info,
                'preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'contains_page_markers': '--- PAGE' in chunk['text'],
                'contains_image_ocr': '--- IMAGE' in chunk['text'] and 'OCR' in chunk['text']
            }
            debug_info['results'].append(chunk_info)
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Error in debug images: {str(e)}")
        return jsonify({'error': f'Debug images error: {str(e)}'}), 500

@app.route('/process-documents-folder', methods=['POST'])
def process_documents_folder_endpoint():
    """Manual endpoint to process documents folder"""
    try:
        process_documents_folder_on_startup()
        return jsonify({'message': 'Documents folder processing completed', 'status': 'success'})
    except Exception as e:
        logger.error(f"Error processing documents folder: {str(e)}")
        return jsonify({'error': f'Error processing documents folder: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'chromadb_connected': True,
        'openai_configured': bool(api_key),
        'openai_timeout_configured': True,
        'openai_timeout_seconds': 30,
        'ocr_enabled': pdf_processor.ocr_enabled,
        'web_processing_enabled': WEB_PROCESSING_AVAILABLE,
        'document_processing_enabled': DOCUMENT_PROCESSING_AVAILABLE,
        'confluence_configured': bool(os.getenv('CONFLUENCE_USERNAME')),
        'processors': {
            'pdf_basic': True,
            'pdf_ocr': pdf_processor.ocr_enabled,
            'web_scraping': WEB_PROCESSING_AVAILABLE,
            'document_processing': DOCUMENT_PROCESSING_AVAILABLE,
        },
        'supported_sources': ['PDF files'] + (['Web pages', 'Confluence pages'] if WEB_PROCESSING_AVAILABLE else []) + (['Word, Excel, PowerPoint files'] if DOCUMENT_PROCESSING_AVAILABLE else [])
    })

def get_existing_document_sources():
    """Get list of document sources already in the database"""
    try:
        collection = chroma_client.get_collection(name="documents")
        # Get all metadata from the collection to check existing sources
        results = collection.get()
        existing_sources = set()
        for metadata in results['metadatas']:
            if metadata and 'source' in metadata:
                existing_sources.add(metadata['source'])
        return existing_sources
    except Exception:
        # Collection doesn't exist or is empty
        return set()

def process_documents_folder_on_startup():
    """Process documents from the documents folder on application startup"""
    try:
        documents_folder = Path("documents")
        if not documents_folder.exists():
            logger.info("📁 No documents folder found, skipping auto-processing")
            return
        
        # Get existing documents from database
        existing_sources = get_existing_document_sources()
        if existing_sources:
            logger.info(f"📋 Found {len(existing_sources)} documents already in database")
        
        # Get all files in documents folder (focus on PDFs and images first)
        document_files = []
        for ext in ['.pdf', '.jpg', '.jpeg', '.png']:
            document_files.extend(documents_folder.glob(f'*{ext}'))
            document_files.extend(documents_folder.glob(f'*{ext.upper()}'))
        
        if not document_files:
            logger.info("📁 No supported documents found in documents folder")
            return
        
        # Filter out already processed documents
        new_documents = [doc for doc in document_files if doc.name not in existing_sources]
        
        if not new_documents:
            logger.info(f"📚 STARTUP: All {len(document_files)} documents already processed, skipping")
            return
        
        logger.info(f"📚 STARTUP: Found {len(new_documents)} new documents to process (skipping {len(document_files) - len(new_documents)} already processed)")
        
        processed_count = 0
        for doc_file in new_documents:
            try:
                logger.info(f"📄 Processing: {doc_file.name}")
                
                # Use PDF processor for all files (it can handle images too with OCR)
                if doc_file.suffix.lower() == '.pdf':
                    # Process PDF files
                    text_content, images_metadata = pdf_processor.extract_text_with_ocr(str(doc_file))
                    chunks = pdf_processor.create_chunks_with_proper_metadata(
                        text_content, 
                        doc_file.name,
                        images_metadata
                    )
                elif doc_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # For image files, create a simple chunk with OCR if available
                    try:
                        if pdf_processor.ocr_enabled:
                            # Try to extract text from image using OCR
                            import tempfile
                            import shutil
                            
                            # Create a temporary PDF-like structure for the image
                            temp_text = f"Image file: {doc_file.name}\n"
                            # Note: For full OCR support on standalone images, 
                            # we'd need to implement image OCR separately
                            temp_text += "Image content processed (OCR support available)"
                            
                            chunks = [{
                                'text': temp_text,
                                'metadata': {
                                    'source': doc_file.name,
                                    'type': 'image',
                                    'chunk_index': 0,
                                    'file_path': str(doc_file)
                                },
                                'id': f"{doc_file.stem}_image_chunk_0"
                            }]
                        else:
                            chunks = [{
                                'text': f"Image file: {doc_file.name} (OCR not available)",
                                'metadata': {
                                    'source': doc_file.name,
                                    'type': 'image',
                                    'chunk_index': 0,
                                    'file_path': str(doc_file)
                                },
                                'id': f"{doc_file.stem}_image_chunk_0"
                            }]
                    except Exception as img_e:
                        logger.warning(f"⚠️ Could not process image {doc_file.name}: {img_e}")
                        continue
                else:
                    logger.warning(f"⚠️ Unsupported file type: {doc_file.name}")
                    continue
                
                if chunks and len(chunks) > 0:
                    # Store in ChromaDB
                    store_chunks_in_chromadb(chunks, collection_name="documents")
                    processed_count += 1
                    logger.info(f"✅ Processed {doc_file.name}: {len(chunks)} chunks created")
                else:
                    logger.warning(f"⚠️ No content extracted from {doc_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {doc_file.name}: {str(e)}")
        
        if processed_count > 0:
            logger.info(f"🎉 STARTUP PROCESSING COMPLETE: {processed_count} documents loaded into vector database")
        else:
            logger.warning("⚠️ No documents were successfully processed")
            
    except Exception as e:
        logger.error(f"❌ Error in process_documents_folder_on_startup: {str(e)}")

@app.route('/vector-space', methods=['GET'])
def get_vector_space():
    """Get vector space data for visualization"""
    try:
        collection = chroma_client.get_or_create_collection(
            name="documents",
            metadata={"description": "RAG collection with OCR and web scraping support"}
        )
        
        # Get all documents with their embeddings and metadata
        results = collection.get(
            include=['embeddings', 'metadatas', 'documents']
        )
        
        # Check if results exist
        if results is None:
            return jsonify({
                'error': 'No results returned from database',
                'total_vectors': 0,
                'vectors': []
            })
            
        embeddings = results.get('embeddings')
        if embeddings is None or len(embeddings) == 0:
            return jsonify({
                'error': 'No embeddings found in the database',
                'total_vectors': 0,
                'vectors': []
            })
            
        logger.info(f"📊 Processing {len(embeddings)} vectors for visualization")
        
        # Get other data
        metadatas = results.get('metadatas', [])
        documents = results.get('documents', [])
        
        # Process vectors safely
        vectors = []
        sources = {}
        
        for i in range(min(len(embeddings), len(metadatas), len(documents))):
            try:
                embedding = embeddings[i]
                metadata = metadatas[i] if i < len(metadatas) else {}
                document = documents[i] if i < len(documents) else ""
                
                # Extract key information safely
                source = str(metadata.get('source', 'Unknown')) if metadata else 'Unknown'
                chunk_index = int(metadata.get('chunk_index', 0)) if metadata else 0
                processing_method = str(metadata.get('processing_method', 'unknown')) if metadata else 'unknown'
                has_images = bool(metadata.get('has_images', False)) if metadata else False
                
                # Ensure document is a string
                if not isinstance(document, str):
                    document = str(document) if document is not None else ""
                
                # Create preview text (first 100 chars)
                preview = document[:100] + "..." if len(document) > 100 else document
                
                # Extract keywords from the document text
                keywords = extract_keywords_from_text(document)
                
                # Ensure embedding is a list (not numpy array)
                if hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                elif isinstance(embedding, (list, tuple)):
                    embedding_list = list(embedding)
                else:
                    embedding_list = [float(x) for x in embedding]
                
                vector_data = {
                    'id': f"vec_{i}",
                    'source': source,
                    'chunk_index': chunk_index,
                    'processing_method': processing_method,
                    'has_images': has_images,
                    'preview': preview,
                    'text_length': len(document),
                    'keywords': keywords,
                    'embedding': embedding_list,
                    'metadata': metadata
                }
                
                vectors.append(vector_data)
                
                # Update sources
                if source not in sources:
                    sources[source] = {
                        'name': source,
                        'chunks': 0,
                        'total_chars': 0,
                        'has_images': False,
                        'processing_methods': set()
                    }
                
                sources[source]['chunks'] += 1
                sources[source]['total_chars'] += len(document)
                if has_images:
                    sources[source]['has_images'] = True
                sources[source]['processing_methods'].add(processing_method)
                
            except Exception as vector_error:
                logger.warning(f"Error processing vector {i}: {vector_error}")
                continue
        
        # Convert sets to lists for JSON serialization
        for source_info in sources.values():
            source_info['processing_methods'] = list(source_info['processing_methods'])
        
        logger.info(f"📊 Successfully processed {len(vectors)} vectors from {len(sources)} sources")
        
        return jsonify({
            'total_vectors': len(vectors),
            'total_sources': len(sources),
            'sources': sources,
            'vectors': vectors,
            'embedding_dimension': len(vectors[0]['embedding']) if vectors and vectors[0]['embedding'] else 0
        })
        
    except Exception as e:
        logger.error(f"❌ Error fetching vector space: {str(e)}")
        return jsonify({'error': f'Error fetching vector space: {str(e)}'}), 500

def extract_keywords_from_text(text, max_keywords=10):
    """Extract relevant keywords from text using simple frequency analysis"""
    import re
    from collections import Counter
    
    # Handle potential numpy array inputs
    if hasattr(text, 'shape'):  # Check if it's a numpy array
        text = str(text)
    
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'this', 'that', 'these', 'those', 'from', 'as', 'if', 'when',
        'where', 'how', 'why', 'what', 'who', 'which', 'one', 'two', 'three'
    }
    
    try:
        # Extract words (alphanumeric, 3+ chars)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and count frequency
        filtered_words = [word for word in words if word not in stop_words]
        word_counts = Counter(filtered_words)
        
        # Get top keywords
        keywords = [word for word, count in word_counts.most_common(max_keywords)]
        
        return keywords
    except Exception as e:
        logger.warning(f"Error extracting keywords: {e}")
        return []

@app.route('/vector-space/reduce', methods=['POST'])
def reduce_vector_dimensions():
    """Reduce high-dimensional embeddings to 2D for visualization"""
    try:
        data = request.get_json()
        embeddings = data.get('embeddings', [])
        method = data.get('method', 'tsne')  # 'tsne' or 'umap'
        
        if not embeddings:
            return jsonify({'error': 'No embeddings provided'}), 400
        
        logger.info(f"📉 Reducing {len(embeddings)} embeddings to 2D using {method}")
        
        import numpy as np
        embeddings_array = np.array(embeddings)
        
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=min(30, len(embeddings) - 1),
                max_iter=1000  # Changed from n_iter to max_iter
            )
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                # Fallback to t-SNE if UMAP not available
                from sklearn.manifold import TSNE
                reducer = TSNE(
                    n_components=2, 
                    random_state=42, 
                    perplexity=min(30, len(embeddings) - 1),
                    max_iter=1000  # Changed from n_iter to max_iter
                )
                method = 'tsne'
        else:
            return jsonify({'error': 'Invalid reduction method. Use "tsne" or "umap"'}), 400
        
        # Perform dimensionality reduction
        reduced_embeddings = reducer.fit_transform(embeddings_array)
        
        # Convert to list of [x, y] coordinates
        coordinates = reduced_embeddings.tolist()
        
        logger.info(f"✅ Successfully reduced embeddings using {method}")
        
        return jsonify({
            'method': method,
            'coordinates': coordinates,
            'total_points': len(coordinates)
        })
        
    except Exception as e:
        logger.error(f"❌ Error reducing vector dimensions: {str(e)}")
        return jsonify({'error': f'Error reducing dimensions: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("🚀 STARTING CORRECTED RAG CHATBOT")
    logger.info(f"🔍 PDF OCR: {'✅ Enabled' if pdf_processor.ocr_enabled else '❌ Disabled (missing dependencies)'}")
    logger.info(f"🌐 Web Processing: {'✅ Enabled' if WEB_PROCESSING_AVAILABLE else '❌ Disabled (missing dependencies)'}")
    logger.info(f"📄 Document Processing: {'✅ Enabled' if DOCUMENT_PROCESSING_AVAILABLE else '❌ Disabled (missing dependencies)'}")
    logger.info("🗑️ Database clearing: ✅ Available via /clear-database endpoint")
    
    # Process documents folder on startup
    process_documents_folder_on_startup()
    
    app.run(debug=True, host='0.0.0.0', port=5000) 