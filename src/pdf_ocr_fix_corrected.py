import os
import logging
from pathlib import Path
from typing import List, Dict
import io
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional imports - fallback gracefully if not available
try:
    import fitz  # PyMuPDF - better for images and OCR
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF not available, using PyPDF2 only (no image OCR)")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
    
    # Configure Tesseract path for Windows
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",  # Linux
        "/opt/homebrew/bin/tesseract"  # macOS
    ]
    
    # Try to find and set Tesseract executable path
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR libraries not available (pytesseract/PIL)")

logger = logging.getLogger(__name__)

class EnhancedPDFProcessor:
    """Enhanced PDF processor with OCR support for images within PDFs"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, images_folder: str = "extracted_images"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Set up images folder
        self.images_folder = Path(images_folder)
        self.images_folder.mkdir(exist_ok=True)
        
        # Check available features
        self.ocr_enabled = PYMUPDF_AVAILABLE and OCR_AVAILABLE
        if self.ocr_enabled:
            logger.info("✅ OCR ENABLED: PyMuPDF + Tesseract available")
        else:
            logger.warning("⚠️ OCR DISABLED: Missing PyMuPDF or Tesseract")
    
    def extract_text_with_ocr(self, pdf_path: str) -> tuple[str, List[Dict]]:
        """Extract text from PDF including OCR on embedded images, return text and image metadata"""
        logger.info(f"📖 PDF PROCESSING: {pdf_path}")
        
        if self.ocr_enabled:
            return self._extract_with_pymupdf_ocr(pdf_path)
        else:
            return self._extract_with_pypdf2_only(pdf_path), []
    
    def _extract_with_pymupdf_ocr(self, pdf_path: str) -> tuple[str, List[Dict]]:
        """Extract with PyMuPDF and OCR (enhanced method), return text and image metadata"""
        full_text = ""
        images_metadata = []  # Track all extracted images
        ocr_warned = False  # Track if we've already warned about OCR issues
        total_images = 0
        successful_ocr = 0
        
        try:
            pdf_document = fitz.open(pdf_path)
            logger.info(f"📄 PDF has {len(pdf_document)} pages")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Reduced verbosity - only log every 10th page
                if (page_num + 1) % 10 == 0 or page_num == 0:
                    logger.info(f"  📄 Processing page {page_num + 1}/{len(pdf_document)}...")
                
                # Extract regular text
                page_text = page.get_text()
                full_text += f"\n--- PAGE {page_num + 1} ---\n"
                full_text += page_text
                
                # Extract and OCR images on this page
                image_list = page.get_images()
                
                if image_list:
                    # Only log image count for pages with many images or first occurrence
                    if len(image_list) > 5 or (page_num < 3 and image_list):
                        logger.info(f"    🖼️ Found {len(image_list)} images on page {page_num + 1}")
                    total_images += len(image_list)
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Extract image
                            xref = img[0]
                            pix = fitz.Pixmap(pdf_document, xref)
                            
                            # Convert to PIL Image for OCR
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_data = pix.tobytes("ppm")
                                pil_image = Image.open(io.BytesIO(img_data))
                                
                                # Perform OCR on the image
                                ocr_text = pytesseract.image_to_string(pil_image)
                                
                                # Create image metadata regardless of OCR success
                                pdf_filename = Path(pdf_path).stem
                                img_filename = f"pdf_{pdf_filename}_page{page_num + 1}_img{img_index + 1}.png"
                                img_path = self.images_folder / img_filename
                                
                                image_metadata = {
                                    'page_number': page_num + 1,
                                    'image_index': img_index + 1,
                                    'filename': img_filename,
                                    'local_path': str(img_path),
                                    'ocr_text': ocr_text.strip(),
                                    'has_ocr': bool(ocr_text.strip()),
                                    'width': pil_image.width,
                                    'height': pil_image.height,
                                    'format': pil_image.format or 'PNG'
                                }
                                
                                if ocr_text.strip():  # Only add if OCR found text
                                    logger.info(f"      🔍 OCR extracted from image {img_index + 1}: {len(ocr_text)} chars")
                                    logger.info(f"      📝 OCR Preview: '{ocr_text[:100]}...'")
                                    
                                    # Save the image file for display in web interface
                                    try:
                                        pil_image.save(img_path, 'PNG')
                                        logger.info(f"      💾 Saved image: {img_filename}")
                                    except Exception as save_e:
                                        logger.warning(f"      ⚠️ Could not save image: {save_e}")
                                    
                                    full_text += f"\n--- IMAGE {img_index + 1} OCR (Page {page_num + 1}) ---\n"
                                    full_text += ocr_text
                                    full_text += f"\n--- END IMAGE {img_index + 1} OCR ---\n"
                                    successful_ocr += 1
                                else:
                                    # Still save the image even if no OCR text
                                    try:
                                        pil_image.save(img_path, 'PNG')
                                        logger.info(f"      💾 Saved image (no OCR): {img_filename}")
                                    except Exception as save_e:
                                        logger.warning(f"      ⚠️ Could not save image: {save_e}")
                                
                                images_metadata.append(image_metadata)
                            
                            pix = None  # Free memory
                            
                        except Exception as e:
                            # Only show warning once per document for Tesseract issues
                            if not ocr_warned and "tesseract" in str(e).lower():
                                logger.warning(f"⚠️ OCR not available: {e}")
                                logger.info(f"ℹ️ Found {total_images} images but OCR is disabled. Install Tesseract for image text extraction.")
                                ocr_warned = True
                            elif "tesseract" not in str(e).lower():
                                # Show individual warnings for non-Tesseract errors
                                logger.warning(f"      ⚠️ Failed to OCR image {img_index + 1}: {e}")
                            continue
                
                full_text += f"\n--- END PAGE {page_num + 1} ---\n\n"
            
            pdf_document.close()
            
            # Show OCR summary
            if total_images > 0:
                if successful_ocr > 0:
                    logger.info(f"🔍 OCR SUMMARY: Successfully processed {successful_ocr}/{total_images} images")
                elif not ocr_warned:
                    logger.info(f"ℹ️ OCR SUMMARY: Found {total_images} images but OCR is not available")
            
        except Exception as e:
            logger.error(f"Error in PyMuPDF extraction: {e}")
            logger.info("Falling back to PyPDF2...")
            return self._extract_with_pypdf2_only(pdf_path), []
        
        logger.info(f"📝 TOTAL EXTRACTED: {len(full_text)} characters, {len(images_metadata)} images")
        return full_text, images_metadata
    
    def _extract_with_pypdf2_only(self, pdf_path: str) -> tuple[str, List[Dict]]:
        """Fallback extraction with PyPDF2 only (no OCR)"""
        full_text = ""
        image_metadata_list = []
        
        try:
            logger.info("📖 Using PyPDF2 extraction (no OCR)")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    full_text += f"\n--- PAGE {page_num + 1} ---\n"
                    full_text += page.extract_text()
                    full_text += f"\n--- END PAGE {page_num + 1} ---\n\n"
                    
        except Exception as e:
            logger.error(f"Error in PyPDF2 extraction: {e}")
            raise
        
        logger.info(f"📝 TOTAL EXTRACTED: {len(full_text)} characters")
        return full_text, image_metadata_list
    
    def create_chunks_with_proper_metadata(self, text: str, original_filename: str, images_metadata: List[Dict] = None) -> List[Dict]:
        """Create chunks with proper source attribution and image associations"""
        
        if images_metadata is None:
            images_metadata = []
        
        # Preserve original filename for source attribution
        logger.info(f"✂️ CHUNKING: '{original_filename}' -> {len(text)} characters, {len(images_metadata)} images")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"📦 CREATED: {len(chunks)} chunks")
        
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            processing_method = 'enhanced_ocr' if self.ocr_enabled else 'basic_text_only'
            
            # Find images relevant to this chunk
            associated_images = self._find_relevant_images_for_chunk(chunk, images_metadata)
            
            chunk_objects.append({
                'id': f"{original_filename}_chunk_{i}",  # More descriptive ID
                'text': chunk,
                'metadata': {
                    'source': original_filename,  # Preserve EXACT original filename
                    'file_type': 'pdf',
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'processing_method': processing_method,
                    'ocr_enabled': self.ocr_enabled,
                    'has_images': len(associated_images) > 0,
                    'images_count': len(associated_images)
                },
                'images': associated_images  # Associated images for this chunk
            })
            
            # Log first few chunks for debugging
            if i < 3:
                logger.info(f"    📦 Chunk {i+1}: {len(chunk)} chars | Images: {len(associated_images)} | Source: '{original_filename}' | Preview: '{chunk[:100]}...'")
        
        return chunk_objects
    
    def _find_relevant_images_for_chunk(self, chunk_text: str, images_metadata: List[Dict]) -> List[Dict]:
        """Find images that are relevant to a specific text chunk"""
        relevant_images = []
        
        # Extract page numbers mentioned in the chunk
        chunk_pages = set()
        lines = chunk_text.split('\n')
        
        for line in lines:
            # Look for page markers
            if line.startswith('--- PAGE ') and line.endswith(' ---'):
                try:
                    page_num = int(line.replace('--- PAGE ', '').replace(' ---', ''))
                    chunk_pages.add(page_num)
                except ValueError:
                    continue
            # Look for image OCR markers that include page numbers
            elif 'IMAGE' in line and 'OCR' in line and 'Page' in line:
                try:
                    # Extract page number from "--- IMAGE X OCR (Page Y) ---"
                    if '(Page ' in line and ')' in line:
                        page_part = line.split('(Page ')[1].split(')')[0]
                        page_num = int(page_part)
                        chunk_pages.add(page_num)
                except (ValueError, IndexError):
                    continue
        
        # Find images from the same pages as the chunk content
        for image_meta in images_metadata:
            image_page = image_meta.get('page_number', 0)
            
            if image_page in chunk_pages:
                # Check if the image's OCR text is actually in this chunk
                image_ocr = image_meta.get('ocr_text', '')
                if image_ocr and len(image_ocr.strip()) > 10:  # Only if substantial OCR text
                    # Simple check: if part of the image OCR text is in the chunk
                    ocr_words = image_ocr.split()[:5]  # First 5 words of OCR
                    if any(word.lower() in chunk_text.lower() for word in ocr_words if len(word) > 3):
                        relevant_images.append(image_meta)
                elif not image_ocr:
                    # If no OCR text, include all images from the same page
                    relevant_images.append(image_meta)
        
        return relevant_images

# Test function to debug your specific case
def debug_pdf_processing(pdf_path: str):
    """Debug function to test OCR and source attribution"""
    processor = EnhancedPDFProcessor()
    
    logger.info(f"🔍 DEBUGGING PDF: {pdf_path}")
    logger.info(f"🔧 OCR Enabled: {processor.ocr_enabled}")
    
    # Extract with OCR
    text, image_metadata = processor.extract_text_with_ocr(pdf_path)
    
    # Show extracted text preview
    logger.info(f"📝 EXTRACTED TEXT PREVIEW:")
    logger.info(f"First 500 chars: '{text[:500]}...'")
    
    # Look for potential names/quotes
    lines = text.split('\n')
    for i, line in enumerate(lines[:50]):  # Check first 50 lines
        if any(indicator in line.lower() for indicator in ['said', 'quote', 'mentioned', ':', '"']):
            logger.info(f"🎯 POTENTIAL QUOTE/NAME (Line {i}): '{line.strip()}'")
    
    # Display image metadata if available
    if image_metadata:
        logger.info(f"\n📸 IMAGE METADATA:")
        for img_meta in image_metadata:
            logger.info(f"  - Page: {img_meta['page_number']}, Image Index: {img_meta['image_index']}, OCR Text Length: {len(img_meta['ocr_text'])}")
    
    return text

if __name__ == "__main__":
    # Test with your SAFe PDF
    logging.basicConfig(level=logging.INFO)
    pdf_path = "./uploads/Essential-SAFe-4.6-Overview-and-Assessment.pdf"  # Adjust path
    
    if os.path.exists(pdf_path):
        debug_pdf_processing(pdf_path)
    else:
        print(f"PDF not found at: {pdf_path}")
        print("Please adjust the path to your PDF file") 