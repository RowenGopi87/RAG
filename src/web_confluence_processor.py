import os
import requests
from bs4 import BeautifulSoup
import base64
import hashlib
import json
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import tempfile

# OCR and image processing
from PIL import Image
import pytesseract
import io

# Text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class WebConfluenceProcessor:
    """
    Enhanced web and Confluence content processor with intelligent image-text association.
    
    Features:
    - OCR processing of images from web pages
    - Intelligent image-text association based on content similarity
    - Support for both regular web pages and Confluence pages
    - Table extraction and processing
    - Relevance scoring for image associations
    - Web images saved as local files for full display compatibility
    
    Web images are now stored as local files (like PDF images) for complete display support.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 images_folder: str = "extracted_images",
                 confluence_auth: Optional[Tuple[str, str]] = None):
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Create folder for storing extracted images
        self.images_folder = Path(images_folder)
        self.images_folder.mkdir(exist_ok=True)
        
        # Session for web requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Confluence authentication if provided
        if confluence_auth:
            self.session.auth = confluence_auth
            logger.info("🔐 Confluence authentication configured")
    
    def is_confluence_url(self, url: str) -> bool:
        """Check if URL is a Confluence page"""
        confluence_indicators = [
            '/wiki/', '/display/', 'confluence', 
            '/pages/viewpage.action', '/spaces/'
        ]
        return any(indicator in url.lower() for indicator in confluence_indicators)
    
    def extract_web_content(self, url: str) -> Dict:
        """Extract content from any web URL including Confluence"""
        try:
            logger.info(f"🌐 SCRAPING URL: {url}")
            
            # Check if it's Confluence
            is_confluence = self.is_confluence_url(url)
            if is_confluence:
                logger.info("📚 CONFLUENCE PAGE DETECTED")
            
            # Get the page
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            title = self._extract_title(soup, url)
            
            # Extract main content
            main_text = self._extract_text_content(soup, is_confluence)
            
            # Extract and process images
            images_data = self._extract_and_process_images(soup, url)
            
            # Extract tables
            tables_data = self._extract_tables(soup)
            
            # Combine all content
            full_content = self._combine_content(main_text, tables_data, images_data)
            
            result = {
                'url': url,
                'title': title,
                'text_content': main_text,
                'full_content': full_content,  # Text + OCR + Tables
                'images': images_data,
                'tables': tables_data,
                'is_confluence': is_confluence,
                'total_characters': len(full_content),
                'images_count': len(images_data),
                'tables_count': len(tables_data)
            }
            
            logger.info(f"✅ EXTRACTED: {len(main_text)} chars text, {len(images_data)} images, {len(tables_data)} tables")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error extracting from {url}: {e}")
            raise
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title"""
        # Try various title sources
        title_selectors = [
            'h1.page-title',  # Confluence
            'h1#title-text',  # Confluence alternative
            'title',          # HTML title
            'h1',             # First h1
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                if title:
                    return title
        
        # Fallback to URL
        return urlparse(url).path.split('/')[-1] or url
    
    def _extract_text_content(self, soup: BeautifulSoup, is_confluence: bool) -> str:
        """Extract main text content"""
        if is_confluence:
            # Confluence-specific content selectors
            content_selectors = [
                '#main-content .wiki-content',
                '.wiki-content',
                '#content',
                '.page-content'
            ]
        else:
            # General web page selectors
            content_selectors = [
                'main',
                'article',
                '.content',
                '#content',
                'body'
            ]
        
        # Remove unwanted elements
        for unwanted in soup(['script', 'style', 'nav', 'footer', 'header']):
            unwanted.decompose()
        
        # Try to find main content
        main_content = None
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _extract_and_process_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract images and perform OCR"""
        images_data = []
        
        img_tags = soup.find_all('img')
        logger.info(f"🖼️ FOUND {len(img_tags)} images on page")
        
        for i, img in enumerate(img_tags):
            try:
                img_src = img.get('src')
                if not img_src:
                    continue
                
                # Handle relative URLs
                img_url = urljoin(base_url, img_src)
                
                logger.info(f"  📷 Processing image {i+1}: {img_url}")
                
                # Download image
                img_response = self.session.get(img_url, timeout=15)
                img_response.raise_for_status()
                
                # Open image with PIL
                pil_image = Image.open(io.BytesIO(img_response.content))
                
                # Generate unique filename
                img_hash = hashlib.md5(img_response.content).hexdigest()[:8]
                img_filename = f"web_image_{i+1}_{img_hash}.png"
                img_path = self.images_folder / img_filename
                
                # Save image
                pil_image.save(img_path, 'PNG')
                
                # Perform OCR
                ocr_text = pytesseract.image_to_string(pil_image)
                
                # Get image metadata
                alt_text = img.get('alt', '')
                title_text = img.get('title', '')
                
                # Convert to base64 for storage
                img_base64 = base64.b64encode(img_response.content).decode('utf-8')
                
                image_data = {
                    'index': i + 1,
                    'url': img_url,
                    'filename': img_filename,
                    'local_path': str(img_path),
                    'alt_text': alt_text,
                    'title_text': title_text,
                    'ocr_text': ocr_text.strip(),
                    'base64_data': img_base64,
                    'width': pil_image.width,
                    'height': pil_image.height,
                    'format': pil_image.format or 'Unknown'
                }
                
                images_data.append(image_data)
                
                if ocr_text.strip():
                    logger.info(f"    🔍 OCR extracted: {len(ocr_text)} chars")
                    logger.info(f"    📝 OCR preview: '{ocr_text[:100]}...'")
                else:
                    logger.info(f"    ⚪ No text found in image")
                
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process image {i+1}: {e}")
                continue
        
        return images_data
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract tables from the page"""
        tables_data = []
        
        tables = soup.find_all('table')
        logger.info(f"📊 FOUND {len(tables)} tables on page")
        
        for i, table in enumerate(tables):
            try:
                rows_data = []
                rows = table.find_all('tr')
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    if row_data:  # Only add non-empty rows
                        rows_data.append(row_data)
                
                if rows_data:
                    # Convert table to text representation
                    table_text = self._table_to_text(rows_data)
                    
                    table_data = {
                        'index': i + 1,
                        'rows_count': len(rows_data),
                        'columns_count': max(len(row) for row in rows_data) if rows_data else 0,
                        'data': rows_data,
                        'text_representation': table_text
                    }
                    
                    tables_data.append(table_data)
                    logger.info(f"  📋 Table {i+1}: {len(rows_data)} rows, {table_data['columns_count']} columns")
                    
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to process table {i+1}: {e}")
                continue
        
        return tables_data
    
    def _table_to_text(self, rows_data: List[List[str]]) -> str:
        """Convert table data to readable text"""
        if not rows_data:
            return ""
        
        text_lines = []
        for i, row in enumerate(rows_data):
            if i == 0:  # Header row
                text_lines.append("TABLE HEADERS: " + " | ".join(row))
                text_lines.append("-" * 50)
            else:
                text_lines.append("ROW: " + " | ".join(row))
        
        return "\n".join(text_lines)
    
    def _combine_content(self, main_text: str, tables_data: List[Dict], images_data: List[Dict]) -> str:
        """Combine all extracted content into one text"""
        combined = []
        
        # Main text content
        combined.append("=== MAIN CONTENT ===")
        combined.append(main_text)
        combined.append("")
        
        # Tables
        if tables_data:
            combined.append("=== TABLES ===")
            for table in tables_data:
                combined.append(f"--- TABLE {table['index']} ---")
                combined.append(table['text_representation'])
                combined.append("")
        
        # OCR from images
        ocr_texts = [img['ocr_text'] for img in images_data if img['ocr_text']]
        if ocr_texts:
            combined.append("=== TEXT FROM IMAGES (OCR) ===")
            for i, ocr_text in enumerate(ocr_texts, 1):
                combined.append(f"--- IMAGE {i} OCR ---")
                combined.append(ocr_text)
                combined.append("")
        
        return "\n".join(combined)
    
    def create_chunks_with_images(self, content_data: Dict, source_url: str) -> List[Dict]:
        """Create chunks with image references"""
        full_content = content_data['full_content']
        images_data = content_data['images']
        
        logger.info(f"✂️ CHUNKING: {len(full_content)} characters from {source_url}")
        
        # Split content into chunks
        text_chunks = self.text_splitter.split_text(full_content)
        
        chunk_objects = []
        for i, chunk in enumerate(text_chunks):
            # Find relevant images for this chunk
            relevant_images = self._find_relevant_images(chunk, images_data)
            
            chunk_obj = {
                'id': f"{self._url_to_id(source_url)}_chunk_{i}",
                'text': chunk,
                'metadata': {
                    'source': source_url,
                    'title': content_data['title'],
                    'source_type': 'confluence' if content_data['is_confluence'] else 'web',
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'processing_method': 'web_scraping_ocr',
                    'images_count': len(relevant_images),
                    'has_images': len(relevant_images) > 0
                },
                'images': relevant_images  # Include relevant images with chunk
            }
            
            chunk_objects.append(chunk_obj)
            
            # Log first few chunks
            if i < 3:
                logger.info(f"    📦 Chunk {i+1}: {len(chunk)} chars | Images: {len(relevant_images)} | Preview: '{chunk[:100]}...'")
        
        logger.info(f"📦 CREATED: {len(chunk_objects)} chunks with image references")
        return chunk_objects
    
    def _find_relevant_images(self, chunk_text: str, images_data: List[Dict]) -> List[Dict]:
        """Find images that are truly relevant to the current chunk - enhanced logic"""
        relevant_images = []
        chunk_text_lower = chunk_text.lower()
        
        # Get meaningful words from chunk (filter out common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'from', 'as', 'if', 'when', 'where', 'how', 'why', 'what', 'who', 'which'}
        chunk_meaningful_words = [word.strip('.,!?;:()[]{}') for word in chunk_text_lower.split() if len(word) > 3 and word not in common_words]
        
        for img in images_data:
            relevance_score = 0
            reasons = []
            
            # Check OCR text similarity (high weight)
            if img['ocr_text'] and len(img['ocr_text'].strip()) > 10:
                ocr_text_lower = img['ocr_text'].lower()
                
                # Exact phrase matching (highest score)
                ocr_phrases = [phrase.strip() for phrase in ocr_text_lower.split('.') if len(phrase.strip()) > 5]
                for phrase in ocr_phrases:
                    if phrase in chunk_text_lower:
                        relevance_score += 10
                        reasons.append(f"OCR phrase match: '{phrase[:50]}...'")
                        break
                
                # Word overlap scoring
                ocr_words = [word.strip('.,!?;:()[]{}') for word in ocr_text_lower.split() if len(word) > 3 and word not in common_words]
                overlap_count = len(set(chunk_meaningful_words) & set(ocr_words))
                if overlap_count > 0:
                    word_overlap_score = min(overlap_count * 2, 8)  # Cap at 8 points
                    relevance_score += word_overlap_score
                    reasons.append(f"OCR word overlap: {overlap_count} words")
            
            # Check alt text (medium weight)
            if img['alt_text'] and len(img['alt_text'].strip()) > 3:
                alt_text_lower = img['alt_text'].lower()
                alt_words = [word.strip('.,!?;:()[]{}') for word in alt_text_lower.split() if len(word) > 3]
                alt_overlap = len(set(chunk_meaningful_words) & set(alt_words))
                if alt_overlap > 0:
                    relevance_score += min(alt_overlap * 1.5, 4)  # Cap at 4 points
                    reasons.append(f"Alt text overlap: {alt_overlap} words")
            
            # Check title text (medium weight)
            if img['title_text'] and len(img['title_text'].strip()) > 3:
                title_text_lower = img['title_text'].lower()
                title_words = [word.strip('.,!?;:()[]{}') for word in title_text_lower.split() if len(word) > 3]
                title_overlap = len(set(chunk_meaningful_words) & set(title_words))
                if title_overlap > 0:
                    relevance_score += min(title_overlap * 1.5, 4)  # Cap at 4 points
                    reasons.append(f"Title text overlap: {title_overlap} words")
            
            # Only include images with meaningful relevance (threshold)
            if relevance_score >= 3:  # Minimum threshold for relevance
                logger.info(f"  📸 Relevant image: {img['filename']} (score: {relevance_score:.1f}) - {', '.join(reasons)}")
                relevant_images.append({
                    'filename': img['filename'],
                    'local_path': img['local_path'],  # Include local path for file-based display
                    'base64_data': img['base64_data'],
                    'ocr_text': img['ocr_text'],
                    'alt_text': img['alt_text'],
                    'title_text': img.get('title_text', ''),
                    'width': img['width'],
                    'height': img['height'],
                    'relevance_score': relevance_score,
                    'relevance_reasons': reasons
                })
            else:
                if relevance_score > 0:  # Log near-misses for debugging
                    logger.debug(f"  ⚪ Low relevance image: {img['filename']} (score: {relevance_score:.1f}) - below threshold")
        
        # Sort by relevance score and return top images
        relevant_images.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Limit to top 5 most relevant images per chunk
        return relevant_images[:5]
    
    def _url_to_id(self, url: str) -> str:
        """Convert URL to safe ID"""
        # Create hash of URL for unique ID
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        # Extract domain for readability
        domain = urlparse(url).netloc.replace('.', '_')
        return f"{domain}_{url_hash}"
    
    def process_url(self, url: str) -> Dict:
        """Main method to process any URL"""
        try:
            logger.info(f"🚀 STARTING WEB PROCESSING: {url}")
            
            # Extract content
            content_data = self.extract_web_content(url)
            
            # Create chunks with images
            chunks = self.create_chunks_with_images(content_data, url)
            
            result = {
                'url': url,
                'title': content_data['title'],
                'chunks': chunks,
                'total_characters': content_data['total_characters'],
                'images_count': content_data['images_count'],
                'tables_count': content_data['tables_count'],
                'chunks_count': len(chunks),
                'processing_method': 'web_scraping_with_ocr'
            }
            
            logger.info(f"✅ WEB PROCESSING COMPLETE: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing URL {url}: {e}")
            raise

# Test function
def test_confluence_processing(url: str, confluence_auth: Optional[Tuple[str, str]] = None):
    """Test web/Confluence processing"""
    processor = WebConfluenceProcessor(confluence_auth=confluence_auth)
    
    logger.info(f"🧪 TESTING URL PROCESSING: {url}")
    
    result = processor.process_url(url)
    
    logger.info(f"📊 RESULTS:")
    logger.info(f"  Title: {result['title']}")
    logger.info(f"  Characters: {result['total_characters']}")
    logger.info(f"  Images: {result['images_count']}")
    logger.info(f"  Tables: {result['tables_count']}")
    logger.info(f"  Chunks: {result['chunks_count']}")
    
    # Show sample chunk with images
    for chunk in result['chunks'][:2]:
        if chunk['images']:
            logger.info(f"🖼️ Chunk with images: {len(chunk['images'])} images")
            logger.info(f"  Preview: {chunk['text'][:200]}...")
    
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample URL
    test_url = "https://example.com"  # Replace with your Confluence URL
    
    # For Confluence with authentication:
    # confluence_auth = ("your_username", "your_api_token")
    # test_confluence_processing(test_url, confluence_auth) 