# 🤖 RAG Chatbot - Complete Solution

A production-ready **Retrieval-Augmented Generation (RAG) chatbot** that combines ChromaDB vector database with OpenAI's GPT-4 for intelligent document processing and conversational AI.

## ✨ Features

- 🔍 **Advanced PDF Processing** with OCR for images and tables
- 🌐 **Web Content Processing** including Confluence pages
- 📄 **Multi-format Support** (PDF, Word, PowerPoint, Excel, Images)
- 🗃️ **Vector Database** powered by ChromaDB
- 💬 **ChatGPT-like Interface** with source attribution
- 🚀 **One-Click Startup** with Windows batch files
- 🗑️ **Database Management** with easy clearing and status checking
- 📊 **Real-time Debugging** with comprehensive logging

## 🚀 Quick Start

### 1. **First Time Setup**
```bash
# Double-click to run full setup and validation
start_rag_chatbot.bat
```

### 2. **Daily Use**
```bash
# Double-click for quick startup
quick_start.bat
```

### 3. **Database Management**
```bash
# Double-click for interactive database management
manage_database.bat
```

## 📁 Project Structure

```
RAG_Chatbot/
├── 📱 app_corrected_final.py          # Main Flask application
├── 🔧 requirements_ultimate_fixed.txt  # Python dependencies
├── 🔑 .env                            # Environment variables (API keys)
├── 
├── 🚀 start_rag_chatbot.bat          # Complete setup & startup
├── ⚡ quick_start.bat                # Fast startup
├── 🗑️ manage_database.bat            # Database management
├── 
├── 📁 src/                           # Core modules
│   ├── config.py                     # Configuration settings
│   ├── pdf_ocr_fix_corrected.py      # Advanced PDF processing
│   ├── web_confluence_processor.py   # Web scraping & Confluence
│   └── document_processor.py         # Multi-format document handling
├── 
├── 📁 scripts/                       # Utility scripts
│   ├── clear_database.py             # Database management CLI
│   └── system_check.py               # System diagnostics
├── 
├── 📁 docs/                          # Documentation
│   ├── BATCH_FILES_GUIDE.md          # Batch file usage guide
│   └── FIXED_SETUP_GUIDE.md          # Setup troubleshooting
├── 
├── 📁 templates/                     # HTML templates
├── 📁 uploads/                       # Uploaded files storage
├── 📁 chroma_db/                     # Vector database storage
├── 📁 extracted_images/              # OCR processed images
└── 📁 data/                          # Runtime data files
```

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Tesseract OCR** (Optional, for image text extraction)

### Automatic Setup
1. **Download/Clone** this repository
2. **Run**: `start_rag_chatbot.bat`
3. **Edit** the created `.env` file with your API key:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```
4. **Run**: `start_rag_chatbot.bat` again

### Manual Setup
```bash
# Install dependencies
pip install -r requirements_ultimate_fixed.txt

# Create environment file
echo OPENAI_API_KEY=sk-your-key-here > .env

# Start the application
python app_corrected_final.py
```

## 🎯 Usage

### Upload Documents
1. **Start** the application: `quick_start.bat`
2. **Visit**: `http://localhost:5000`
3. **Upload** PDF, Word, PowerPoint, Excel files or images
4. **Process** web URLs or Confluence pages

### Chat with Documents
1. **Ask questions** about your uploaded content
2. **Get responses** with source attribution
3. **View images** extracted from documents
4. **See debug logs** for transparency

### Database Management
1. **Run**: `manage_database.bat`
2. **Choose option**:
   - `1` - Show database status
   - `2` - Clear entire database
   - `3` - Verify cleanup
   - `4` - Exit

## 🔧 Configuration

### Environment Variables (`.env`)
```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional - For Confluence support
CONFLUENCE_USERNAME=your-email@company.com
CONFLUENCE_API_TOKEN=your-confluence-api-token
```

### Application Settings (`src/config.py`)
```python
# RAG Configuration
CHUNK_SIZE = 1000          # Text chunk size
CHUNK_OVERLAP = 200        # Overlap between chunks
RETRIEVAL_RESULTS = 5      # Number of results to retrieve

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o"    # Model to use
MAX_TOKENS = 1500          # Max response tokens
TEMPERATURE = 0.7          # Response creativity
```

## 📊 API Endpoints

### Core Endpoints
- `GET /` - Main web interface
- `POST /upload` - Upload documents
- `POST /chat` - Chat with documents
- `POST /process-url` - Process web URLs

### Management Endpoints
- `GET /health` - System health check
- `GET /database-status` - Database information
- `POST /clear-database` - Clear all data
- `GET /processed-files` - List processed files

### Debug Endpoints
- `GET /debug-sources` - Debug source attribution
- `POST /test-ocr` - Test OCR functionality

## 🛠️ Troubleshooting

### Common Issues

**"Python not found"**
- Install Python 3.8+ and ensure it's in PATH

**"Dependencies failed to install"**
```bash
pip install --upgrade pip
pip install -r requirements_ultimate_fixed.txt
```

**"OpenAI API key not configured"**
- Check your `.env` file exists and contains: `OPENAI_API_KEY=sk-...`

**"Port 5000 already in use"**
- Close other applications or edit `app_corrected_final.py` to use different port

**"OCR not working"**
- Install Tesseract OCR: [Download here](https://github.com/UB-Mannheim/tesseract/wiki)

### Debug Mode
```bash
# Enable detailed logging
set FLASK_DEBUG=1
python app_corrected_final.py
```

## 🧪 Testing

### System Check
```bash
python scripts\system_check.py
```

### Health Check
```bash
curl http://localhost:5000/health
```

### Database Status
```bash
curl http://localhost:5000/database-status
```

## 🏗️ Architecture

### RAG Pipeline
1. **Document Upload** → Text extraction with OCR
2. **Text Chunking** → Semantic splitting with overlap
3. **Embedding Creation** → ChromaDB vector storage
4. **Query Processing** → Semantic search + context building
5. **Response Generation** → OpenAI GPT-4 with context

### Key Components
- **Flask Backend** - REST API and web server
- **ChromaDB** - Vector database for semantic search
- **PyMuPDF + Tesseract** - Advanced PDF processing with OCR  
- **BeautifulSoup + Requests** - Web scraping capabilities
- **OpenAI API** - GPT-4 for response generation

## 📈 Performance

### Optimization Features
- **Graceful Fallbacks** - Works without optional dependencies
- **Duplicate Detection** - Prevents reprocessing same files
- **Chunking Strategy** - Preserves semantic meaning
- **Image Caching** - Efficient image storage and retrieval
- **Connection Pooling** - Optimized database connections

## 🔒 Security

- **API Key Protection** - Environment variable storage
- **File Validation** - Secure upload handling
- **Path Sanitization** - Prevents directory traversal
- **Error Handling** - No sensitive data in error messages

## 📝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature-name`
3. **Commit** changes: `git commit -am 'Add feature'`
4. **Push** to branch: `git push origin feature-name`
5. **Submit** pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ChromaDB** - Vector database
- **OpenAI** - GPT-4 language model
- **PyMuPDF** - PDF processing
- **Tesseract** - OCR engine
- **Flask** - Web framework

---

## 🎉 Ready to Go!

Your RAG chatbot is now **production-ready**! 

**Start chatting**: `quick_start.bat` → `http://localhost:5000`

For questions or issues, check the `docs/` folder or create an issue in the repository.

**Happy chatting! 🤖💬** 