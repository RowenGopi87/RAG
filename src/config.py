import os

class Config:
    """Configuration class for the RAG Chatbot"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Upload Configuration
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'pdf'}
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    CHROMA_COLLECTION_NAME = "documents"
    
    # RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RETRIEVAL_RESULTS = 5
    
    # OpenAI Model Configuration
    OPENAI_MODEL = "gpt-4o"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    # Timeout Configuration
    OPENAI_TIMEOUT = 30  # 30 seconds timeout for OpenAI API calls
    OPENAI_MAX_RETRIES = 2  # Maximum retries for failed requests 