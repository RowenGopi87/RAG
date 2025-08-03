#!/usr/bin/env python3
"""
System status checker for RAG Chatbot
"""

import sys
import os

def check_system_status():
    """Check and display system status"""
    print("📊 System Status:")
    print(f"   Python: {sys.version.split()[0]}")
    
    # Check PyMuPDF
    try:
        import fitz
        print("   PyMuPDF: ✅ Available (Enhanced PDF + OCR)")
    except ImportError:
        print("   PyMuPDF: ❌ Not available (Basic PDF only)")
    
    # Check Tesseract
    try:
        import pytesseract
        print("   Tesseract: ✅ Available (OCR enabled)")
    except ImportError:
        print("   Tesseract: ❌ Not available (OCR disabled)")
    
    # Check web scraping
    try:
        import requests
        import bs4
        print("   Web Scraping: ✅ Available")
    except ImportError:
        print("   Web Scraping: ❌ Not available")
    
    # Check OpenAI
    api_key = os.getenv('OPENAI_API_KEY', '')
    if api_key.startswith('sk-'):
        print("   OpenAI: ✅ Configured")
    else:
        print("   OpenAI: ❌ Not configured")

if __name__ == "__main__":
    check_system_status() 