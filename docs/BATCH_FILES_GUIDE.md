# 🚀 RAG Chatbot - Batch Files Guide

## 📁 **All-in-One Startup Scripts**

I've created **3 convenient batch files** to make running your RAG chatbot super easy on Windows:

---

## 🎯 **1. start_rag_chatbot.bat** (MAIN STARTUP)

**The complete all-in-one solution!**

### **What it does:**
- ✅ Checks Python installation
- ✅ Validates project files
- ✅ Creates/checks `.env` configuration
- ✅ Installs/updates all dependencies
- ✅ Checks for Tesseract OCR
- ✅ Creates necessary directories
- ✅ Shows system status
- ✅ Starts the RAG chatbot
- ✅ Provides troubleshooting tips if issues occur

### **When to use:**
- **First time setup**
- **After updating code/dependencies**
- **When troubleshooting issues**
- **When you want full system validation**

### **Usage:**
```batch
# Double-click the file or run from command prompt:
start_rag_chatbot.bat
```

### **Output Example:**
```
==========================================
   🤖 RAG Chatbot - All-in-One Startup
==========================================

[1/6] 🐍 Checking Python installation...
Python 3.13.0
✅ Python found!

[2/6] 📁 Checking project directory...
✅ Project files found!

[3/6] 🔑 Checking environment configuration...
✅ OpenAI API key found!

[4/6] 📦 Installing/Updating dependencies...
✅ Dependencies installed successfully!

[5/6] 🔍 Checking OCR support...
✅ Tesseract OCR found - Full OCR support available!

[6/6] 📂 Setting up directories...
✅ Directories ready!

==========================================
   🚀 STARTING RAG CHATBOT
==========================================

📊 System Status:
   Python: 3.13.0
   PyMuPDF: ✅ Available (Enhanced PDF + OCR)
   Tesseract: ✅ Available (OCR enabled)
   Web Scraping: ✅ Available
   OpenAI: ✅ Configured

🌐 Starting server on http://localhost:5000
🗑️ Database management available in web UI
📝 Logs will appear below...
```

---

## ⚡ **2. quick_start.bat** (FAST STARTUP)

**For when everything is already set up!**

### **What it does:**
- ✅ Quick file check
- ✅ Starts the app immediately
- ✅ No dependency installation or validation

### **When to use:**
- **Daily use** after initial setup
- **When you know everything is working**
- **Quick restarts**
- **Development/testing**

### **Usage:**
```batch
# Double-click or run:
quick_start.bat
```

### **Output Example:**
```
🚀 Quick Starting RAG Chatbot...

✅ Starting server on http://localhost:5000
📝 Press Ctrl+C to stop the server

🚀 STARTING CORRECTED RAG CHATBOT
🔍 PDF OCR: ✅ Enabled
🌐 Web Processing: ✅ Enabled
🗑️ Database clearing: ✅ Available
```

---

## 🗑️ **3. manage_database.bat** (DATABASE MANAGER)

**Easy database management from command line!**

### **What it does:**
- ✅ Interactive menu system
- ✅ Show database status
- ✅ Clear entire database with confirmation
- ✅ Verify cleanup completion
- ✅ User-friendly interface

### **When to use:**
- **Before starting fresh**
- **When database has issues**
- **Regular maintenance**
- **Testing different documents**

### **Usage:**
```batch
# Double-click or run:
manage_database.bat
```

### **Output Example:**
```
==========================================
   🗑️ RAG Chatbot - Database Manager
==========================================

Please choose an option:

[1] 📊 Show database status
[2] 🗑️ Clear entire database
[3] ✅ Verify cleanup
[4] 🚪 Exit

Enter your choice (1-4): 1

📊 Getting database status...
📊 CURRENT DATABASE STATUS:
  📚 Collection 'documents': 95 documents
  📈 Total documents: 95
  📄 Image registry: 1245 bytes
  📁 Downloaded images: 15 files
  📝 Processing registry: 892 bytes
```

---

## 🎮 **How to Use - Quick Guide**

### **First Time Setup:**
1. **Run**: `start_rag_chatbot.bat`
2. **Edit** the created `.env` file with your OpenAI API key
3. **Run**: `start_rag_chatbot.bat` again

### **Daily Use:**
1. **Run**: `quick_start.bat`
2. **Visit**: `http://localhost:5000`

### **Database Management:**
1. **Run**: `manage_database.bat`
2. **Choose option** from the menu

### **Troubleshooting:**
1. **Run**: `start_rag_chatbot.bat` (it includes diagnostics)
2. **Check** the error messages and follow suggestions

---

## 🔧 **Troubleshooting Common Issues**

### **"Python not found"**
- Install Python 3.8+ from: https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation

### **"Dependencies failed to install"**
- Run manually: `pip install -r requirements_ultimate_fixed.txt`
- Check internet connection
- Try: `python -m pip install --upgrade pip`

### **"Port 5000 already in use"**
- Close other applications using port 5000
- Or edit `app_corrected_final.py` to use a different port

### **"OpenAI API key not configured"**
- Edit the `.env` file created by the startup script
- Add your OpenAI API key: `OPENAI_API_KEY=sk-your-key-here`

### **"OCR not working"**
- Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- The app will still work without it (basic PDF text extraction only)

---

## 📊 **File Comparison**

| Feature | start_rag_chatbot.bat | quick_start.bat | manage_database.bat |
|---------|----------------------|-----------------|-------------------|
| **Full Setup** | ✅ | ❌ | ❌ |
| **Dependency Check** | ✅ | ❌ | ❌ |
| **Environment Validation** | ✅ | ❌ | ❌ |
| **Quick Launch** | ❌ | ✅ | ❌ |
| **Database Management** | ❌ | ❌ | ✅ |
| **Troubleshooting** | ✅ | ❌ | ❌ |
| **Best For** | Setup & Issues | Daily Use | Database Tasks |

---

## 🎯 **Recommended Workflow**

### **Initial Setup (Once):**
```batch
1. start_rag_chatbot.bat    # Full setup
2. Edit .env file           # Add API keys
3. start_rag_chatbot.bat    # Verify everything works
```

### **Daily Usage:**
```batch
1. quick_start.bat          # Fast startup
2. Use web interface        # Upload & chat
3. Ctrl+C                   # Stop when done
```

### **Maintenance:**
```batch
1. manage_database.bat      # Clean database when needed
2. start_rag_chatbot.bat    # If issues occur
```

---

## 🚀 **Ready to Go!**

Your RAG chatbot is now super easy to manage with these batch files:

- **🔧 Full Setup**: `start_rag_chatbot.bat`
- **⚡ Quick Start**: `quick_start.bat`  
- **🗑️ Database Manager**: `manage_database.bat`

Just double-click any of these files to get started! 🎉 