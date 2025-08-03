#!/usr/bin/env python3
"""
Simple script to clear the ChromaDB vector database and related files
"""

import os
import shutil
import sys
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings

# Add parent directory to path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_vector_database():
    """Clear the ChromaDB vector database and related files"""
    try:
        logger.info("🗑️ STARTING DATABASE CLEANUP")
        logger.info("⚠️ WARNING: Make sure the RAG application is stopped before running this!")
        
        # Files and folders to clean (order matters - files first, then directories)
        items_to_clean = [
            ("image_registry.json", "file"),  # Image registry file
            ("processed_files.json", "file"), # Document processing registry
            (str(Config.PROJECT_ROOT / "extracted_images"), "dir"),      # Downloaded images folder
            (Config.CHROMA_PERSIST_DIRECTORY, "dir")              # ChromaDB storage (last due to potential locks)
        ]
        
        cleanup_count = 0
        
        for item_name, item_type in items_to_clean:
            item_path = Path(item_name)
            
            if item_path.exists():
                try:
                    if item_type == "file":
                        logger.info(f"🗑️ Removing file: {item_name}")
                        item_path.unlink()
                        cleanup_count += 1
                    elif item_type == "dir":
                        file_count = len(list(item_path.rglob('*')))
                        logger.info(f"🗑️ Removing directory: {item_name} ({file_count} files)")
                        
                        # Try multiple times with increasing delays for Windows file locking
                        max_attempts = 3
                        for attempt in range(max_attempts):
                            try:
                                shutil.rmtree(item_path)
                                logger.info(f"✅ Successfully removed: {item_name}")
                                cleanup_count += 1
                                break
                            except PermissionError as pe:
                                if attempt < max_attempts - 1:
                                    import time
                                    wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                                    logger.warning(f"⚠️ Attempt {attempt + 1} failed for {item_name}, waiting {wait_time}s and retrying...")
                                    logger.warning(f"   Error: {pe}")
                                    time.sleep(wait_time)
                                else:
                                    logger.error(f"❌ Failed to remove {item_name} after {max_attempts} attempts")
                                    logger.error(f"   This usually means the RAG application is still running")
                                    logger.error(f"   Please stop the application (Ctrl+C) and try again")
                                    raise pe
                            except Exception as e:
                                logger.error(f"❌ Unexpected error removing {item_name}: {e}")
                                break
                                
                except Exception as e:
                    logger.error(f"❌ Failed to remove {item_name}: {e}")
                    logger.warning(f"⚠️ Continuing with other items...")
                    # Continue with other items even if one fails
                    
            else:
                logger.info(f"⚪ Skipping (not found): {item_name}")
        
        # Also clean uploads folder if desired
        uploads_path = Path("uploads")
        if uploads_path.exists():
            files_in_uploads = list(uploads_path.glob("*"))
            if files_in_uploads:
                response = input(f"\n❓ Also clear uploads folder? ({len(files_in_uploads)} files) [y/N]: ")
                if response.lower() in ['y', 'yes']:
                    for file in files_in_uploads:
                        if file.is_file():
                            file.unlink()
                            logger.info(f"🗑️ Removed upload: {file.name}")
                    cleanup_count += len(files_in_uploads)
        
        logger.info(f"✅ DATABASE CLEANUP COMPLETE")
        logger.info(f"📊 Cleaned {cleanup_count} items")
        logger.info("🔄 You can now restart your RAG application with a fresh database")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        return False

def verify_cleanup():
    """Verify that cleanup was successful"""
    items_to_check = [Config.CHROMA_PERSIST_DIRECTORY, "image_registry.json", str(Config.PROJECT_ROOT / "extracted_images"), "processed_files.json"]
    
    all_clean = True
    for item in items_to_check:
        if Path(item).exists():
            logger.warning(f"⚠️ Still exists: {item}")
            all_clean = False
    
    if all_clean:
        logger.info("✅ All database files successfully removed")
    else:
        logger.warning("⚠️ Some files may still exist")
    
    return all_clean

def get_database_info():
    """Get information about current database state"""
    try:
        logger.info("📊 CURRENT DATABASE STATUS:")
        
        # Check ChromaDB
        chroma_path = Path(Config.CHROMA_PERSIST_DIRECTORY)
        if chroma_path.exists():
            try:
                client = chromadb.Client(Settings(
                    persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
                    is_persistent=True
                ))
                collections = client.list_collections()
                
                total_docs = 0
                for collection in collections:
                    count = collection.count()
                    total_docs += count
                    logger.info(f"  📚 Collection '{collection.name}': {count} documents")
                
                logger.info(f"  📈 Total documents: {total_docs}")
                
            except Exception as e:
                logger.warning(f"  ⚠️ ChromaDB error: {e}")
        else:
            logger.info("  ⚪ ChromaDB: Not initialized")
        
        # Check other files
        other_files = {
            "image_registry.json": "Image registry",
            str(Config.PROJECT_ROOT / "extracted_images"): "Downloaded images",
            "processed_files.json": "Processing registry"
        }
        
        for file_path, description in other_files.items():
            path = Path(file_path)
            if path.exists():
                if path.is_file():
                    logger.info(f"  📄 {description}: {path.stat().st_size} bytes")
                elif path.is_dir():
                    file_count = len(list(path.rglob("*")))
                    logger.info(f"  📁 {description}: {file_count} files")
            else:
                logger.info(f"  ⚪ {description}: Not found")
        
    except Exception as e:
        logger.error(f"❌ Error getting database info: {e}")

def main():
    """Main function with interactive menu"""
    print("🤖 RAG Database Manager")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. 📊 Show database status")
        print("2. 🗑️ Clear entire database")
        print("3. ✅ Verify cleanup")
        print("4. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            get_database_info()
        elif choice == "2":
            print("\n⚠️ WARNING: This will delete ALL documents, images, and processed data!")
            confirm = input("Are you sure? Type 'yes' to confirm: ")
            if confirm.lower() == 'yes':
                clear_vector_database()
            else:
                print("❌ Cancelled")
        elif choice == "3":
            verify_cleanup()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 