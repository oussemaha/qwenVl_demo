import zipfile
import os
from pathlib import Path
from constants import TEMP_DIR
  # You can change this to any path you prefer

def unzip_file(zip_file):
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Clear any existing files in temp directory
        for existing_file in Path(TEMP_DIR).glob('*'):
            existing_file.unlink()
        
        # Save the uploaded zip file to temp dir
        zip_path = os.path.join(TEMP_DIR, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file)
        
        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        
        # Get list of extracted files
        extracted_files = []
        for item in Path(TEMP_DIR).rglob('*'):
            if item.is_file() and item.name != "uploaded.zip":
                extracted_files.append(str(item.relative_to(TEMP_DIR)))
        
        return f"Success! Extracted {len(extracted_files)} files to {TEMP_DIR}:\n" + "\n".join(extracted_files)
    
    except Exception as e:
        return f"Error: {str(e)}"