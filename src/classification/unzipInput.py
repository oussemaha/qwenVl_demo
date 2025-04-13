import zipfile
import os
from pathlib import Path
from configs.constants import TEMP_DIR
import tempfile

  # You can change this to any path you prefer

def unzip_file(uploaded_zip_bytes):
    # Ensure TEMP_DIR exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Create a temporary file to save the zip
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
        tmp_zip.write(uploaded_zip_bytes)
        zip_path = tmp_zip.name
    
    # Extract the ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)
    
    # Clean up the temp zip file
    os.unlink(zip_path)
    
    # List extracted files
    extracted_files = []
    for root, _, files in os.walk(TEMP_DIR):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), TEMP_DIR)
            extracted_files.append(rel_path)
    
    return "\n".join(extracted_files)