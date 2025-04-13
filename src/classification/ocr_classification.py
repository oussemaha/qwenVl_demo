import easyocr
import re
import cv2
import fitz
from Levenshtein import distance
from pathlib import Path
import numpy as np
import os 
import shutil
from configs.constants import UNIVERSAL_KEYWORDS, UNIVERSAL_PATTERNS
# Universal invoice keywords (supports 20+ languages)

def open_file_as_image(file_path):
    """
    Opens a file (PDF or image) and returns it as a NumPy array.
    Uses different methods for PDF vs image files.
    
    Args:
        file_path (str or Path): Path to the file
        
    Returns:
        numpy.ndarray: Image data in BGR format
        None: If the file can't be opened
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Handle PDF files
    if file_path.suffix.lower() == '.pdf':
        try:  # PyMuPDF (import here to avoid conflicts)
            
            doc = fitz.open(file_path)
            page = doc.load_page(0)  # First page
            pix = page.get_pixmap()
            
            # Convert to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape((pix.height, pix.width, pix.n))
            
            # Convert to BGR format expected by OpenCV
            if pix.n == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                
            doc.close()
            return img_array
        
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return None
    
    # Handle image files
    else:
        try:
            # First try standard OpenCV imread
            img = cv2.imread(str(file_path))
            if img is not None:
                return img
                
            # If that fails, try alternative reading method
            with open(file_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return img
                
        except Exception as e:
            print(f"Error reading image {file_path}: {str(e)}")
            return None
            
def extract_text_from_easyocr(results):
    """Extract text from EasyOCR results in the format [[[points], text], ...]"""
    extracted_text = []
    for item in results:
        if len(item) >= 2:  # Should be [[points], text]
            text = item[1]
            if isinstance(text, str):
                extracted_text.append(text)
    return " ".join(extracted_text)

def is_invoice(image_path: str, min_text_length: int = 50, keyword_threshold: int = 3) -> bool:
    """
    Detects invoices in ANY language without prior knowledge.
    - Uses EasyOCR's multilingual model (auto-detects language).
    - Handles Arabic by including 'en' in the language list.
    - Checks for universal keywords + regex patterns.
    """
    # Initialize EasyOCR with compatible languages
    reader = easyocr.Reader(["en", "fr", "es", "it", "de"])

    # Read image
    image = open_file_as_image(image_path)
    if image is None:
        return False,image

    # Extract text (auto-detects language and orientation)
    try:
        results = reader.readtext(image, paragraph=False)  # Explicitly set paragraph=False
        extracted_text = extract_text_from_easyocr(results)
    except Exception as e:
        print(f"OCR processing failed: {e}")
        return False,image

    # Rule 1: Minimum text check
    if len(extracted_text) < min_text_length:
        return False,image

    normalized_text = extracted_text.lower()
    
    # Rule 2: Fuzzy keyword matching across languages
    keyword_count = 0
    for keyword in UNIVERSAL_KEYWORDS:
        if keyword.lower() in normalized_text:
            print( keyword)
            keyword_count += 1
        else:
            # Fuzzy match for OCR errors (e.g., "invoic" → "invoice")
            for word in normalized_text.split():
                if distance(word.lower(), keyword.lower()) <= 1:
                    keyword_count += 1
                    print(word, keyword)
                    break

    # Rule 3: Universal regex patterns
    has_invoice_patterns = any(
        re.search(p, extracted_text, re.IGNORECASE)
        for p in UNIVERSAL_PATTERNS
    )
    
    print("has_invoice_patterns ",has_invoice_patterns)
    #print("has_invoice_structure ",has_invoice_structure)
    print("keyword_count ",keyword_count)
    return (keyword_count) >= keyword_threshold or has_invoice_patterns , image

def organize_invoices(root_folder):
    """
    Scan through folders and move invoices to 'Invoices' subfolders
    while preserving the original folder structure.
    """
    root_path = Path(root_folder)
    invoice_list=[]
    # Walk through all files in the directory tree
    for folder_path, _, filenames in os.walk(root_folder):
        folder_path = Path(folder_path)
        #invoices_subfolder = folder_path / "Invoices"
        
        # Create invoices subfolder if it doesn't exist
        #invoices_subfolder.mkdir(exist_ok=True)
        
        for filename in filenames:
            print(filename)
            file_path = folder_path / filename
            
            # Skip files already in an "Invoices" folder
            if "Invoices" in file_path.parts:
                continue
            result,image=is_invoice(file_path)
            if result:
                """"
                #putting in an INVOICES dir
                # Construct destination path
                dest_path = invoices_subfolder / filename
                
                # Handle potential filename conflicts
                counter = 1
                while dest_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    dest_path = invoices_subfolder / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                #Move the file
                shutil.move(str(file_path), str(dest_path))
                print(f"Moved invoice: {file_path} -> {dest_path}")
                """
                #create list 
                invoice_list.append(image)
    #  delete the root folder 
    try:
        shutil.rmtree(root_folder)
        deletion_status = f"Successfully deleted folder: {root_folder}"
    except Exception as e:
        deletion_status = f"Error deleting folder {root_folder}: {str(e)}"
    return invoice_list