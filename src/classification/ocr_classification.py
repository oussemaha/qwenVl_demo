import easyocr
import re
import cv2
import fitz
from Levenshtein import distance
from pathlib import Path
import numpy as np
import os 
import shutil
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
# Universal invoice keywords (supports 20+ languages)

class Classifier :
    def __init__(self,UNIVERSAL_KEYWORDS,UNIVERSAL_PATTERNS,KEYWORD_THRESHOLD:int,CLASSIFICATION_MIN_TEXT_LENGTH:int,gpu:bool):
        self.UNIVERSAL_KEYWORDS=UNIVERSAL_KEYWORDS
        self.UNIVERSAL_PATTERNS=UNIVERSAL_PATTERNS
        self.KEYWORD_THRESHOLD:int=KEYWORD_THRESHOLD
        self.CLASSIFICATION_MIN_TEXT_LENGTH:int=CLASSIFICATION_MIN_TEXT_LENGTH
        self.gpu:bool=gpu
    def open_file_as_image(self, file_path):
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
            
        elif file_path.suffix.lower() == '.gif':
            try:
                # Read the GIF using OpenCV which will return the first frame
                gif = cv2.VideoCapture(str(file_path))
                ret, frame = gif.read()
                gif.release()

                if ret:
                    return frame
                else:
                    print(f"Error reading GIF {file_path}: Could not read frames")
                    return None

            except Exception as e:
                print(f"Error reading GIF {file_path}: {str(e)}")
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

    def extract_text_from_easyocr(self, results):
        """Extract text from EasyOCR results in the format [[[points], text], ...]"""
        extracted_text = []
        for item in results:
            if len(item) >= 2:  # Should be [[points], text]
                text = item[1]
                if isinstance(text, str):
                    extracted_text.append(text)
        return " ".join(extracted_text)

    def is_invoice(self, image_path: str) -> bool:
        """
        Detects invoices in ANY language without prior knowledge.
        - Uses EasyOCR's multilingual model (auto-detects language).
        - Handles Arabic by including 'en' in the language list.
        - Checks for universal keywords + regex patterns.
        """
        # Initialize EasyOCR with compatible languages
        reader = easyocr.Reader(["en", "fr", "es", "it", "de"],gpu=True)
        print("is invoice"+str(image_path))
        # Read image
        image = self.open_file_as_image(image_path)
        if image is None:
            return None

        # Extract text (auto-detects language and orientation)
        try:
            results = reader.readtext(image, paragraph=False)  # Explicitly set paragraph=False
            extracted_text = self.extract_text_from_easyocr(results)
        except Exception as e:
            print(f"OCR processing failed: {e}")
            return None

        # Rule 1: Minimum text check
        if len(extracted_text) < self.CLASSIFICATION_MIN_TEXT_LENGTH:
            return None

        normalized_text = extracted_text.lower()

        # Rule 2: Fuzzy keyword matching across languages
        keyword_count = 0
        for keyword in self.UNIVERSAL_KEYWORDS:
            if keyword.lower() in normalized_text:
                keyword_count += 1
            else:
                # Fuzzy match for OCR errors (e.g., "invoic" → "invoice")
                for word in normalized_text.split():
                    if distance(word.lower(), keyword.lower()) <= 1:
                        keyword_count += 1
                        break

        # Rule 3: Universal regex patterns
        has_invoice_patterns = any(
            re.search(p, extracted_text, re.IGNORECASE)
            for p in self.UNIVERSAL_PATTERNS
        )

        #print("has_invoice_structure ",has_invoice_structure)
        if (keyword_count) >= self.KEYWORD_THRESHOLD or has_invoice_patterns :
            return image
        return None
    def is_invoice_without_ocr(self, extracted_text: str) -> bool:
        """
        Detects invoices in ANY language without prior knowledge.
        - Uses EasyOCR's multilingual model (auto-detects language).
        - Handles Arabic by including 'en' in the language list.
        - Checks for universal keywords + regex patterns.
        """
        # Rule 1: Minimum text check

        if len(extracted_text) < self.CLASSIFICATION_MIN_TEXT_LENGTH:
            return False

        normalized_text = extracted_text.lower()

        # Rule 2: Fuzzy keyword matching across languages
        keyword_count = 0
        for keyword in self.UNIVERSAL_KEYWORDS:
            if keyword.lower() in normalized_text:
                keyword_count += 1
            else:
                # Fuzzy match for OCR errors (e.g., "invoic" → "invoice")
                for word in normalized_text.split():
                    if distance(word.lower(), keyword.lower()) <= 1:
                        keyword_count += 1
                        break


        #print("has_invoice_structure ",has_invoice_structure)
        if (keyword_count) >= self.KEYWORD_THRESHOLD :
            return True
        return False
    def organize_invoices_para(self, root_folder):

        invoice_list=[]
        for folder_path, _, filenames in os.walk(root_folder):
            folder_path = Path(folder_path)
            for i in range(len(filenames)):
                filenames[i] = folder_path / filenames[i]
            invoice_list.extend(self.process_batch(filenames))
        #  delete the root folder 
        try:
            shutil.rmtree(root_folder)
            deletion_status = f"Successfully deleted folder: {root_folder}"
        except Exception as e:
            deletion_status = f"Error deleting folder {root_folder}: {str(e)}"
        return invoice_list
    def organize_invoices(self,root_folder):
 
        invoice_list=[]
        # Walk through all files in the directory tree
        for folder_path, _, filenames in os.walk(root_folder):
            folder_path = Path(folder_path)
            #invoices_subfolder = folder_path / "Invoices"

            # Create invoices subfolder if it doesn't exist
            #invoices_subfolder.mkdir(exist_ok=True)

            for filename in filenames:
                file_path = folder_path / filename

                # Skip files already in an "Invoices" folder
                if "Invoices" in file_path.parts:
                    continue
                result,image=self.is_invoice(file_path,self.CLASSIFICATION_MIN_TEXT_LENGTH,self.KEYWORD_THRESHOLD)
                if result:
                    #create list 
                    invoice_list.append(image)
        #  delete the root folder 
        
        try:
            shutil.rmtree(root_folder)
            deletion_status = f"Successfully deleted folder: {root_folder}"
        except Exception as e:
            deletion_status = f"Error deleting folder {root_folder}: {str(e)}"
        return invoice_list
    def open_all_as_image(self, root_folder):
        invoice_list=[]
        # Walk through all files in the directory tree
        for folder_path, _, filenames in os.walk(root_folder):
            folder_path = Path(folder_path)

            for filename in filenames:
                file_path = folder_path / filename
                image=self.open_file_as_image(file_path)
                invoice_list.append(image)
        return invoice_list

    def process_batch(self, file_paths):
        """Process multiple files in a true batch with resizing"""
        reader = easyocr.Reader(["en", "fr", "es", "it", "de"], gpu=self.gpu)

        try:
            # Load and resize all images to a common size
            images = []
            valid_paths = []
            max_height, max_width = 0, 0

            # First pass: find max dimensions
            for path in file_paths:
                img = self.open_file_as_image(path)
                if img is not None:
                    h, w = img.shape[:2]
                    max_height = max(max_height, h)
                    max_width = max(max_width, w)
                    images.append(img)
                    valid_paths.append(path)

            if not images:
                return []

            # Second pass: pad images
            padded_images = []
            for img in images:
                h, w = img.shape[:2]
                # Create padded image
                padded = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                padded[:h, :w] = img
                padded_images.append(padded)

            # Process batch
            results = reader.readtext_batched(padded_images, batch_size=len(padded_images),paragraph=False)

            batch_output = []
            
            for img, path, ocr_results in zip(images,valid_paths, results):
                extracted_text = " ".join([res[1] for res in ocr_results if len(res) >= 2])
                print(path)
                if self.is_invoice_without_ocr(extracted_text) :
                    print(path)
                    batch_output.append(path)

            return batch_output

        except Exception as e:
            print(e)
            return []
    def organize_invoices_para_temp(self, root_folder):

        invoice_list=[]
        for folder_path, _, filenames in os.walk(root_folder):
            folder_path = Path(folder_path)

            for i in range(len(filenames)):
                filenames[i] = root_folder / filenames[i]
            invoice_list.extend(self.process_batch(filenames))
            print("invoice_list",invoice_list)
        return invoice_list
        #  delete the root folder 

  