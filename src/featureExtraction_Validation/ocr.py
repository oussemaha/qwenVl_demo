from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)

    def extract_text(self, image):
        img=""
        if type(image) is str:
            img=Image.open(image)
            img = np.array(img)
        else:
            img = np.array(image)

        result = self.ocr.predict(
            input=img)
        return result["rec_texts"]