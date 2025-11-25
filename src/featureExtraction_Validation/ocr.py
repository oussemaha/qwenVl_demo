from paddleocr import PaddleOCR


class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)

    def extract_text(self, image):
        try:
            result = self.ocr.predict(
                input=image)
            return result[0]["rec_texts"]
        except Exception as e:
            print(f"Error during OCR extraction: {e}")
            return ""