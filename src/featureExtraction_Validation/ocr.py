from paddleocr import PaddleOCR


class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)

    def extract_text(self, image):
        result = self.ocr.predict(
            input=image)
        return result["rec_texts"]