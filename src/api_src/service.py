
from http.client import HTTPException
from src.featureExtraction_Validation.ocr import OCRProcessor
from src.featureExtraction_Validation.json_validator import JSONComparator
from src.featureExtraction_Validation.gemma_extraction import GemmaImageProcessor
from configs.constants import *

import logging


class Service:
    
    def __init__(self):
        self.ai_engine=GemmaImageProcessor(MODEL_NAME)
        self.ocr_processor=OCRProcessor()
        logging.basicConfig(level=logging.INFO)
        self.logger =logging.getLogger(__name__)
        self.validator=JSONComparator(self.ai_engine)

        
    def extract(self, images, prompts): 

        try:
            llm_json = dict()
            ocr_text = " "
            i=0
            for img in images:
                i+=1
                ocr_text = ocr_text + f" \n### Image {i} OCR Text:\n "
                ocr_results = self.ocr_processor.extract_text(img)
                ocr_text = ocr_text + " ".join(ocr_results)
            # Extract and compare data
            for sys_prompt in prompts:
                sys_prompt = sys_prompt.replace("<OCR TEXT>", ocr_text)
                llm_json.update(self.ai_engine.extract_data(images, sys_prompt=sys_prompt))
            return llm_json
        except Exception as e:
            raise HTTPException(status_code=500, detail="error during data extraction")
        
    def extract_and_compare(self, images, prompts, reference_json):

        extracted_data = self.extract(images, prompts)
        comparison_result = self.validator.compare_jsons(reference_json, extracted_data)
        return extracted_data, comparison_result
       
    
    def read_file_to_string( self,file_path: str) -> str:
            """
            Read content from a file and return as string.

            Args:
                file_path: Path to the file to read

            Returns:
                Content of the file as string
            """
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except FileNotFoundError:
                return "File not found."
            except Exception as e:
                return f"An error occurred: {e}"