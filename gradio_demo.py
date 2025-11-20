from src.featureExtraction_Validation.llm_extraction import LLM_extractor
from src.featureExtraction_Validation.json_validator import JSONComparator
from src.classification.unzipInput import unzip_file
from src.classification.ocr_classification import Classifier
from configs.constants import *
import time as t
import gradio as gr


llm_extractor=LLM_extractor(MODEL_NAME,SYSTEM_PROMPT_PATH)
classifier= Classifier(UNIVERSAL_KEYWORDS,UNIVERSAL_PATTERNS,CLASSIFICATION_KEYWORD_THRESHOLD,CLASSIFICATION_MIN_TEXT_LENGTH,gpu=CLASSIFICATION_GPU)
validator=JSONComparator()

def format_of_response(user_json):
    response_format="{\n"
    for key in user_json:
        response_format=response_format+f"\"{key}\": , \n"
    response_format=response_format+"}"
    return response_format

def main(zip_file, form_json):
    t1=t.time()
    user_json=llm_extractor.string_to_JSON(form_json)
    response_format=format_of_response(user_json)
    
    unzip_file(zip_file)
    
    invoice_list=classifier.organize_invoices_para(TEMP_DIR)
    #invoice_list=classifier.open_all_as_image(TEMP_DIR)
    llm_json=llm_extractor.extract_data(invoice_list,response_format)
    comparison_result=validator.compare_jsons(user_json,llm_json)
    print(t.time()-t1)
    return comparison_result,llm_json
    #return {"yse":1},{"yse":1}


def greet(name,file):
    print(file)
    return "Hello, " + name + "!"

demo = gr.Interface(
    fn=main,  # Function to call
    inputs= [gr.File(label="ZIP File", type="binary"),gr.Textbox(label="Form to valdiate JSON") ],  # Input: Image (as PIL image)
    outputs=[gr.JSON(label="LLM exraction"),gr.JSON(label="Validator")],  # Output: JSON
    title="Invoice to JSON Processor",
    description="Upload an image of the invoice and get the content as JSON."
)

demo.launch(share=True)
