from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from src.featureExtraction.llm_extraction import extract_data_QWEN
from src.classification.unzipInput import unzip_file
from src.classification.ocr_classification import organize_invoices
from configs.constants import *
import torch

import gradio as gr



model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

def main(zip_file):
    unzip_file(zip_file)
    invoice_list=organize_invoices(TEMP_DIR)
    result=extract_data_QWEN(invoice_list,model,processor)
    return result


def greet(name,file):
    print(file)
    return "Hello, " + name + "!"

demo = gr.Interface(
    fn=main,  # Function to call
    inputs= gr.File(label="ZIP File", type="binary"),  # Input: Image (as PIL image)
    outputs=gr.JSON(label="Output JSON"),  # Output: JSON
    title="Invoice to JSON Processor",
    description="Upload an image of the invoice and get the content as JSON."
)

demo.launch(share=True)
