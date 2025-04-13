from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.featureExtraction.llm_extraction import extract_data_QWEN
import torch

import gradio as gr
from constants import *

import json


model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

def main(zip_file):
    unzip_file(zip_file)
    organize_invoices(TEMP_DIR)
    return {"status": "success", "message": "Files organized successfully."}


def greet(name,file):
    print(file)
    return "Hello, " + name + "!"

demo = gr.Interface(
    fn=main,  # Function to call
    inputs=gr.File(label="Upload ZIP file", file_types=[".zip"]),  # Input: Image (as PIL image)
    outputs=gr.JSON(label="Output JSON"),  # Output: JSON
    title="Invoice to JSON Processor",
    description="Upload an image of the invoice and get the content as JSON."
)

demo.launch(share=True)
