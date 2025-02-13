from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
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

def read_file_to_string(file_path:str):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

def extract_data_QWEN(file):
    sys_prompt=read_file_to_string(SYSTEM_PROMPT_PATH)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text":sys_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": file,
                }            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(type(result))
    result=result[0]        
    result.replace('\n','')
    result.replace('\t','')

    result=result[result.find('{'):result.find('}')+1]
    result=json.loads(result)
    print(result)
    return result

def greet(name,file):
    print(file)
    return "Hello, " + name + "!"

demo = gr.Interface(
    fn=extract_data_QWEN,  # Function to call
    inputs=gr.Image(type="pil", label="Upload Image"),  # Input: Image (as PIL image)
    outputs=gr.JSON(label="Output JSON"),  # Output: JSON
    title="Invoice to JSON Processor",
    description="Upload an image of the invoice and get the content as JSON."
)

demo.launch(share=True)
