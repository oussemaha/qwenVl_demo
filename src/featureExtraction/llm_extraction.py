from qwen_vl_utils import process_vision_info
from configs.constants import SYSTEM_PROMPT_PATH
import json
from PIL import Image

def read_file_to_string(file_path:str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

def extract_data_QWEN(images,model,processor):
    sys_prompt=read_file_to_string(SYSTEM_PROMPT_PATH)
    message_content = [
        {
            "type": "text",
            "text": "Describe these images:"  # Optional text prompt
        }
    ]
    message_content = []
    # Add images using a for loop
    for img in images:
        pil_img = Image.fromarray(img)
        message_content.append({
            "type": "image",
            "image": pil_img  # Convert to RGB PIL Image
        })

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
            "content": message_content,
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
    generated_ids = model.generate(**inputs, max_new_tokens=600)
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
    print(result)

    result=result[result.find('{'):result.find('}')+1]
    result=json.loads(result)
    print(result)
    return result