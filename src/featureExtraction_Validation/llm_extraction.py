from transformers import Qwen2VLForConditionalGeneration, AutoProcessor,AutoModelForVision2Seq
import torch

from qwen_vl_utils import process_vision_info
import json
from PIL import Image



class LLM_extractor:
    def __init__(self,model_name,prompt_file_path):
        """
            for more generic use, change ```Qwen2VLForConditionalGeneration``` by ```AutoModelForVision2Seq```
        """
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.prompt_file_path=prompt_file_path
        
    def read_file_to_string(self,file_path:str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            return "File not found."
        except Exception as e:
            return f"An error occurred: {e}"
        
    def string_to_JSON(self,text):
        text.replace('\n','')
        text.replace('\t','')
        text=text[text.find('{'):text.find('}')+1]
        result=json.loads(text)
        for key in result:
            try:
                result[key]=int(result[key])
            except:
                try:
                    result[key]=float(result[key])
                except:
                    "none"
        return result
    
    def extract_data(self,images, response_format):
        sys_prompt=self.read_file_to_string(self.prompt_file_path) % response_format
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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=600)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        result = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result=result[0]        
        result=self.string_to_JSON(result)
        return result