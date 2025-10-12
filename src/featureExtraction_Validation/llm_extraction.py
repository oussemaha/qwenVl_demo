from transformers import AutoModelForImageTextToText,Qwen2VLForConditionalGeneration, AutoProcessor,AutoModelForVision2Seq,Qwen3VLMoeForConditionalGeneration
import torch

from qwen_vl_utils import process_vision_info
import json
from PIL import Image



class LLM_extractor:
    def __init__(self,model_name):
        """
            for more generic use, change ```Qwen2VLForConditionalGeneration``` by ```AutoModelForVision2Seq```
        """
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
             dtype="auto",
            device_map="auto",
            #     attn_implementation="flash_attention_2",

        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        
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
    def llm_compare_texts(self,text1,text2):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text":"you're an expert in text comparison, compare the two texts and return a boolean value indicating whether they are similar or not. don't be too strict. return only True or False."
                    }
                ]
            },
            {
                "role": "user",
                "content":"compare these two texts:\n\nText 1: %s\n\nText 2: %s" % (text1, text2)
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
        return bool(result.strip().lower() == "true")  # Assuming the model returns "True" or "False"        
    
    def extract_data(self,images, sys_prompt):
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
        try:
            result = result[result.find('{'):result.rfind('}')+1]
            result=self.string_to_JSON(result)
        except:
            result ={
                      "REF_CONTRAT": None,
                      "CURRENCY": None,
                      "AMOUNT_PTFN": None,
                      "AMOUNT_FOB": None,
                      "INVOICE_NUMBER": None,
                      "INVOICE_DATE": None,
                      "SELLER_NAME": None,
                      "SELLER_ADDRESS": None,
                      "BUYER_NAME": None,
                      "BUYER_ADDRESS": None,
                      "MODE_REGLEMENT_CODE": None,
                      "CODE_DELAI_REGLEMENT": None,
                      "CODE_MODE_LIVRAISON": None,
                      "ADVANCE_PAYMENT": None
                    }
        
        return result