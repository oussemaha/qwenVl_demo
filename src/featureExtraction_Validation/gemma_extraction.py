from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import json
from PIL import Image
import time
from typing import List, Union, Dict, Any
import numpy as np

class GemmaImageProcessor:
    def __init__(self, model_name: str = "google/gemma-3-12b-it"):
        """
        Initialize the Gemma image processor.
        
        Args:
            model_name: Name of the pretrained model (default: "google/gemma-3-12b-it")
            prompt_file_path: Path to the prompt file (optional)
        """
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
   
        
    def string_to_JSON(self, text: str) -> Dict[str, Any]:
        """
        Convert a string containing JSON data to a Python dictionary.
        
        Args:
            text: String containing JSON data
            
        Returns:
            Parsed JSON as dictionary
        """
        text = text.replace('\n', '').replace('\t', '')
        try:
            # Find first { and last } to extract JSON portion
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                return {}
                
            json_str = text[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Try to convert values to int or float where possible
            for key in result:
                try:
                    if key in ["AMOUNT_PTFN", "AMOUNT_FOB","ADVANCE_PAYMENT"]:
                        result[key] = float(result[key])
                    elif key in ["CURRENCY","CODE_DELAI_REGLEMENT","CODE_MODE_LIVRAISON","MODE_REGLEMENT_CODE"]:
                        result[key] = int(result[key])
                    else: 
                        pass
                except (ValueError, TypeError):
                    pass
            return result
        except json.JSONDecodeError:
            return {}
            
    def generate_response(self, images: Union[List[Image.Image], List[np.ndarray], Image.Image, np.ndarray], 
                         prompt: str = None, max_new_tokens: int = 600) -> str:
        """
        Generate text response for the given image(s) and prompt.
        
        Args:
            images: Single image or list of images (PIL Image or numpy array)
            prompt: Text prompt to use (optional)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        # Convert single image to list for consistent processing
        if not isinstance(images, list):
            images = [images]
            
        # Convert numpy arrays to PIL Images if needed
        
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
                
        # Use default prompt if none provided
        if prompt is None:
            if self.prompt_file_path:
                prompt = self.read_file_to_string(self.prompt_file_path)
            else:
                prompt = """<start_of_turn>user
Describe the contents of this image with details. Be verbose.
<start_of_image>
<end_of_turn>
<start_of_turn>model"""
        
        # Process inputs and generate response
        inputs = self.processor(images=pil_images, text=prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text
    def generate_text_response_without_image(self, prompt: str = None, max_new_tokens: int = 600) -> str:
        """
        Generate text response for the given prompt without requiring images.

        Args:
            prompt: Text prompt to use (optional)
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        # Use default prompt if none provided
        if prompt is None:
            if self.prompt_file_path:
                prompt = self.read_file_to_string(self.prompt_file_path)
            else:
                prompt = """<start_of_turn>user
Describe the contents of the document in detail.
<end_of_turn>
<start_of_turn>model"""

        # Prepare inputs without images
        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return generated_text

    def extract_data(self, images, sys_prompt) -> Dict[str, Any]:
        """
        Extract structured data from image(s) according to specified format.
        
        Args:
            images: Single image or list of images (PIL Image or numpy array)
            sys_prompt: System prompt to use for extraction

        Returns:
            Dictionary containing extracted data
        """
        # Get system prompt from file or use default
        if sys_prompt is None:
            sys_prompt = """Extract all relevant information from this document image. 
            Return the data in JSON format with the following fields: 
            REF_CONTRAT, CURRENCY, AMOUNT_PTFN, AMOUNT_FOB, INVOICE_NUMBER, 
            INVOICE_DATE, SELLER_NAME, SELLER_ADDRESS, BUYER_NAME, BUYER_ADDRESS, 
            MODE_REGLEMENT_CODE, CODE_DELAI_REGLEMENT, CODE_MODE_LIVRAISON, ADVANCE_PAYMENT."""
        
        # Format the prompt for Gemma
        prompt = f"""<start_of_turn>user
{sys_prompt}
"""
        for i in images:
            prompt += "<start_of_image>\n"
        prompt += "<end_of_turn>\n<start_of_turn>modelSMI\n"

        
        # Generate response
        generated_text = self.generate_response(images, prompt)
        # Try to parse JSON from response
        generated_text = generated_text.split("modelSMI")[1]
        try:
            result = self.string_to_JSON(generated_text)
            # Ensure all expected fields are present
            
            return result
        except:
            # Return default empty structure if parsing fails
            return {
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
            
    def llm_compare_texts(self, text1: str, text2: str) -> bool:
        """
        Compare two texts using the model and return similarity result.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            True if texts are similar, False otherwise
        """
        prompt = f"""<start_of_turn>user
You're an expert in text comparison. Compare these two texts and return only "True" if they are similar or "False" if they are not. 
Don't be too strict. Return only True or False.

Text 1: {text1}

Text 2: {text2}
<end_of_turn>
<start_of_turn>modelSMI"""
        
        generated_text = self.generate_text_response_without_image( prompt= prompt)  # Empty image list
        response=generated_text.split("modelSMI")[1].strip().lower() == "true"
        return response