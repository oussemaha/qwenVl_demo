from unittest import result
from configs.constants import *
from src.featureExtraction_Validation.llm_extraction import LLM_extractor
from src.featureExtraction_Validation.gemma_extraction import GemmaImageProcessor
from src.featureExtraction_Validation.json_validator import JSONComparator
from src.featureExtraction_Validation.ocr import OCRProcessor
from src.classification.ocr_classification import Classifier
import kagglehub
import pandas as pd
import os
import json
import torch
import gc
import csv
import telegram
import asyncio
import time
from datetime import timedelta

# Initialize global variables
llm_extractor = None
classifier = None
validator = None
ocr_processor = OCRProcessor()

async def send_message(api_key, user_id, message):
    bot = telegram.Bot(token=api_key)
    async with bot:
        await bot.send_message(chat_id=user_id, text=message)

def initialize_models():
    """Initialize or reinitialize all models"""
    global llm_extractor, classifier, validator
    if llm_extractor is not None:
        del llm_extractor
    if classifier is not None:
        del classifier
    if validator is not None:
        del validator
    
    torch.cuda.empty_cache()
    gc.collect()

    llm_extractor = LLM_extractor(MODEL_NAME)
    classifier = Classifier(UNIVERSAL_KEYWORDS, UNIVERSAL_PATTERNS, 
                          CLASSIFICATION_KEYWORD_THRESHOLD, 
                          CLASSIFICATION_MIN_TEXT_LENGTH,
                          gpu=CLASSIFICATION_GPU)
    validator = JSONComparator(llm_extractor)
    asyncio.run(send_message(api_key, user_id, "Models reinitialized after GPU cleanup."))

def clean_gpu_if_high_usage(threshold_gb=35):
    """Cleans GPU memory if allocated VRAM > threshold_gb"""
    torch.cuda.empty_cache()
    gc.collect()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"Current GPU memory allocated: {allocated:.2f} GB")
        if allocated > threshold_gb:
            initialize_models()
def read_file_to_string( file_path: str) -> str:
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
        
if __name__ == "__main__":
    api_key = "7847416478:AAGw5Rxu81D1DKyQJtPWmnMD9qrCqq905TU"
    user_id = "6473083926"
    
    # Initialize models first time
    initialize_models()
    prompts=[]
    try:
        for i in range(7):
           prompts.append(read_file_to_string(f"/home/ubuntu/qwenVl_demo/configs/prompts/prompt{i+1}.txt"))
           
        dataset_dir = kagglehub.dataset_download("oussemahamouda/factures-biat")
        dataset_dir = dataset_dir + "/pj_smi_biat"
        df = pd.read_csv(dataset_dir + "/all_data.csv")
        total_items = len(df["REF_CONTRAT"])
        
        # Get starting index from user or environment variable
        start_index =0
        if start_index >= total_items:
            raise ValueError(f"Start index {start_index} is out of range (total items: {total_items})")
        
        asyncio.run(send_message(api_key, user_id, 
                    f"Starting dataset processing from index {start_index} (total items: {total_items})..."))
        
        header = ["REF_CONTRAT", "CURRENCY", "AMOUNT_PTFN", "AMOUNT_FOB", "INVOICE_NUMBER", "INVOICE_DATE", 
                 "SELLER_NAME", "SELLER_ADDRESS", "BUYER_NAME", "BUYER_ADDRESS", "MODE_REGLEMENT_CODE", 
                 "CODE_DELAI_REGLEMENT", "CODE_MODE_LIVRAISON", "ADVANCE_PAYMENT"]
        
        # Open file in append mode if starting from non-zero index
        file_mode = 'a' if start_index > 0 else 'w'
        with open("result.csv", file_mode, newline='') as result_file, \
             open("llm_output.csv", file_mode, newline='') as llm_file:

            result_writer = csv.writer(result_file)
            llm_writer = csv.writer(llm_file)

            # Write headers if new files
            if file_mode == 'w':
                result_writer.writerow(header)
                llm_writer.writerow(header)

            start_time = time.time()
            processed_items = 0

            for index in range(start_index, total_items):
                contract_ref = df["REF_CONTRAT"].iloc[index]
                item_start_time = time.time()
                images = []

                # Find and load JSON data
                data = dict()
                for root, dirs, files in os.walk(dataset_dir + f"/{contract_ref}"):
                    for file in files:
                        if file.endswith(".json"):
                            with open(os.path.join(root, file)) as json_data:
                                data = json.load(json_data)
                                break
                            
                # Process images
                # Open all images in the contract_ref directory except JSON files
                images = []
                contract_dir = os.path.join(dataset_dir, f"{contract_ref}")
                for file in os.listdir(contract_dir):
                    if not file.lower().endswith(".json"):
                        full_path = os.path.join(contract_dir, file)
                        images.append(classifier.open_file_as_image(full_path))

                llm_json = dict()
                """
                for img in images:
                    ocr_results = ocr_processor.extract_text(img)
                    result = "".join(ocr_results)"""
                # Extract and compare data
                for sys_prompt in prompts:
                    llm_json.update(llm_extractor.extract_data(images, sys_prompt=sys_prompt))
                print(llm_json)

                # Process comparison results
                comparison_result = validator.compare_jsons(data["data"][0], llm_json)
                comparison_result["REF_CONTRAT"] = contract_ref
                row=[]
                for column in header:
                    if column in llm_json:
                        row.append(llm_json[column])
                    else:
                        row.append(None)
                llm_writer.writerow(row)    
                row = []
                for column in header:
                    if column in comparison_result:
                        row.append(comparison_result[column])
                    else:
                        row.append(None)
                result_writer.writerow(row)

                # Calculate progress and time estimates
                processed_items += 1
                current_index = index + 1  # Convert to 1-based counting
                elapsed_time = time.time() - start_time
                avg_time_per_item = elapsed_time / processed_items
                remaining_items = total_items - current_index
                estimated_remaining_time = avg_time_per_item * remaining_items

                # Format time for display
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                remaining_str = str(timedelta(seconds=int(estimated_remaining_time)))

                progress_percent = (current_index/total_items)*100
                progress_msg = (
                    f"Progress: {progress_percent:.2f}% | "
                    f"Items: {current_index}/{total_items} | "
                    f"Elapsed: {elapsed_str} | "
                    f"Remaining: ~{remaining_str} | "
                    f"Last item: {time.time() - item_start_time:.2f}s"
                )

                print(progress_msg)


        
        completion_msg = (
            f"Processing completed from index {start_index}!\n"
            f"Total time: {timedelta(seconds=int(time.time() - start_time))}\n"
            f"Processed items: {total_items - start_index}/{total_items}"
        )
        asyncio.run(send_message(api_key, user_id, completion_msg))
        print("\n" + completion_msg)
    
    except Exception as e:
        error_msg = f"Error processing dataset at index {index if 'index' in locals() else 'N/A'}: {str(e)}"
        asyncio.run(send_message(api_key, user_id, error_msg))
        print(error_msg)