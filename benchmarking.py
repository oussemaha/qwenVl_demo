from configs.constants import *
from src.featureExtraction_Validation.llm_extraction import LLM_extractor
from src.featureExtraction_Validation.json_validator import JSONComparator
from src.classification.ocr_classification import Classifier
import kagglehub
import pandas as pd
import os
import json
import torch
import gc


llm_extractor=LLM_extractor(MODEL_NAME,SYSTEM_PROMPT_PATH)
classifier= Classifier(UNIVERSAL_KEYWORDS,UNIVERSAL_PATTERNS,CLASSIFICATION_KEYWORD_THRESHOLD,CLASSIFICATION_MIN_TEXT_LENGTH,gpu=CLASSIFICATION_GPU)
validator=JSONComparator(llm_extractor)

def clean_gpu_if_high_usage(threshold_gb=21):
    """Cleans GPU memory only if allocated VRAM > threshold_gb (silently)."""
    if torch.cuda.memory_allocated() / (1024 ** 3) > threshold_gb:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
if __name__ == "__main__":
    dataset_dir = kagglehub.dataset_download("oussemahamouda/factures-biat")
    dataset_dir = dataset_dir+"/pj_smi_biat"
    df = pd.read_csv(dataset_dir+"/all_data.csv")
    result_df = pd.DataFrame(columns=["REF_CONTRAT", "CURRENCY", "AMOUNT_PTFN","AMOUNT_FOB","INVOICE_NUMBER","INVOICE_DATE","SELLER_NAME",
                                   "SELLER_ADDRESS	","BUYER_NAME","BUYER_ADDRESS","MODE_REGLEMENT_CODE","CODE_DELAI_REGLEMENT","CODE_MODE_LIVRAISON","ADVANCE_PAYMENT"])
    index= 0
    for i in df["REF_CONTRAT"] :
        index+=1
        images=[]
        clean_gpu_if_high_usage(20)
        for root,dirs,files in os.walk(dataset_dir+f"/{i}") :
            data=dict()
            for file in files:
                if file.split('.')[1]=="json":
                    with open(root+"/"+file) as json_data:
                        data = json.load(json_data)
                        break
        for file_path in data["invoice_paths"]:
            file_path=dataset_dir+ f"/{i}/"+file_path.split('/')[-1]  # Get the file name
            images.append(classifier.open_file_as_image(file_path))
        llm_json=llm_extractor.extract_data(images,"")

        comparison_result=validator.compare_jsons(data["data"][0],llm_json)
        result_df.loc[len(df)] = comparison_result
        print(f"{(index/len(df))* 100:.2f}% completed", end='\r')

    pd.DataFrame.to_csv(result_df,"result.csv", index=False)

