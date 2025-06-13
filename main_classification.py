from pathlib import Path
from src.classification.unzipInput import unzip_file
from src.classification.ocr_classification import Classifier
from configs.constants import *
import time as t
import json
import pandas as pd 
import os

classifier= Classifier(UNIVERSAL_KEYWORDS,UNIVERSAL_PATTERNS,CLASSIFICATION_KEYWORD_THRESHOLD,CLASSIFICATION_MIN_TEXT_LENGTH,gpu=CLASSIFICATION_GPU)


if __name__ == "__main__":
    dataset_dir="/home/oussema/Desktop/projet_domi_AI/datasets/pj_smi_biat"
    df=pd.read_csv(dataset_dir+"/all_data.csv")


    for i in df["REF_CONTRAT"] :
        paths=[]
        print(i)
        jsonpath=""
        files = []
        for root,dirs,files in os.walk(dataset_dir+f"/{i}") :
            data=dict()
            for file in files:
                if file.split('.')[1]=="json":
                   
                    with open(root+"/"+file) as json_data:
                        jsonpath=root+"/"+file
                        data = json.load(json_data)
                        break
        paths.extend(classifier.organize_invoices_para_temp(Path(dataset_dir+f"/{i}")))
        paths =[str(path) for path in paths]
        if len(paths) == 0:
            paths= [str(dataset_dir+f"/{file}") for file in files if file.split('.')[1] not in ["json"]]
        paths = list(set(paths))  # Remove duplicates
        print(paths)
        data["invoice_paths"]= paths
        with open(jsonpath, 'w') as outfile:
            json.dump(data, outfile, indent=4)
                