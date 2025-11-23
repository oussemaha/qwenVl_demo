

import logging
import os

from src.classification.ocr_classification import Classifier
from src.api_src.service import Service, extract, read_file_to_string
from configs.constants import *


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import uuid
from typing import List, Optional, Dict, Any
import tempfile


service=Service()
logging.basicConfig(level=logging.INFO)
logger =logging.getLogger(__name__)
classifier = Classifier(UNIVERSAL_KEYWORDS, UNIVERSAL_PATTERNS, 
                  CLASSIFICATION_KEYWORD_THRESHOLD, 
                  CLASSIFICATION_MIN_TEXT_LENGTH,
                  gpu=CLASSIFICATION_GPU)
prompts=[]

for file in os.listdir(PROMPTS_DIR):
           prompts.append(read_file_to_string(f"{PROMPTS_DIR}/{file}"))

app = FastAPI(
    title="Document Extraction API",
    description="API for extracting data from documents and comparing with existing forms",
    version="1.0.0"
)

# Response Models
class Response(BaseModel):
    result: List[Dict[str, Any]]
    message: str



class ComparisonRequest(BaseModel):
    form_data: Dict[str, Any]


def read_files(files):
    images=[]
    for file in files:
        # Validate file type (basic example)
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.gif'}
        file_extension = f".{file.filename.split('.')[-1].lower()}" if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            logger.warning(f"Unsupported file type: {file_extension}")
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            try:
                content = file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {str(e)}")
        try:
            images.extend(classifier.open_file_as_image(temp_file_path))
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    return images
#Controller

@app.post("/extract", response_model=Response)
async def extract_from_files(
    files: List[UploadFile] = File(..., description="Files to extract data from")
):
    """
    Extract data from uploaded files
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        images=await read_files(files)
        results=service.extract_json(images, prompts)
        if not results:
            raise HTTPException(status_code=400, detail="No valid files processed")
        
        return Response(
            result=results,
            message=f"Successfully processed {len(results)} file(s)"
        )
           
    except Exception as e:
        logger.error(f"Error in extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/extract_and_compare", response_model=Response)
async def extract_and_compare(
    files: List[UploadFile] = File(..., description="Files to extract data from"),
    form_data: str = Form(..., description="JSON form data to compare against")
):
    """
    Extract data from files and compare with provided JSON form data
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        images=await read_files(files)
        results=service.extract_and_compare(images, prompts, json.loads(form_data))
        if not results:
            raise HTTPException(status_code=400, detail="No valid files processed")
        return Response(
            result=results,
            message=f"Successfully processed {len(results)} file(s)"
        )
           
    except Exception as e:
        logger.error(f"Error in extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction & comparison failed: {str(e)}")        


@app.get("/")
async def root():
    return {"message": "Document Extraction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": str(uuid.uuid4())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)