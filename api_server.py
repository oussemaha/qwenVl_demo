

import os
from src.classification.ocr_classification import Classifier
from configs.constants import *

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import uuid
from typing import List, Optional, Dict, Any
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
classifier = Classifier(UNIVERSAL_KEYWORDS, UNIVERSAL_PATTERNS, 
                          CLASSIFICATION_KEYWORD_THRESHOLD, 
                          CLASSIFICATION_MIN_TEXT_LENGTH,
                          gpu=CLASSIFICATION_GPU)
app = FastAPI(
    title="Document Extraction API",
    description="API for extracting data from documents and comparing with existing forms",
    version="1.0.0"
)

# Response Models
class ExtractionResponse(BaseModel):
    result: List[Dict[str, Any]]
    message: str

class ComparisonResponse(BaseModel):
    result: List[Dict[str, Any]]
    message: str

class ComparisonRequest(BaseModel):
    form_data: Dict[str, Any]

#Controller

@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_files(
    files: List[UploadFile] = File(..., description="Files to extract data from")
):
    """
    Extract data from uploaded files
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        results = {}
        file_id = str(uuid.uuid4())
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
                    # Write uploaded content to temporary file
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {str(e)}")
            try:
                images.extend(classifier.open_file_as_image(temp_file_path))
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        if not results:
            raise HTTPException(status_code=400, detail="No valid files processed")
        
        
        
    except Exception as e:
        logger.error(f"Error in extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/extract_and_compare", response_model=ComparisonResponse)
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
        
        # Parse form data
        try:
            form_data_dict = json.loads(form_data)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON form data: {str(e)}")
        
        # Process files and extract data
        extracted_results = {}
        file_id = str(uuid.uuid4())
        
        for file in files:
            content = await file.read()
            extracted_data = extract_data_from_file(content, file.filename)
            extracted_results[file.filename] = extracted_data
        
        if not extracted_results:
            raise HTTPException(status_code=400, detail="No valid files processed")
        
        # Compare extracted data with form data
        comparison_results = {}
        for filename, extracted_data in extracted_results.items():
            comparison_result = compare_extracted_with_form(extracted_data, form_data_dict)
            comparison_results[filename] = comparison_result
        
        return ComparisonResponse(
            success=True,
            extracted_data=extracted_results,
            comparison_result=comparison_results,
            file_id=file_id,
            message=f"Successfully processed and compared {len(extracted_results)} file(s)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract and compare: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Document Extraction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": str(uuid.uuid4())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)