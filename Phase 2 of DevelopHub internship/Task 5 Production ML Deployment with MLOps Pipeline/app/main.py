from fastapi import FastAPI, HTTPException, UploadFile, File
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse
from app.model_loader import model_manager
from app.config import settings
from app.logging_config import logger
import pandas as pd
import io
import json

app = FastAPI(title=settings.PROJECT_NAME)

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "healthy", "model_version": settings.VERSION}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        logger.info(f"Received prediction request: {request.features}")
        prediction, probability = model_manager.predict(request.features)
        return {
            "prediction": prediction,
            "probability": probability,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Separate File Upload Endpoints ---

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """Handle CSV file upload and return summary."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    logger.info(f"CSV uploaded: {file.filename}, shape: {df.shape}")
    
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "summary": df.describe().to_dict()
    }

@app.post("/upload/txt")
async def upload_txt(file: UploadFile = File(...)):
    """Handle TXT file upload."""
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only TXT files are allowed")
    
    contents = await file.read()
    text = contents.decode('utf-8')
    logger.info(f"TXT uploaded: {file.filename}, length: {len(text)}")
    
    return {
        "filename": file.filename,
        "char_count": len(text),
        "preview": text[:100]
    }

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Handle Image (pick) file upload."""
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    contents = await file.read()
    logger.info(f"Image uploaded: {file.filename}, size: {len(contents)} bytes")
    
    return {
        "filename": file.filename,
        "size_bytes": len(contents),
        "content_type": file.content_type
    }

@app.post("/upload/ipynb")
async def upload_ipynb(file: UploadFile = File(...)):
    """Handle IPYNB file upload."""
    if not file.filename.endswith('.ipynb'):
        raise HTTPException(status_code=400, detail="Only IPYNB files are allowed")
    
    contents = await file.read()
    nb_data = json.loads(contents.decode('utf-8'))
    logger.info(f"IPYNB uploaded: {file.filename}")
    
    return {
        "filename": file.filename,
        "cell_count": len(nb_data.get('cells', [])),
        "metadata": nb_data.get('metadata', {})
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
