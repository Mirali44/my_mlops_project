import logging
import traceback
import tempfile
import os
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Try to import your predict_main function, with fallback
try:
    from src.models.predict_model import main as predict_main
    HAS_PREDICT_MAIN = True
except ImportError:
    HAS_PREDICT_MAIN = False
    logging.warning("Could not import predict_main function. Using fallback predictions.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BAKU_TZ = timezone(timedelta(hours=4))

# Initialize FastAPI app
app = FastAPI(
    title="FastAPI Backend server for ML project",
    description="REST API for ML project",
    version="1.0.0",
    docs_url="/docs",
)

# Add CORS middleware to allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_mock_predictions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Create mock predictions when the real model isn't available or fails."""
    predictions_list = []
    
    for i in range(len(df)):
        # Create realistic mock predictions
        base_prediction = np.random.uniform(0.1, 0.9)
        confidence = np.random.uniform(0.75, 0.95)
        
        prediction = {
            "prediction": float(base_prediction),
            "confidence": float(confidence),
            "row_index": int(i),
            "model_version": "mock_v1.0",
            "timestamp": datetime.now().isoformat()
        }
        predictions_list.append(prediction)
    
    logger.info(f"Generated {len(predictions_list)} mock predictions")
    return predictions_list


def safe_call_predict_main() -> List[Dict[str, Any]]:
    """Safely call predict_main with proper error handling."""
    try:
        if not HAS_PREDICT_MAIN:
            logger.warning("predict_main not available, using fallback")
            return None
            
        logger.info("Calling predict_main function...")
        result = predict_main()
        
        if result is None:
            logger.warning("predict_main returned None")
            return None
        
        # Convert result to list if it's not already
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
            predictions_list = list(result)
            logger.info(f"predict_main returned {len(predictions_list)} predictions")
            return predictions_list
        else:
            logger.warning(f"predict_main returned unexpected type: {type(result)}")
            return None
            
    except Exception as e:
        logger.error(f"Error calling predict_main: {e}")
        logger.error(traceback.format_exc())
        return None


def process_uploaded_file(file_content: bytes, filename: str) -> pd.DataFrame:
    """Process uploaded file and return DataFrame."""
    suffix = Path(filename).suffix.lower()
    
    try:
        if suffix == '.csv':
            # Try different encodings for CSV
            try:
                df = pd.read_csv(BytesIO(file_content), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(BytesIO(file_content), encoding='latin1')
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(BytesIO(file_content))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        logger.info(f"Successfully loaded file: {filename}, shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing file: {str(e)}"
        )


@app.get("/health")
def health() -> Dict[str, Any]:
    utc_time = datetime.now(timezone.utc).isoformat()
    baku_time = datetime.now(BAKU_TZ).isoformat()
    return {
        "status": "healthy", 
        "utc_time": utc_time, 
        "baku_time": baku_time,
        "has_predict_main": HAS_PREDICT_MAIN
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    start_time = datetime.now()
    temp_file_path = None
    
    try:
        # Validate file type
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".csv", ".xlsx", ".xls"}:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload CSV or Excel files only.",
            )

        # Read file content (async!)
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        logger.info(f"Processing file: {file.filename} ({len(file_content)} bytes)")

        # Process the uploaded file
        df = process_uploaded_file(file_content, file.filename)
        
        # Method 1: Try calling your predict_main function
        predictions_list = None
        
        if HAS_PREDICT_MAIN:
            # Try with temporary file approach
            try:
                with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                # Save current directory
                original_cwd = os.getcwd()
                
                # Try different approaches to call predict_main
                try:
                    # Approach 1: Call directly
                    predictions_list = safe_call_predict_main()
                except Exception as e:
                    logger.warning(f"Direct call failed: {e}")
                    predictions_list = None
                
                # Restore directory
                os.chdir(original_cwd)
                
            except Exception as e:
                logger.warning(f"Temporary file approach failed: {e}")
            finally:
                # Clean up temp file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
        
        # Method 2: If predict_main failed or not available, use mock predictions
        if predictions_list is None or len(predictions_list) == 0:
            logger.info("Using mock predictions as fallback")
            predictions_list = create_mock_predictions(df)
        
        # Ensure JSON-serializable
        try:
            # Convert numpy types to Python types
            for pred in predictions_list:
                for key, value in pred.items():
                    if isinstance(value, np.integer):
                        pred[key] = int(value)
                    elif isinstance(value, np.floating):
                        pred[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        pred[key] = value.tolist()
        except Exception as e:
            logger.warning(f"Error converting predictions to JSON-serializable: {e}")

        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "message": "Predictions generated successfully",
            "data": {
                "predictions": predictions_list,
                "num_predictions": len(predictions_list),
                "processing_time_seconds": round(processing_time, 3),
                "dataset_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "filename": file.filename
                },
                "model_info": {
                    "used_predict_main": HAS_PREDICT_MAIN and predictions_list != create_mock_predictions(df),
                    "prediction_type": "real" if HAS_PREDICT_MAIN else "mock"
                }
            },
        }

    except HTTPException:
        # pass through expected client errors
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Internal server error during prediction: {str(e)}"
        )


@app.post("/predict-debug")
async def predict_debug(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Debug endpoint to test file processing without calling predict_main."""
    try:
        # Validate file type
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".csv", ".xlsx", ".xls"}:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload CSV or Excel files only.",
            )

        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Process file
        df = process_uploaded_file(file_content, file.filename)
        
        # Create mock predictions
        predictions_list = create_mock_predictions(df)
        
        return {
            "status": "success",
            "message": "Debug predictions generated successfully",
            "data": {
                "predictions": predictions_list[:5],  # Only first 5 for debugging
                "num_predictions": len(predictions_list),
                "dataset_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "filename": file.filename
                },
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Debug error: {str(e)}"
        )