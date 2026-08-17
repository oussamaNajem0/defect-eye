"""FastAPI application for DefectEye."""

from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib  # Assuming you saved your XGBoost model using joblib
import logging

from defect_eye.schemas import PredictionRequest, PredictionResponse
# from defect_eye.models.explainability import get_local_explanation

# Initialize logger and FastAPI app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DefectEye API",
    description="ML-powered software defect prediction API.",
    version="1.0.0"
)

# Global variable to hold the model
MODEL = None

@app.on_event("startup")
def load_artifacts():
    """Load the trained model into memory on startup."""
    global MODEL
    try:
        # Assuming you saved the model in a 'models' directory
        MODEL = joblib.load("models/xgboost_model.pkl")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")

@app.get("/health", tags=["System"])
def health_check():
    """Check if the API and model are up and running."""
    if MODEL is None:
        raise HTTPException(status_status=503, detail="Model not loaded.")
    return {"status": "healthy", "model_status": "loaded"}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_defect(request: PredictionRequest):
    """Predict the likelihood of a defect based on static code metrics."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.model_dump()])
        
        # Perform inference
        prob = MODEL.predict_proba(input_data)[0][1]
        pred = MODEL.predict(input_data)[0]
        
        # Optional: Integrate local SHAP explanation here
        # shap_vals = get_local_explanation(MODEL, input_data)
        
        return PredictionResponse(
            defect_probability=float(prob),
            prediction=int(pred),
            shap_values={} # Replace with actual shap_vals if implemented
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing prediction request.")