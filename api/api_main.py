from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat
from xgboost import XGBClassifier
from typing import Literal
import uvicorn
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="Bank Marketing Prediction API")

# Input schema - Only include features used by the model
class CustomerData(BaseModel):
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: conint(ge=1)
    pdays: conint(ge=-1)
    previous: conint(ge=0)
    poutcome: str
    cons_conf_idx: confloat(ge=-60.0, le=0.0)
    nr_employed: confloat(ge=4000, le=5500)
    age_group: str

# Load model artifacts
def load_artifacts():
    try:
        possible_paths = [
            Path("/app/ml_model/saved_models"),
            Path("/app/app/ml_model/saved_models"), 
            Path("ml_model/saved_models"),
        ]
        
        for path in possible_paths:
            model_path = path / "voting_model.pkl"
            if model_path.exists():
                logger.info(f"✅ Found models at: {path}")

                model = joblib.load(model_path)
                
                return {
                    'model': model,
                    'features': joblib.load(path / "model_features.pkl"),
                    'encoders': joblib.load(path / "label_encoders.pkl"),
                    'scaler': joblib.load(path / "scaler.pkl")
                }

        raise FileNotFoundError(f"❌ Model files not found in any of: {possible_paths}")
    except Exception as e:
        logger.error(f"❌ Failed to load artifacts: {str(e)}", exc_info=True)
        raise


artifacts = load_artifacts()

def make_prediction(raw_data: dict):
    try:
        input_df = pd.DataFrame([raw_data])
        logger.info(f"🧾 Raw input columns: {input_df.columns.tolist()}")

        # Rename columns to match training format
        rename_map = {
            "cons_conf_idx": "cons.conf.idx",
            "nr_employed": "nr.employed"
        }
        input_df = input_df.rename(columns=rename_map)

        # Label encode categorical features
        for col, encoder in artifacts['encoders'].items():
            if col in input_df.columns:
                input_df[col] = input_df[col].apply(lambda x: x if x in encoder.classes_ else "unknown")
                if "unknown" not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, "unknown")
                input_df[col] = encoder.transform(input_df[col])

        # ⚠️ Step 1: Scale using original scaler features
        scaler_features = (
            artifacts['scaler'].feature_names_in_
            if hasattr(artifacts['scaler'], "feature_names_in_")
            else input_df.columns
        )

        missing_for_scaler = set(scaler_features) - set(input_df.columns)
        if missing_for_scaler:
            raise ValueError(f"Missing features for scaler: {missing_for_scaler}")

        input_df_for_scaling = input_df[scaler_features]
        scaled_all = artifacts['scaler'].transform(input_df_for_scaling)
        scaled_df = pd.DataFrame(scaled_all, columns=scaler_features)

        # ✅ Step 2: Subset to model features
        input_for_model = scaled_df[artifacts['features']]

        logger.info(f"📌 Final input to model: {input_for_model.columns.tolist()}")

        prediction = artifacts['model'].predict(input_for_model)[0]
        probability = artifacts['model'].predict_proba(input_for_model)[0][1]

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4)
        }

    except Exception as e:
        logger.error(f"❗ Prediction failed: {str(e)}", exc_info=True)
        raise ValueError({
            "error": str(e),
            "message": "Something went wrong during prediction. Please verify your input values or contact the maintainer."
        })

@app.post("/predict")
def predict(data: CustomerData):
    try:
        return make_prediction(data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing (only use this in dev, not Docker)
# if __name__ == "__main__":
#    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)