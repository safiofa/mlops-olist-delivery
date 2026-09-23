from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
from typing import List
from src.pipeline import InferencePipeline
from src.utils import setup_logger

logger = setup_logger()

app = FastAPI(
    title="Olist Delivery Delay Prediction API",
    description="API MLOps pour prédire les retards de livraison Olist",
    version="1.0.0"
)

pipeline = InferencePipeline()

# Schéma Pydantic pour la validation des requêtes HTTP
class OrderData(BaseModel):
    order_purchase_timestamp: str = Field(..., example="2026-09-01 10:00:00")
    total_items: int = Field(..., ge=1, example=1)
    total_price: float = Field(..., ge=0.0, example=50.0)
    total_freight: float = Field(..., ge=0.0, example=15.0)
    customer_state: str = Field(..., example="RJ")
    seller_state: str = Field(..., example="SP")
    avg_product_weight_g: float = Field(..., ge=0.0, example=500.0)
    main_payment_type: str = Field(..., example="credit_card")

class PredictionResponse(BaseModel):
    is_late: int
    probability: float

@app.get("/", status_code=status.HTTP_200_OK)
def health_check():
    """
    Endpoint Health Check pour vérifier l'état de l'API.
    """
    return {"status": "healthy", "service": "olist-delay-predictor"}

@app.post("/predict", response_model=List[PredictionResponse], status_code=status.HTTP_200_OK)
def predict_delay(orders: List[OrderData]):
    """
    Endpoint principal d'inférence (accepte une liste de commandes).
    """
    try:
        # Conversion de la liste de Pydantic vers un DataFrame Pandas
        raw_data = [order.dict() for order in orders]
        df_input = pd.DataFrame(raw_data)
        
        # Exécution du pipeline
        predictions, probabilities = pipeline.predict(df_input)
        
        results = [
            {"is_late": int(pred), "probability": round(float(prob), 4)}
            for pred, prob in zip(predictions, probabilities)
        ]
        
        return results

    except Exception as e:
        logger.error(f"Erreur API lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))