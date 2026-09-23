import pandas as pd
import pytest
from src.pipeline import InferencePipeline

def test_pipeline_prediction():
    pipeline = InferencePipeline()
    
    # Données de test complètes avec toutes les variables requises
    sample_data = pd.DataFrame([{
        "order_purchase_timestamp": "2026-09-01 10:00:00",
        "total_items": 1,
        "total_price": 50.0,
        "total_freight": 15.0,
        "customer_state": "RJ",
        "seller_state": "SP",
        "avg_product_weight_g": 500.0,
        "main_payment_type": "credit_card"
    }])
    
    predictions, probabilities = pipeline.predict(sample_data)
    
    # Vérifications des sorties
    assert len(predictions) == 1
    assert len(probabilities) == 1
    assert predictions[0] in [0, 1]
    assert 0.0 <= probabilities[0] <= 1.0