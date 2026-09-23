import pandas as pd
from src.features import create_features

def test_create_features():
    # Création d'un mini DataFrame de test
    data = {
        "order_purchase_timestamp": ["2026-09-01 10:00:00"],
        "customer_state": ["SP"],
        "seller_state": ["SP"],
        "total_items": [2],
        "total_price": [100.0],
        "total_freight": [20.0]
    }
    df = pd.DataFrame(data)
    
    # Application de la transformation
    df_transformed = create_features(df)
    
    # Vérifications (assertions)
    assert "purchase_month" in df_transformed.columns
    assert "same_state" in df_transformed.columns
    assert df_transformed["same_state"].iloc[0] == 1
    assert df_transformed["price_per_item"].iloc[0] == 50.0