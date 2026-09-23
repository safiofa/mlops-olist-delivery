import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère les variables temporelles, géographiques et financières 
    nécessaires pour le modèle de prédiction des retards.
    """
    df = df.copy()
    
    # 1. Variables temporelles
    if 'order_purchase_timestamp' in df.columns:
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['purchase_month'] = df['order_purchase_timestamp'].dt.month
        df['purchase_day'] = df['order_purchase_timestamp'].dt.day
        df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
        df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
        df['is_weekend'] = df['purchase_dayofweek'].isin([5, 6]).astype(int)
    
    # 2. Variable géographique
    if 'customer_state' in df.columns and 'seller_state' in df.columns:
        df['same_state'] = (df['customer_state'] == df['seller_state']).astype(int)
    else:
        df['same_state'] = 0
        
    # 3. Ratios financiers
    if 'total_items' in df.columns:
        items_safe = np.maximum(df['total_items'], 1)
        if 'total_price' in df.columns:
            df['price_per_item'] = df['total_price'] / items_safe
        if 'total_freight' in df.columns:
            df['freight_per_item'] = df['total_freight'] / items_safe
            
    return df