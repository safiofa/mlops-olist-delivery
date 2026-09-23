import joblib
import yaml
import time
import pandas as pd
from pathlib import Path
from src.features import create_features
from src.utils import setup_logger

logger = setup_logger()

class InferencePipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                
            self.preprocessor_path = Path(self.config["paths"]["preprocessor_path"])
            self.model_path = Path(self.config["paths"]["model_path"])
            self.threshold = self.config["model_params"].get("threshold", 0.5)
            
            # Chargement des artefacts avec logs
            logger.info(f"Chargement du préprocesseur depuis : {self.preprocessor_path}")
            self.preprocessor = joblib.load(self.preprocessor_path)
            
            logger.info(f"Chargement du modèle depuis : {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info("Pipeline d'inférence initialisé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du pipeline : {str(e)}")
            raise e

    def predict(self, raw_df: pd.DataFrame):
        start_time = time.time()
        try:
            if raw_df.empty:
                logger.warning("Le DataFrame reçu est vide.")
                return [], []

            # 1. Feature engineering
            df_featured = create_features(raw_df)
            
            # 2. Preprocessing
            X_processed = self.preprocessor.transform(df_featured)
            
            # 3. Prédiction
            probabilities = self.model.predict_proba(X_processed)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)
            
            latency = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Prédiction réussie pour {len(raw_df)} enregistrement(s) en {latency} ms.")
            
            return predictions, probabilities

        except Exception as e:
            logger.error(f"Échec lors de la prédiction : {str(e)}")
            raise e