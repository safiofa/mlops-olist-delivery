import logging
import sys
from pathlib import Path

def setup_logger(log_file: str = "app.log") -> logging.Logger:
    """
    Configure un logger qui écrit à la fois dans un fichier et dans la console.
    """
    logger = logging.getLogger("mlops_olist")
    logger.setLevel(logging.INFO)
    
    # Éviter de doubler les handlers si le logger existe déjà
    if logger.hasHandlers():
        return logger

    # Format structuré pour les logs
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler 1 : Sortie Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler 2 : Fichier Log
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger