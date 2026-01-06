from typing import Dict, Any
from utils.logger import LoggerSetup

logger = LoggerSetup.get_logger("summarization")

def get_summarizer(config: Dict[str, Any]):
    logger.info(f"Loading summarizer with model: {config['model']['name']}")
    from .summarizer import Summarizer
    model_config = {
        "model": config["model"],
        "inference": config["inference"],
        "optimization": config.get("optimization", {})
    }
    return Summarizer(model_config)
    
__all__ = ["get_summarizer"]
