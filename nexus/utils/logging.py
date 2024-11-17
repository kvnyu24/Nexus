import logging
from typing import Optional, Dict, Any

class Logger:
    def __init__(self, name: Optional[str] = None):
        self.logger = logging.getLogger(name or __name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, msg: str, **kwargs):
        self.logger.info(msg, **kwargs)
        
    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, **kwargs)
        
    def error(self, msg: str, **kwargs):
        self.logger.error(msg, **kwargs)
        
    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, **kwargs) 