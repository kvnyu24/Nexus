import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import json
import traceback
import threading

class Logger:
    _instances: Dict[str, 'Logger'] = {}
    _lock = threading.Lock()

    def __new__(cls, name: Optional[str] = None, *args, **kwargs):
        """Implement singleton pattern per logger name"""
        with cls._lock:
            if name in cls._instances:
                return cls._instances[name]
            instance = super().__new__(cls)
            cls._instances[name] = instance
            return instance

    def __init__(
        self,
        name: Optional[str] = None,
        log_file: Optional[Union[str, Path]] = None,
        log_level: int = logging.INFO,
        format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        propagate: bool = False,
        rotation: str = 'daily',  # none, daily, size
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        json_format: bool = False
    ):
        # Skip initialization if already initialized
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        
        self.name = name or __name__
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(log_level)
        self.logger.propagate = propagate
        self.json_format = json_format
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # Create formatters
        if json_format:
            formatter = self._json_formatter
        else:
            formatter = logging.Formatter(format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if log_file specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            if rotation == 'daily':
                from logging.handlers import TimedRotatingFileHandler
                file_handler = TimedRotatingFileHandler(
                    log_file, when='midnight', interval=1,
                    backupCount=backup_count
                )
            elif rotation == 'size':
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=max_bytes,
                    backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(log_file)
                
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _json_formatter(self, record):
        """Custom JSON formatter"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }
        if record.exc_info:
            log_data['exception'] = traceback.format_exception(*record.exc_info)
        return json.dumps(log_data)
            
    def set_level(self, level: Union[int, str]):
        """Set logging level"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
            
    def add_handler(self, handler: logging.Handler):
        """Add custom handler"""
        if self.json_format:
            handler.setFormatter(self._json_formatter)
        self.logger.addHandler(handler)
        
    def remove_handler(self, handler: logging.Handler):
        """Remove a handler"""
        self.logger.removeHandler(handler)

    def get_handlers(self) -> List[logging.Handler]:
        """Get all handlers"""
        return self.logger.handlers
        
    def info(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        self.logger.info(msg, *args, extra=extra, **kwargs)
        
    def warning(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        self.logger.warning(msg, *args, extra=extra, **kwargs)
        
    def error(self, msg: str, *args, exc_info: bool = True, extra: Dict[str, Any] = None, **kwargs):
        self.logger.error(msg, *args, exc_info=exc_info, extra=extra, **kwargs)
        
    def debug(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        self.logger.debug(msg, *args, extra=extra, **kwargs)
        
    def critical(self, msg: str, *args, exc_info: bool = True, extra: Dict[str, Any] = None, **kwargs):
        self.logger.critical(msg, *args, exc_info=exc_info, extra=extra, **kwargs)
        
    def exception(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        self.logger.exception(msg, *args, extra=extra, **kwargs)