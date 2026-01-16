from typing import Dict, Any, List, Optional, Union, Tuple
import openai
import anthropic
from pathlib import Path
from ..logging import Logger
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ...core.base import NexusModule
from ..metrics import MetricsTracker
from .experiment_generator import ExperimentGenerator

class LLMInterface(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM interface with configuration and clients
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        super().__init__(config)
        self.logger = Logger(__name__)
        
        # Initialize clients with error handling
        try:
            self.openai_client = openai.OpenAI(api_key=config["openai_api_key"])
            self.anthropic_client = anthropic.Anthropic(api_key=config["anthropic_api_key"])
        except KeyError as e:
            self.logger.error(f"Missing required API key: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing LLM clients: {e}")
            raise
            
        # Configuration
        self.default_model = config.get("default_model", "gpt-4")
        self.metrics = MetricsTracker()
        self.experiment_gen = ExperimentGenerator()
        
        # Model configs
        self.model_configs = {
            "gpt-4": {"max_tokens": 8000, "default_temp": 0.7},
            "gpt-3.5-turbo": {"max_tokens": 4000, "default_temp": 0.7}, 
            "claude-3-opus": {"max_tokens": 4000, "default_temp": 0.7},
            "claude-3-sonnet": {"max_tokens": 3000, "default_temp": 0.7}
        }
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((openai.APIError, anthropic.APIError))
    )
    def generate_code(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_artifacts: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate code using specified LLM with enhanced error handling and metrics
        
        Args:
            prompt: Input prompt for code generation
            model: Model name to use (defaults to configured default)
            temperature: Sampling temperature (defaults to model-specific value)
            max_tokens: Maximum tokens to generate (defaults to model limit)
            save_artifacts: Whether to save generation artifacts
            
        Returns:
            Tuple of (generated code, generation metrics)
        """
        model = model or self.default_model
        model_config = self.model_configs.get(model)
        
        if not model_config:
            raise ValueError(f"Unsupported model: {model}")
            
        temperature = temperature or model_config["default_temp"]
        max_tokens = max_tokens or model_config["max_tokens"]
        
        try:
            if "gpt" in model:
                code, metrics = self._generate_openai(prompt, model, temperature, max_tokens)
            elif "claude" in model:
                code, metrics = self._generate_anthropic(prompt, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown model type: {model}")
                
            if save_artifacts:
                self._save_generation_artifacts(prompt, code, metrics, model)
                
            self.metrics.update(metrics)
            return code, metrics
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            raise
            
    def _generate_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate code using OpenAI models"""
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "text"}
        )
        
        metrics = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tokens_used": response.usage.total_tokens,
            "provider": "openai"
        }
        
        return response.choices[0].message.content, metrics
        
    def _generate_anthropic(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate code using Anthropic models"""
        response = self.anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=max_tokens,
            temperature=temperature,
            system=self._get_system_prompt(),
            messages=[{"role": "user", "content": prompt}]
        )
        
        metrics = {
            "model": "claude-3-opus",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
            "provider": "anthropic"
        }
        
        return response.content[0].text, metrics
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for code generation"""
        return """You are an expert Python developer specializing in deep learning and the Nexus framework.
        Generate clean, efficient code following Nexus patterns and best practices.
        Include proper type hints, docstrings, and error handling.
        Ensure compatibility with existing Nexus modules and utilities.
        Follow PEP standards and implement robust error handling."""
        
    def _save_generation_artifacts(
        self,
        prompt: str,
        code: str,
        metrics: Dict[str, Any],
        model: str
    ) -> None:
        """Save generation artifacts for tracking and analysis"""
        try:
            artifacts = {
                "prompt": prompt,
                "generated_code": code,
                "metrics": metrics,
                "model": model
            }
            self.experiment_gen.save_artifact(
                artifacts,
                f"generation_{metrics['provider']}_{model}.json"
            )
        except Exception as e:
            self.logger.warning(f"Failed to save generation artifacts: {e}")