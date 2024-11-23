from typing import Dict, Any, List, Optional, Union
import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMInterface:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_client = openai.OpenAI(api_key=config["openai_api_key"])
        self.anthropic_client = anthropic.Anthropic(api_key=config["anthropic_api_key"])
        self.default_model = config.get("default_model", "gpt-4")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_code(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate code using specified LLM"""
        model = model or self.default_model
        
        if "gpt" in model:
            return self._generate_openai(prompt, model, temperature, max_tokens)
        elif "claude" in model:
            return self._generate_anthropic(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported model: {model}")
            
    def _generate_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
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
        return response.choices[0].message.content
        
    def _generate_anthropic(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        response = self.anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=max_tokens,
            temperature=temperature,
            system=self._get_system_prompt(),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
        
    def _get_system_prompt(self) -> str:
        return """You are an expert Python developer specializing in deep learning and the Nexus framework.
        Generate clean, efficient code following Nexus patterns and best practices.
        Include proper type hints, docstrings, and error handling.""" 