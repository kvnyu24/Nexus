from typing import Dict, Any, Optional
from ..nlp.llm import BaseLLMConfig

class LLaVARLHFConfig(BaseLLMConfig):
    """Configuration class for LLaVA-RLHF with enhanced features"""
    def __init__(
        self,
        vision_config: Dict[str, Any],
        language_config: Dict[str, Any],
        hidden_size: int = 768,
        num_heads: int = 12,
        max_seq_length: int = 1024,
        reward_scale: float = 0.1,
        kl_coef: float = 0.1,
        use_hallucination_reducer: bool = True,
        use_rag: bool = True,
        bank_size: int = 10000,
        num_retrieval_docs: int = 5,
        quality_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(
            vocab_size=language_config.get("vocab_size", 32000),
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            **kwargs
        )
        self.vision_config = vision_config
        self.language_config = language_config
        self.reward_scale = reward_scale
        self.kl_coef = kl_coef
        self.use_hallucination_reducer = use_hallucination_reducer
        self.use_rag = use_rag
        self.bank_size = bank_size
        self.num_retrieval_docs = num_retrieval_docs
        self.quality_threshold = quality_threshold 