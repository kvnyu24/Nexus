from typing import Dict, Type, Any

class Registry:
    def __init__(self):
        self._registry: Dict[str, Type[Any]] = {}
        
    def register(self, name: str):
        def wrapper(cls):
            self._registry[name] = cls
            return cls
        return wrapper
        
    def get(self, name: str) -> Type[Any]:
        if name not in self._registry:
            raise KeyError(f"Component '{name}' not found in registry")
        return self._registry[name]

# Create global registries
MODELS = Registry()
COMPONENTS = Registry() 