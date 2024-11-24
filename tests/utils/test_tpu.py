import pytest
import torch
from nexus.utils.tpu import TPUManager, TPUModule
from nexus.core.base import NexusModule

class SimpleModel(TPUModule):
    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(10, 2)
        
    def forward(self, x):
        return self.linear(x)

def test_tpu_manager_initialization():
    config = {"tpu_cores": 8}
    manager = TPUManager(config)
    
    # Test device initialization
    assert hasattr(manager, "device")
    assert manager.config == config

def test_tpu_module_integration():
    config = {"tpu_cores": 8}
    model = SimpleModel(config)
    
    # Test TPU conversion
    tpu_model = model.to_tpu()
    assert isinstance(tpu_model, TPUModule)
    
    # Test optimizer creation
    optimizer = model.create_tpu_optimizer(
        torch.optim.Adam,
        lr=0.001
    )
    assert isinstance(optimizer, torch.optim.Optimizer)

def test_tpu_metrics():
    config = {"tpu_cores": 8}
    manager = TPUManager(config)
    metrics = manager.get_tpu_metrics()
    
    assert isinstance(metrics, dict)
    if manager.initialized:
        assert "device_type" in metrics
        assert "memory_allocated" in metrics 