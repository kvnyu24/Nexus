import torch
import torch.nn as nn
from typing import Optional, Union, Callable

class WeightInitializer:
    @staticmethod
    def initialize_weights(
        module: nn.Module,
        method: str = 'kaiming_normal',
        nonlinearity: str = 'relu',
        scale: float = 1.0
    ):
        for name, param in module.named_parameters():
            if 'weight' in name:
                if method == 'kaiming_normal':
                    nn.init.kaiming_normal_(
                        param,
                        mode='fan_out',
                        nonlinearity=nonlinearity
                    )
                    param.data *= scale
                elif method == 'xavier_normal':
                    nn.init.xavier_normal_(param)
                    param.data *= scale
                elif method == 'orthogonal':
                    nn.init.orthogonal_(param)
                    param.data *= scale
                    
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    @staticmethod
    def apply_weight_norm(module: nn.Module, name: str = 'weight'):
        return nn.utils.weight_norm(module, name=name)
        
    @staticmethod
    def remove_weight_norm(module: nn.Module):
        for m in module.modules():
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                pass 