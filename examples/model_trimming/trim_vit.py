import torch
from nexus.models.cv.vit import VisionTransformer
from nexus.core.trimming import AdvancedModelTrimmer

def main():
    # Initialize ViT model
    vit_config = {
        "image_size": 224,
        "patch_size": 16,
        "in_channels": 3,
        "num_classes": 1000,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "use_flash_attention": True
    }
    
    model = VisionTransformer(vit_config)
    
    # Initialize trimmer with configuration
    trimmer_config = {
        "sparsity_target": 0.3,  # Target 30% sparsity
        "min_channels": 8,
        "importance_metric": "l1_norm",
        "granularity": "channel",
        "preserve_outputs": True
    }
    
    trimmer = AdvancedModelTrimmer(trimmer_config)
    
    # Create example input
    batch_size = 4
    example_input = torch.randn(batch_size, 3, 224, 224)
    
    # Trim model
    trimmed_model, metrics = trimmer(model, example_input)
    
    # Print metrics
    print("Trimming Results:")
    print(f"Original parameters: {model.get_parameter_count()['total']:,}")
    print(f"Trimmed parameters: {trimmed_model.get_parameter_count()['total']:,}")
    print("\nLayer-wise sparsity:")
    for layer_name, layer_metrics in metrics.items():
        if "importance" in layer_metrics:
            sparsity = (layer_metrics["masks"]["weight"] == 0).float().mean()
            print(f"{layer_name}: {sparsity:.2%} sparse")

if __name__ == "__main__":
    main()
