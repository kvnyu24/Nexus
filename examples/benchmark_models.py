from nexus.benchmarks import ImageClassificationBenchmark
from nexus.models.cv import CompactCNN, VisionTransformer

# Configure benchmark
config = {
    "image_size": 224,
    "benchmark_config": {
        "batch_sizes": [1, 8, 16, 32],
        "num_iterations": 100,
        "warmup_iterations": 10,
        "measure_memory": True
    }
}

# Create benchmark
benchmark = ImageClassificationBenchmark(config)

# Benchmark models
models = {
    "compact_cnn": CompactCNN({"num_classes": 10}),
    "vit": VisionTransformer({"image_size": 224, "patch_size": 16, "num_classes": 10})
}

results = {}
for name, model in models.items():
    results[name] = benchmark.run_benchmark(model)
    
# Print results
for model_name, metrics in results.items():
    print(f"\nResults for {model_name}:")
    print(f"Mean latency: {metrics['performance']['mean_latency']:.4f}s")
    print(f"Throughput: {metrics['performance']['throughput']:.2f} samples/s")
    print(f"Peak memory: {metrics['performance']['peak_memory'] / 1e6:.2f} MB")
    print(f"Accuracy: {metrics['accuracy']['top1']:.2f}%") 