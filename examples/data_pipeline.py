from nexus.data.inputs import InputProcessor, InputConfig
from nexus.data.augmentation import AugmentationPipeline, MixupAugmentation
from nexus.data.cache import DataCache
from nexus.data.streaming import StreamingDataset

# Setup input processing
input_config = InputConfig(
    input_type="image",
    image_size=(224, 224),
    normalize=True,
    augment=True
)

processor = InputProcessor(input_config)
augmentation = AugmentationPipeline(
    image_size=(224, 224),
    augmentation_strength=0.8
)
mixup = MixupAugmentation(alpha=0.2)
cache = DataCache()

# Create data pipeline
def data_generator():
    while True:
        # Get data from source
        data = get_next_batch()
        
        # Process inputs
        processed = processor.process(data)
        
        # Apply augmentation
        if input_config.augment:
            processed["image"] = augmentation(processed["image"])
            
        # Cache processed data
        cache_key = cache._get_cache_key(processed)
        cache.save(cache_key, processed)
        
        yield processed

# Create streaming dataset
dataset = StreamingDataset(
    data_generator(),
    buffer_size=1000
)

# Use dataset in training
for batch in dataset:
    # Apply mixup augmentation
    if training:
        images, labels_a, labels_b, lam = mixup(
            batch["image"],
            batch["labels"]
        )
        # Train with mixed samples 