import logging
from pipelines.load._init__ import verify_dataset_exists
from .vqvae_pipeline import get_vqvae_model, apply_vqvae
from .base_preprocess import preprocess_all

logger = logging.getLogger("AugmentationPipeline")

def run_preprocessing_pipeline(
    dataset: str,
    augmentation: str,
    batch_size: int = 64,
    train_size: float = 0.8,
    test_size: float = 0.2,
    device: str = "cpu",
):
    logger.info("Running preprocessing pipeline...")

    verify_dataset_exists(dataset)

    if augmentation != "vqvae":
        preprocess_all(
            dataset=dataset,
            augmentation=augmentation,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            device=device,
        )

    if augmentation in ["vqvae", "all"]:
        model = get_vqvae_model(dataset, device=device)
        apply_vqvae(dataset, model, device=device, batch_size=batch_size)

    logger.info("✅ Preprocessing pipeline complete.")
