"""
Train Knowledge Graph Embedding models (TransE, ComplEx) using PyKEEN.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default training configuration
DEFAULT_CONFIG = {
    "embedding_dim": 200,
    "lr": 0.001,
    "batch_size": 256,
    "epochs": 100,
    "num_negatives": 10,
}


def load_pykeen_dataset(data_dir: str) -> tuple:
    """Load train/valid/test splits into PyKEEN TriplesFactory objects."""
    from pykeen.triples import TriplesFactory

    train_path = Path(data_dir) / "train.txt"
    valid_path = Path(data_dir) / "valid.txt"
    test_path = Path(data_dir) / "test.txt"

    training = TriplesFactory.from_path(train_path)
    validation = TriplesFactory.from_path(
        valid_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    testing = TriplesFactory.from_path(
        test_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )

    logger.info(
        f"Dataset loaded: train={training.num_triples}, "
        f"valid={validation.num_triples}, test={testing.num_triples}"
    )
    logger.info(f"  Entities: {training.num_entities}, Relations: {training.num_relations}")

    return training, validation, testing


def train_model(
    model_name: str,
    training,
    validation,
    config: dict | None = None,
):
    """Train a single KGE model using pykeen.pipeline."""
    from pykeen.pipeline import pipeline

    cfg = {**DEFAULT_CONFIG, **(config or {})}

    logger.info(f"Training {model_name} with config: {cfg}")

    result = pipeline(
        model=model_name,
        training=training,
        validation=validation,
        model_kwargs={"embedding_dim": cfg["embedding_dim"]},
        training_kwargs={
            "num_epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
        },
        optimizer_kwargs={"lr": cfg["lr"]},
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negatives_per_positive": cfg["num_negatives"]},
        random_seed=42,
    )

    logger.info(f"{model_name} training complete")
    return result


def save_model(result, output_dir: str, model_name: str) -> Path:
    """Save trained model and metrics."""
    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)

    result.save_to_directory(str(out))
    logger.info(f"Model {model_name} saved to {out}")
    return out


def run_training(
    data_dir: str = "kge_datasets",
    output_dir: str = "data/kge_models",
    models: list[str] | None = None,
    config: dict | None = None,
) -> dict:
    """Train all specified KGE models."""
    if models is None:
        models = ["TransE", "ComplEx"]

    training, validation, testing = load_pykeen_dataset(data_dir)

    results = {}
    for model_name in models:
        logger.info(f"{'=' * 60}")
        logger.info(f"Training model: {model_name}")
        logger.info(f"{'=' * 60}")

        result = train_model(model_name, training, validation, config)
        save_model(result, output_dir, model_name)
        results[model_name] = result

    # Save training config for reproducibility
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg_path = Path(output_dir) / "training_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    logger.info(f"All models trained: {list(results.keys())}")
    return results


if __name__ == "__main__":
    run_training()
