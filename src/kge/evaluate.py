"""
Evaluate KGE models on link prediction: MRR, Hits@1, Hits@3, Hits@10.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(result, testing) -> dict:
    """
    Extract link prediction metrics from a PyKEEN pipeline result.
    Returns dict with MRR, Hits@1, Hits@3, Hits@10 (filtered).
    """
    from pykeen.evaluation import RankBasedEvaluator

    evaluator = RankBasedEvaluator()
    metric_results = evaluator.evaluate(
        model=result.model,
        mapped_triples=testing.mapped_triples,
        additional_filter_triples=[
            result.training.mapped_triples,
        ],
    )

    metrics = {
        "MRR": metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank"),
        "Hits@1": metric_results.get_metric("both.realistic.hits_at_1"),
        "Hits@3": metric_results.get_metric("both.realistic.hits_at_3"),
        "Hits@10": metric_results.get_metric("both.realistic.hits_at_10"),
        "Head_MRR": metric_results.get_metric("head.realistic.inverse_harmonic_mean_rank"),
        "Tail_MRR": metric_results.get_metric("tail.realistic.inverse_harmonic_mean_rank"),
        "Head_Hits@10": metric_results.get_metric("head.realistic.hits_at_10"),
        "Tail_Hits@10": metric_results.get_metric("tail.realistic.hits_at_10"),
    }

    logger.info(f"  MRR: {metrics['MRR']:.4f}")
    logger.info(f"  Hits@1: {metrics['Hits@1']:.4f}")
    logger.info(f"  Hits@3: {metrics['Hits@3']:.4f}")
    logger.info(f"  Hits@10: {metrics['Hits@10']:.4f}")

    return metrics


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison table across all evaluated models."""
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name, **metrics}
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def run_evaluation(
    trained_results: dict | None = None,
    data_dir: str = "kge_datasets",
    models_dir: str = "data/kge_models",
    output_path: str = "data/kge_evaluation.csv",
) -> pd.DataFrame:
    """
    Evaluate all trained models.
    Can accept pre-loaded results dict or load from disk.
    """
    from kge.train_models import load_pykeen_dataset

    training, validation, testing = load_pykeen_dataset(data_dir)

    if trained_results is None:
        # Load models from disk
        from pykeen.pipeline import PipelineResult
        trained_results = {}
        models_path = Path(models_dir)
        for model_dir in models_path.iterdir():
            if model_dir.is_dir() and (model_dir / "trained_model.pkl").exists():
                model_name = model_dir.name
                logger.info(f"Loading model: {model_name}")
                result = PipelineResult.from_directory(str(model_dir))
                trained_results[model_name] = result

    all_metrics = {}
    for model_name, result in trained_results.items():
        logger.info(f"Evaluating {model_name}:")
        metrics = evaluate_model(result, testing)
        all_metrics[model_name] = metrics

    comparison_df = compare_models(all_metrics)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out, index=False)
    logger.info(f"Evaluation results saved to {out}")

    # Display
    logger.info(f"\n{'=' * 70}")
    logger.info("Model Comparison (Link Prediction — Filtered Metrics)")
    logger.info(f"{'=' * 70}")
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    return comparison_df


if __name__ == "__main__":
    run_evaluation()
