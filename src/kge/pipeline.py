"""
Knowledge Graph Embedding pipeline orchestrator.
Runs: data preparation → training → evaluation → experiments.
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="KGE Pipeline")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-experiments", action="store_true")
    parser.add_argument("--models", default="TransE,ComplEx", help="Comma-separated model names")
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--expanded-nt", default="kg_artifacts/expanded.nt")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    config = {
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
    }

    # Step 1: Data preparation
    if not args.skip_prepare:
        logger.info("=" * 60)
        logger.info("Step 1: Data Preparation")
        from kge.prepare_data import run_data_preparation
        run_data_preparation(expanded_nt=args.expanded_nt)

    # Step 2: Training
    trained_results = None
    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("Step 2: Training KGE Models")
        from kge.train_models import run_training
        trained_results = run_training(models=model_names, config=config)

    # Step 3: Evaluation
    if not args.skip_evaluate:
        logger.info("=" * 60)
        logger.info("Step 3: Evaluation")
        from kge.evaluate import run_evaluation
        run_evaluation(trained_results=trained_results)

    # Step 4: Experiments
    if not args.skip_experiments:
        logger.info("=" * 60)
        logger.info("Step 4: Experiments & Analysis")
        from kge.experiments import run_experiments
        run_experiments(trained_results=trained_results)

    logger.info("=" * 60)
    logger.info("KGE pipeline complete!")


if __name__ == "__main__":
    main()
