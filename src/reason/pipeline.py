"""
Reasoning pipeline orchestrator.
Runs: SWRL on family.owl + custom SWRL on AI-News KB.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # Part 1: SWRL on family.owl
    logger.info("Starting SWRL reasoning pipeline")

    from reason.family_swrl import run_family_swrl
    family_results = run_family_swrl()

    # Part 2: Custom SWRL on AI-News KB
    from reason.custom_swrl import run_custom_swrl
    custom_results = run_custom_swrl()

    logger.info("=" * 60)
    logger.info("Reasoning pipeline complete!")
    logger.info(f"  Family OldPerson inferred: {len(family_results)}")
    logger.info(f"  AI Company inferred: {len(custom_results)}")


if __name__ == "__main__":
    main()
