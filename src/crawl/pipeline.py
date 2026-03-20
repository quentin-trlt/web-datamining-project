"""
Full pipeline for Lab 1: Crawl → Clean → NER → Relations.
Run this script to execute the entire data acquisition & IE pipeline.
"""

import argparse
import logging

from src.crawl.crawler import crawl
from src.ie.ner import run_extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Lab 1: Web Crawling & Information Extraction")
    parser.add_argument(
        "--skip-crawl", action="store_true",
        help="Skip crawling and use existing crawler_output.jsonl",
    )
    parser.add_argument(
        "--crawl-output", default="data/crawler_output.jsonl",
        help="Path to crawler output JSONL file",
    )
    args = parser.parse_args()

    # Phase 1: Crawl & Clean
    if not args.skip_crawl:
        logger.info("=" * 60)
        logger.info("PHASE 1: Web Crawling & Cleaning")
        logger.info("=" * 60)
        crawl(output_path=args.crawl_output)
    else:
        logger.info("Skipping crawl phase (using existing data)")

    # Phase 2: Information Extraction
    logger.info("=" * 60)
    logger.info("PHASE 2: Information Extraction (NER + Relations)")
    logger.info("=" * 60)
    run_extraction(input_path=args.crawl_output)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  - Crawler output: data/crawler_output.jsonl")
    logger.info("  - Entities:       data/extracted_entities.csv")
    logger.info("  - Relations:      data/extracted_relations.csv")


if __name__ == "__main__":
    main()
