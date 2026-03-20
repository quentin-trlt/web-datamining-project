"""
Web crawler with robots.txt compliance and content cleaning via trafilatura.
Outputs cleaned text to JSONL format.
"""

import json
import logging
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED_URLS = [
    # Stanford AI Index
    "https://aiindex.stanford.edu/report/",
    # Sequoia Capital – Generative AI
    "https://www.sequoiacap.com/article/generative-ai-a-creative-new-world/",
    # MIT Technology Review – What's next for AI
    "https://www.technologyreview.com/2024/01/04/1086046/whats-next-for-ai-in-2024/",
    # BBC – AI chatbots trying to take over search
    "https://www.bbc.com/news/technology-65855333",
    # The Verge – Chatbots & conversational AI
    "https://www.theverge.com/23610427/chatbots-chatgpt-new-bing-google-bard-conversational-ai",
    # Stanford HAI – State of AI in 13 charts
    "https://hai.stanford.edu/news/ai-index-state-ai-13-charts",
    # IEEE Spectrum – AI Index 2024
    "https://spectrum.ieee.org/ai-index-2024",
    # BBC – AI risks
    "https://www.bbc.com/news/technology-67012224",
    # Pew Research – Public awareness of AI
    "https://www.pewresearch.org/science/2023/02/15/public-awareness-of-artificial-intelligence-in-everyday-activities/",
    # IBM – What is Artificial Intelligence
    "https://www.ibm.com/think/topics/artificial-intelligence",
    # ZDNet – Everything you need to know about AI
    "https://www.zdnet.com/article/what-is-ai-heres-everything-you-need-to-know-about-artificial-intelligence/",
    # The Verge – Microsoft Bing AI
    "https://www.theverge.com/2023/1/24/23567448/microsoft-bing-search-ai-chatgpt-openai",
]

USER_AGENT = "AIBubbleResearchBot/1.0 (academic project; respects robots.txt)"
MIN_WORD_COUNT = 500
REQUEST_DELAY = 2  # seconds between requests to same domain


def check_robots_txt(url: str) -> bool:
    """Check if the URL is allowed by the site's robots.txt."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception as e:
        logger.warning(f"Could not read robots.txt for {parsed.netloc}: {e}")
        return True  # assume allowed if robots.txt unreachable


def fetch_page(url: str, client: httpx.Client) -> str | None:
    """Fetch raw HTML from a URL."""
    try:
        response = client.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def extract_clean_text(html: str, url: str) -> dict | None:
    """Use trafilatura to extract main content and metadata from HTML."""
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        output_format="txt",
    )
    if text is None:
        logger.warning(f"Trafilatura could not extract content from {url}")
        return None

    metadata = trafilatura.extract(
        html,
        include_comments=False,
        output_format="xmltei",
    )
    title = trafilatura.metadata.extract_metadata(html)

    return {
        "url": url,
        "title": title.title if title and title.title else "",
        "text": text,
        "word_count": len(text.split()),
    }


def is_useful(page_data: dict) -> bool:
    """Check if the extracted content is substantial enough (>= MIN_WORD_COUNT words)."""
    return page_data["word_count"] >= MIN_WORD_COUNT


def crawl(urls: list[str] | None = None, output_path: str = "data/crawler_output.jsonl") -> Path:
    """
    Crawl the given URLs (or SEED_URLS by default), clean content, and save to JSONL.

    Returns the path to the output file.
    """
    if urls is None:
        urls = SEED_URLS

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    last_domain = None

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for url in urls:
            domain = urlparse(url).netloc
            logger.info(f"Processing: {url}")

            # Respect robots.txt
            if not check_robots_txt(url):
                logger.warning(f"Blocked by robots.txt: {url}")
                continue

            # Polite delay between requests to the same domain
            if last_domain == domain:
                time.sleep(REQUEST_DELAY)
            last_domain = domain

            # Fetch
            html = fetch_page(url, client)
            if html is None:
                continue

            # Extract & clean
            page_data = extract_clean_text(html, url)
            if page_data is None:
                continue

            # Filter by usefulness
            if not is_useful(page_data):
                logger.info(
                    f"Skipped (only {page_data['word_count']} words): {url}"
                )
                continue

            results.append(page_data)
            logger.info(
                f"Saved: {page_data['title']!r} ({page_data['word_count']} words)"
            )

    # Write JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Crawling complete: {len(results)}/{len(urls)} pages saved to {output_file}")
    return output_file


if __name__ == "__main__":
    crawl()
