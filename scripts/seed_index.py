#!/usr/bin/env python3
"""
Script to seed the vector index with content from avatar corpus files.
Usage: python scripts/seed_index.py --avatar avatars/alex.yaml --sources corpus/alex/links.csv
"""

import argparse
import csv
import sys
import os
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
import time

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.indexer import get_indexer
from app.io.scraper import get_scraper
from app.io.files import get_file_manager
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_avatar_info(avatar_path):
    """Load avatar information from YAML file"""
    file_manager = get_file_manager()
    avatar_data = file_manager.load_yaml(avatar_path)
    
    if not avatar_data:
        raise ValueError(f"Failed to load avatar from {avatar_path}")
    
    return {
        "name": avatar_data.get("name", Path(avatar_path).stem),
        "role": avatar_data.get("role", "Debater"),
        "tags": [avatar_data.get("name", Path(avatar_path).stem).lower()]
    }

def load_sources(sources_path):
    """Load sources from CSV file"""
    sources = []
    
    try:
        with open(sources_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            
            url_col = 0
            title_col = None
            trust_col = None
            
            # Find column indices
            for i, col in enumerate(header):
                col = col.lower()
                if 'url' in col or 'link' in col:
                    url_col = i
                elif 'title' in col:
                    title_col = i
                elif 'trust' in col or 'score' in col:
                    trust_col = i
            
            # Read rows
            for row in reader:
                if not row or len(row) <= url_col:
                    continue
                    
                source = {"url": row[url_col].strip()}
                
                if title_col is not None and len(row) > title_col:
                    source["title"] = row[title_col].strip()
                    
                if trust_col is not None and len(row) > trust_col:
                    try:
                        source["trust_score"] = int(row[trust_col])
                    except ValueError:
                        pass  # Ignore if not a number
                        
                sources.append(source)
    except Exception as e:
        logger.error(f"Error loading sources from {sources_path}: {e}")
        return []
    
    return sources

def seed_index(avatar_path, sources_path, max_sources=None, delay=1.0, freshness_days=None):
    """Seed vector index with content from sources"""
    try:
        # Load avatar info and sources
        avatar_info = load_avatar_info(avatar_path)
        sources = load_sources(sources_path)
        
        if not sources:
            logger.error(f"No sources found in {sources_path}")
            return False
        
        logger.info(f"Found {len(sources)} sources for {avatar_info['name']}")
        
        # Limit number of sources if specified
        if max_sources and len(sources) > max_sources:
            logger.info(f"Limiting to {max_sources} sources")
            sources = sources[:max_sources]
        
        # Get services
        indexer = get_indexer()
        scraper = get_scraper()
        
        # Set freshness limit
        if freshness_days is None:
            freshness_days = settings.FRESHNESS_DAYS
        
        # Process each source
        successful = 0
        
        for i, source in enumerate(tqdm(sources, desc="Indexing sources")):
            try:
                url = source["url"]
                if not url:
                    continue
                
                # Add delay between requests to avoid rate limiting
                if i > 0 and delay > 0:
                    time.sleep(delay)
                
                # Fetch content
                html = scraper.fetch_url(url)
                if not html:
                    logger.warning(f"Failed to fetch content from {url}")
                    continue
                
                # Extract content
                content = scraper.extract_content(html, url)
                if not content or not content.get("text"):
                    logger.warning(f"Failed to extract content from {url}")
                    continue
                
                # Add metadata
                content["url"] = url
                content["trust_score"] = source.get("trust_score", 7)  # Default to 7
                content["avatar_tags"] = avatar_info["tags"]
                
                # Index content
                indexed = indexer.index_document(
                    content,
                    freshness_days=freshness_days,
                    metadata={
                        "avatar": avatar_info["name"],
                        "role": avatar_info["role"],
                        "trust_score": content["trust_score"]
                    }
                )
                
                if indexed:
                    successful += 1
                    
            except Exception as e:
                logger.error(f"Error processing source {source.get('url')}: {e}")
        
        logger.info(f"Successfully indexed {successful}/{len(sources)} sources")
        return successful > 0
        
    except Exception as e:
        logger.error(f"Error seeding index: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Seed vector index with content from avatar corpus")
    parser.add_argument("--avatar", required=True, help="Path to avatar YAML file")
    parser.add_argument("--sources", required=True, help="Path to sources CSV file")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of sources to index")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--freshness", type=int, default=None, help="Maximum age of sources in days")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.avatar).exists():
        print(f"❌ Avatar file not found: {args.avatar}")
        sys.exit(1)
        
    if not Path(args.sources).exists():
        print(f"❌ Sources file not found: {args.sources}")
        sys.exit(1)
    
    # Run the indexing
    print(f"🔍 Seeding index for {args.avatar} from {args.sources}")
    success = seed_index(args.avatar, args.sources, args.max, args.delay, args.freshness)
    
    if success:
        print("✅ Index seeded successfully")
        sys.exit(0)
    else:
        print("❌ Failed to seed index")
        sys.exit(1)

if __name__ == "__main__":
    main()
