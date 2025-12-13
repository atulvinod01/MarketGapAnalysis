from __future__ import annotations

import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HNIngestorError(Exception):
    pass


class HNIngestor:
    
    BASE_URL: str = "https://hacker-news.firebaseio.com/v0"
    
    def __init__(
        self,
        max_workers: int = 20,
        timeout: int = 10
    ) -> None:
        self.max_workers: int = max_workers
        self.timeout: int = timeout
        self._session: requests.Session = requests.Session()
        logger.info(f"HNIngestor initialized with {max_workers} workers")
    
    def _fetch_json(self, endpoint: str) -> Optional[Dict[str, Any]]:
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {endpoint}: {e}")
            return None
    
    def _fetch_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        return self._fetch_json(f"item/{item_id}.json")
    
    def fetch_top_story_ids(self, limit: int = 10) -> List[int]:
        logger.info(f"Fetching top {limit} story IDs...")
        story_ids = self._fetch_json("topstories.json")
        
        if story_ids is None:
            raise HNIngestorError("Failed to fetch top stories")
        
        return story_ids[:limit]
    
    def fetch_story_details(self, story_id: int) -> Optional[Dict[str, Any]]:
        story = self._fetch_item(story_id)
        
        if story is None:
            return None
        
        return {
            "id": story.get("id"),
            "title": story.get("title", "Untitled"),
            "url": story.get("url", ""),
            "score": story.get("score", 0),
            "kids": story.get("kids", []),
            "time": story.get("time", 0)
        }
    
    def fetch_comment(self, comment_id: int, story_title: str) -> Optional[Dict[str, Any]]:
        comment = self._fetch_item(comment_id)
        
        if comment is None:
            return None
        
        if comment.get("deleted") or comment.get("dead"):
            return None
        
        text = comment.get("text", "")
        if not text:
            return None
        
        return {
            "story_title": story_title,
            "comment_text": text,
            "comment_id": comment.get("id"),
            "timestamp": datetime.utcfromtimestamp(comment.get("time", 0)),
            "author": comment.get("by", "anonymous")
        }
    
    def fetch_comments_for_story(
        self,
        story: Dict[str, Any],
        max_comments: int = 20
    ) -> List[Dict[str, Any]]:
        comment_ids = story.get("kids", [])[:max_comments]
        story_title = story.get("title", "Untitled")
        
        if not comment_ids:
            logger.debug(f"Story '{story_title[:30]}...' has no comments")
            return []
        
        comments: List[Dict[str, Any]] = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self.fetch_comment, cid, story_title): cid
                for cid in comment_ids
            }
            
            for future in as_completed(future_to_id):
                result = future.result()
                if result:
                    comments.append(result)
        
        logger.debug(f"Fetched {len(comments)} comments for '{story_title[:30]}...'")
        return comments
    
    def fetch_top_stories_with_comments(
        self,
        num_stories: int = 10,
        comments_per_story: int = 20
    ) -> pd.DataFrame:
        start_time = time.time()
        
        story_ids = self.fetch_top_story_ids(limit=num_stories)
        logger.info(f"Got {len(story_ids)} story IDs")
        
        stories: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetch_story_details, sid): sid for sid in story_ids}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    stories.append(result)
        
        logger.info(f"Fetched details for {len(stories)} stories")
        
        all_comments: List[Dict[str, Any]] = []
        for story in stories:
            comments = self.fetch_comments_for_story(story, max_comments=comments_per_story)
            all_comments.extend(comments)
        
        elapsed = time.time() - start_time
        logger.info(
            f"Fetched {len(all_comments)} total comments in {elapsed:.1f}s "
            f"(avg {len(all_comments)/elapsed:.1f} comments/sec)"
        )
        
        if not all_comments:
            logger.warning("No comments fetched!")
            return pd.DataFrame(columns=[
                "story_title", "comment_text", "comment_id", "timestamp", "author"
            ])
        
        df = pd.DataFrame(all_comments)
        return df
    
    def close(self) -> None:
        self._session.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Data Ingestor - Demo")
    print("=" * 60)
    
    ingestor = HNIngestor()
    
    try:
        df = ingestor.fetch_top_stories_with_comments(num_stories=3, comments_per_story=5)
        print(f"\nFetched {len(df)} comments from {df['story_title'].nunique()} stories")
        print("\nSample comments:")
        print(df[["story_title", "comment_text"]].head().to_string())
    except HNIngestorError as e:
        print(f"Error: {e}")
    finally:
        ingestor.close()
