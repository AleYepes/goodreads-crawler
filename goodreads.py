import argparse
import json
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page, Response, TimeoutError as PlaywrightTimeoutError

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path("data")
EXPORT_FILE = DATA_DIR / "goodreads_library_export.csv"
OUTPUT_FILE = DATA_DIR / "books_data.csv"

DATA_DIR.mkdir(exist_ok=True)


class GoodreadsExporter:
    """Handles downloading Goodreads library export."""
    
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
    
    def download_library(self, output_path: Path = EXPORT_FILE) -> pd.DataFrame:
        """Download Goodreads library export and return as DataFrame."""
        print("Starting Goodreads library export...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                accept_downloads=True
            )
            page = context.new_page()
            
            # Login
            print("  Logging in...")
            self._login(page)
            
            # Navigate to export page
            print("  Navigating to export page...")
            page.goto("https://www.goodreads.com/review/import", wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            
            # Check if export already exists today
            current_date = datetime.now().strftime("%m/%d/%Y")
            file_list = page.locator(".fileList").first
            
            if file_list.count() > 0 and current_date not in file_list.text_content():
                print("  Triggering new export...")
                export_button = page.locator(".js-LibraryExport").first
                export_button.click()
            
            # Wait for export to be ready
            print("  Waiting for export to be ready...")
            page.wait_for_selector(".fileList", timeout=60000)
            
            # Poll until today's export appears
            max_attempts = 20
            for attempt in range(max_attempts):
                file_list = page.locator(".fileList").first
                if current_date in file_list.text_content():
                    print("  Export ready! Downloading...")
                    break
                page.wait_for_timeout(3000)
            else:
                raise TimeoutError("Export did not complete in time")
            
            # Download the file
            with page.expect_download() as download_info:
                link = file_list.locator("a").first
                link.click()
            
            download = download_info.value
            download.save_as(output_path)
            
            browser.close()
            print(f"  ✓ Export saved to {output_path}")
        
        # Load and return as DataFrame
        return pd.read_csv(output_path)
    
    def _login(self, page: Page):
        """Handle Goodreads/Amazon login."""
        login_url = (
            "https://www.goodreads.com/ap/signin?"
            "language=en_US&openid.assoc_handle=amzn_goodreads_web_na"
            "&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select"
            "&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select"
            "&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0"
            "&openid.pape.max_auth_age=0"
            "&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in"
        )
        
        page.goto(login_url, wait_until="domcontentloaded")
        
        # Fill in credentials
        page.fill('input[type="email"]', self.email)
        page.fill('input[type="password"]', self.password)
        page.click('input[type="submit"]')
        
        # Wait for successful login (presence of home page element)
        try:
            page.wait_for_selector(".homePrimaryColumn", timeout=30000)
            print("  ✓ Login successful")
        except PlaywrightTimeoutError:
            raise Exception("Login failed - check credentials or handle 2FA manually")


class GoodreadsScraper:
    """Scrapes detailed book data from Goodreads."""
    
    def __init__(self, page: Page):
        self.page = page
        self.captured_payloads = []
        self.page.on("response", self._handle_network_response)
    
    def _handle_network_response(self, response: Response):
        """Background listener: captures similar books data."""
        if self._is_similar_books_request(response):
            try:
                self.captured_payloads.append(response.json())
            except Exception:
                pass
    
    def _is_similar_books_request(self, response: Response) -> bool:
        """Check if response is the getSimilarBooks GraphQL call."""
        if "graphql" in response.url and response.request.method == "POST":
            try:
                req_json = response.request.post_data_json
                return isinstance(req_json, dict) and req_json.get("operationName") == "getSimilarBooks"
            except Exception:
                return False
        return False
    
    def _safe_get_author(self, ld: dict) -> str:
        """Extract author name from LD-JSON."""
        auth = ld.get("author")
        if isinstance(auth, dict):
            return auth.get("name", "")
        if isinstance(auth, list) and auth:
            item = auth[0]
            if isinstance(item, dict):
                return item.get("name", "")
            return str(item)
        return ""
    
    def scrape(self, book_id: int) -> Tuple[Optional[dict], List[int]]:
        """Scrape book details and return data dict + similar book IDs."""
        print(f"  Scraping ID: {book_id}")
        self.captured_payloads.clear()
        url = f"https://www.goodreads.com/book/show/{book_id}"
        
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=45000)
            
            # Extract LD-JSON metadata
            script = self.page.locator('script[type="application/ld+json"]').first
            if script.count() == 0:
                print("    No LD-JSON found - skipping")
                return None, []
            
            try:
                ld = json.loads(script.text_content())
            except json.JSONDecodeError:
                return None, []
            
            # Handle list or single object
            if isinstance(ld, list):
                ld = next((item for item in ld if isinstance(item, dict) and item.get("@type") == "Book"), {})
            if not isinstance(ld, dict):
                return None, []
            
            # Build data dict
            agg_rating = ld.get("aggregateRating", {})
            if not isinstance(agg_rating, dict):
                agg_rating = {}
            
            data = {
                "id": book_id,
                "title": ld.get("name"),
                "author": self._safe_get_author(ld),
                "avg_rating": agg_rating.get("ratingValue"),
                "total_reviews": agg_rating.get("reviewCount"),
                "pages": ld.get("numberOfPages"),
            }
            
            # Star ratings histogram
            for star in range(5, 0, -1):
                el = self.page.locator(f"[data-testid='labelTotal-{star}']").first
                count = 0
                if el.count() > 0:
                    text = el.text_content()
                    clean_text = re.sub(r'[^\d]', '', text.split('(')[0])
                    if clean_text:
                        count = int(clean_text)
                data[f"{star}_star"] = count
            
            # Genres
            genre_locs = self.page.locator(".BookPageMetadataSection__genreButton .Button__labelItem")
            genres = [t for t in genre_locs.all_text_contents() if t != "...more"]
            data["genres"] = "|".join(genres) if genres else ""
            
            # Series
            series_loc = self.page.locator("h3.Text__italic a").first
            data["series"] = ""
            if series_loc.count() > 0:
                href = series_loc.get_attribute("href")
                if href:
                    data["series"] = href.split('/')[-1]
            
            # Publication year
            pub_loc = self.page.locator("[data-testid='publicationInfo']").first
            data["year"] = ""
            if pub_loc.count() > 0:
                try:
                    data["year"] = int(pub_loc.text_content().split(', ')[-1])
                except (ValueError, IndexError):
                    pass
            
            # Similar books (network capture)
            if not self.captured_payloads:
                try:
                    with self.page.expect_response(
                        lambda response: self._is_similar_books_request(response),
                        timeout=5000
                    ):
                        pass
                except PlaywrightTimeoutError:
                    pass
            
            similar_ids = set()
            for payload in self.captured_payloads:
                conn = payload.get("data", {}).get("getSimilarBooks", {})
                edges = conn.get("edges", []) if isinstance(conn, dict) else []
                
                for edge in edges:
                    node = edge.get("node", {}) if isinstance(edge, dict) else {}
                    web_url = node.get("webUrl", "")
                    if web_url:
                        match = re.search(r'show/(\d+)', web_url)
                        if match:
                            try:
                                similar_ids.add(int(match.group(1)))
                            except ValueError:
                                pass
            
            if similar_ids:
                print(f"    Found {len(similar_ids)} similar books")
            
            data["similar_books"] = "|".join(map(str, similar_ids))
            return data, list(similar_ids)
        
        except Exception as e:
            print(f"  Error scraping {book_id}: {e}")
            return None, []


def run_crawler(start_ids: List[int], output_path: Path = OUTPUT_FILE):
    """Run the book crawler starting from given IDs."""
    
    # Load existing data if present
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        visited = set(existing_df['id'].astype(int))
        print(f"Loaded {len(visited)} existing records")
    else:
        existing_df = pd.DataFrame()
        visited = set()
    
    queue = deque(start_ids)
    print(f"Crawler started. Queue: {len(queue)} | Already visited: {len(visited)}")
    
    # Initialize browser
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        scraper = GoodreadsScraper(page)
        
        new_records = []
        
        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            
            data, similar_ids = scraper.scrape(current_id)
            visited.add(current_id)
            
            if data:
                new_records.append(data)
                
                # Add new books to queue
                added = 0
                for sid in similar_ids:
                    if sid not in visited and sid not in queue:
                        queue.append(sid)
                        added += 1
                
                if added:
                    print(f"    Added {added} new books to queue")
                
                # Periodic save every 10 books
                if len(new_records) % 10 == 0:
                    _save_dataframe(existing_df, new_records, output_path)
                    print(f"  [Checkpoint] Saved {len(new_records)} new records")
        
        # Final save
        if new_records:
            _save_dataframe(existing_df, new_records, output_path)
        
        browser.close()
        print(f"\n✓ Finished. Total books: {len(visited)}")


def _save_dataframe(existing_df: pd.DataFrame, new_records: List[dict], output_path: Path):
    """Append new records to existing DataFrame and save."""
    new_df = pd.DataFrame(new_records)
    
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    combined_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Scrape Goodreads book data")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading library export if it already exists"
    )
    parser.add_argument(
        "--start-ids",
        nargs="+",
        type=int,
        help="Manually specify starting book IDs (overrides library export)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of books to scrape (for testing)"
    )
    
    args = parser.parse_args()
    
    # Determine starting IDs
    if args.start_ids:
        start_ids = args.start_ids
        print(f"Using manual start IDs: {start_ids}")
    else:
        # Download or load library export
        if args.skip_download and EXPORT_FILE.exists():
            print(f"Using existing export: {EXPORT_FILE}")
            library_df = pd.read_csv(EXPORT_FILE)
        else:
            email = os.getenv("GOODREADS_EMAIL")
            password = os.getenv("GOODREADS_PASSWORD")
            
            if not email or not password:
                print("ERROR: Set GOODREADS_EMAIL and GOODREADS_PASSWORD in .env file")
                return
            
            exporter = GoodreadsExporter(email, password)
            library_df = exporter.download_library()
        
        # Extract book IDs from library
        start_ids = library_df['Book Id'].astype(int).tolist()
        print(f"Extracted {len(start_ids)} books from library export")
        
        if args.limit:
            start_ids = start_ids[:args.limit]
            print(f"Limited to first {args.limit} books")
    
    # Run crawler
    run_crawler(start_ids)


if __name__ == "__main__":
    main()