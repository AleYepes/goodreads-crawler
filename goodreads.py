import csv
import json
import os
import re
from collections import deque
from typing import List, Optional, Tuple
from playwright.sync_api import sync_playwright, Page, Response

OUTPUT_FILE = "books_data.csv"
CSV_HEADERS = [
    "id", "title", "author", "avg_rating", "total_reviews", "pages", 
    "5_star", "4_star", "3_star", "2_star", "1_star", 
    "genres", "series", "year", "similar_books"
]

class GoodreadsScraper:
    def __init__(self, page: Page):
        self.page = page
        self.captured_payloads = []
        self.page.on("response", self._handle_network_response)

    def _handle_network_response(self, response: Response):
        if "graphql" in response.url and response.request.method == "POST":
            try:
                req_json = response.request.post_data_json
                if isinstance(req_json, dict) and req_json.get("operationName") == "getSimilarBooks":
                    self.captured_payloads.append(response.json())
            except Exception:
                pass

    def _safe_get_author(self, ld: dict) -> str:
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
        print(f"  Scraping ID: {book_id}")
        self.captured_payloads.clear() 
        url = f"https://www.goodreads.com/book/show/{book_id}"

        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=45000)

            # LD-JSON Extraction
            script = self.page.locator('script[type="application/ld+json"]').first
            if script.count() == 0:
                print("    No LD-JSON script tag found.")
                return None, []

            try:
                ld = json.loads(script.text_content())
            except json.JSONDecodeError:
                return None, []

            if isinstance(ld, list):
                ld = next((item for item in ld if isinstance(item, dict) and item.get("@type") == "Book"), {})
            
            if not isinstance(ld, dict):
                return None, []

            agg_rating = ld.get("aggregateRating", {})
            if not isinstance(agg_rating, dict): agg_rating = {}

            row = {
                "id": book_id,
                "title": ld.get("name"),
                "author": self._safe_get_author(ld),
                "avg_rating": agg_rating.get("ratingValue"),
                "total_reviews": agg_rating.get("reviewCount"),
                "pages": ld.get("numberOfPages"),
            }

            # Histogram
            for star in range(5, 0, -1):
                el = self.page.locator(f"[data-testid='labelTotal-{star}']").first
                count = 0
                if el.count() > 0:
                    text = el.text_content()
                    clean_text = re.sub(r'[^\d]', '', text.split('(')[0])
                    if clean_text: count = int(clean_text)
                row[f"{star}_star"] = count

            # Genres
            genre_locs = self.page.locator(".BookPageMetadataSection__genreButton .Button__labelItem")
            genres = [t for t in genre_locs.all_text_contents() if t != "...more"]
            row["genres"] = "~".join(genres) if genres else ""

            # Series
            series_loc = self.page.locator("h3.Text__italic a").first
            row["series"] = ""
            if series_loc.count() > 0:
                href = series_loc.get_attribute("href")
                if href: row["series"] = href.split('/')[-1]

            # Year
            pub_loc = self.page.locator("[data-testid='publicationInfo']").first
            row["year"] = ""
            if pub_loc.count() > 0:
                try:
                    # Logic matches previous bs4: split by comma, take last part
                    row["year"] = int(pub_loc.text_content().split(', ')[-1])
                except (ValueError, IndexError):
                    pass

            # Similar Books (Network)
            self.page.wait_for_timeout(3000)
            similar_ids = set()
            
            for data in self.captured_payloads:
                conn = data.get("data", {}).get("getSimilarBooks", {})
                edges = conn.get("edges", []) if isinstance(conn, dict) else []
                
                for edge in edges:
                    node = edge.get("node", {}) if isinstance(edge, dict) else {}
                    web_url = node.get("webUrl", "")
                    if web_url:
                        match = re.search(r'show/(\d+)', web_url)
                        if match:
                            try: similar_ids.add(int(match.group(1)))
                            except ValueError: pass
            
            if similar_ids:
                print(f"    Found {len(similar_ids)} similar books.")
            
            row["similar_books"] = "~".join(map(str, similar_ids))
            return row, list(similar_ids)

        except Exception as e:
            print(f"  [!!] Error scraping {book_id}: {e}")
            return None, []

def run_crawler(start_ids: List[int]):
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
    
    queue = deque(start_ids)
    visited = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("id"): visited.add(int(row["id"]))
        except Exception: pass

    print(f"Crawler started. Queue: {len(queue)} | Visited: {len(visited)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        scraper = GoodreadsScraper(page)

        while queue:
            current_id = queue.popleft()
            if current_id in visited: continue

            data, new_ids = scraper.scrape(current_id)
            visited.add(current_id)

            if data:
                with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writerow(data)
                
                added = 0
                for nid in new_ids:
                    if nid not in visited and nid not in queue:
                        queue.append(nid)
                        added += 1
                if added: print(f"    Added {added} new books.")
                    
        browser.close()
        print(f"\nFinished. Total: {len(visited)}")

if __name__ == "__main__":
    run_crawler([1])