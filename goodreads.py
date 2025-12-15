import argparse
import asyncio
import json
import os
import re
import glob
import math
import heapq
import html
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from playwright.async_api import async_playwright, Page, Response, BrowserContext, Route

load_dotenv()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Running HEADED is required for Goodreads GraphQL triggers to fire reliably
CONCURRENCY = 4
CONTEXT_LIFETIME = 50
TIMEOUT_MS = 45000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Serialization Helpers

def serialize_similar(sim_list: List[Dict[str, Any]]) -> str:
    """
    Compacts similar books into a pipe-delimited string.
    Format: ID:RATING:COUNT|ID:RATING:COUNT
    Example: 12345:4.5:100|67890:3.8:500
    """
    if not sim_list:
        return ""
    return "|".join(f"{item['id']}:{item['r']}:{item['c']}" for item in sim_list)

def parse_similar(encoded_str: Any) -> List[Dict[str, Any]]:
    """
    Parses the compact string back into a list of dictionaries.
    Robust against NaNs or malformed strings.
    """
    if pd.isna(encoded_str) or not isinstance(encoded_str, str) or not encoded_str:
        return []
    
    results = []
    for item in encoded_str.split("|"):
        parts = item.split(":")
        if len(parts) == 3:
            try:
                results.append({
                    "id": int(parts[0]),
                    "r": float(parts[1]),
                    "c": int(parts[2])
                })
            except ValueError:
                continue
    return results


class GoodreadsExporter:
    """Handles logging in and downloading the initial library export."""
    
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password

    def download_library(self) -> pd.DataFrame:
        from playwright.sync_api import sync_playwright as sync_p
        
        today_str = datetime.now().strftime("%d%m%y")
        filename = f"{today_str}_goodreads_library_export.csv"
        output_path = DATA_DIR / filename
        
        print("    [Exporter] Starting browser for export...")
        with sync_p() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(viewport={"width": 1280, "height": 800}, accept_downloads=True)
            page = context.new_page()
            
            # Login
            page.goto("https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in")
            page.fill('input[type="email"]', self.email)
            page.fill('input[type="password"]', self.password)
            page.click('input[type="submit"]')
            try:
                page.wait_for_selector(".homePrimaryColumn", timeout=60000)
            except Exception:
                print("    [Error] Login failed or 2FA required. Please handle in browser.")
            
            # Navigate to export page
            page.goto("https://www.goodreads.com/review/import", wait_until="domcontentloaded")
            export_button = page.locator(".js-LibraryExport").first
            if export_button.count() > 0:
                export_button.click()
            
            print("    [Exporter] Waiting for export generation...")
            file_list = page.locator(".fileList")
            for _ in range(90):
                if file_list.count() > 0 and file_list.locator("a").count() > 0:
                    break
                page.wait_for_timeout(1000)
            
            with page.expect_download() as download_info:
                file_list.locator("a").first.click()
            
            download_info.value.save_as(output_path)
            browser.close()
            
        print(f"    [Exporter] Downloaded: {filename}")
        return pd.read_csv(output_path)


class GoodreadsScraper:
    def __init__(self):
        pass

    async def _block_media(self, route: Route):
        if route.request.resource_type in ["image", "media"]:
            await route.abort()
        else:
            await route.continue_()

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
    
    def _clean_description(self, text: Optional[str]) -> str:
        if not text: return ""
        text = html.unescape(text)
        
        # Replace breaks and paragraphs with newlines
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
        
        # Remove remaining tags (<i>, <b>, <p>, etc.)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    async def scrape_book(self, context: BrowserContext, book_id: int) -> Tuple[Optional[dict], List[Dict]]:
        page = await context.new_page()
        captured_payloads = []

        async def handle_response(response: Response):
            if "graphql" in response.url and response.request.method == "POST":
                try:
                    json_body = await response.json()
                    captured_payloads.append(json_body)
                except Exception:
                    pass

        page.on("response", handle_response)
        await page.route("**/*", self._block_media)

        url = f"https://www.goodreads.com/book/show/{book_id}"
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            
            # Close sign-up popup if present
            try:
                close_btn = page.locator(".Overlay__close button").first
                if await close_btn.is_visible():
                    await close_btn.click()
            except Exception:
                pass

            # Extract JSON-LD metadata
            script_locator = page.locator('script[type="application/ld+json"]').first
            try:
                await script_locator.wait_for(state="attached", timeout=5000)
                content = await script_locator.text_content()
                ld = json.loads(content)
            except Exception:
                await page.close()
                return None, []

            if isinstance(ld, list):
                ld = next((item for item in ld if isinstance(item, dict) and item.get("@type") == "Book"), {})
            
            agg_rating = ld.get("aggregateRating", {})
            data = {
                "id": book_id,
                "title": ld.get("name"),
                "author": self._safe_get_author(ld),
                "avg_rating": agg_rating.get("ratingValue"),
                "total_reviews": agg_rating.get("reviewCount"),
                "pages": ld.get("numberOfPages"),
                "lang": ld.get("inLanguage"),
                "scraped_at": datetime.now().isoformat()
            }

            # Extract DOM data via JavaScript evaluation
            dom_data = await page.evaluate("""() => {
                const stars = {};
                [5,4,3,2,1].forEach(i => {
                    const el = document.querySelector(`[data-testid='labelTotal-${i}']`);
                    if (el) {
                        const match = el.innerText.match(/^([\d,]+)/);
                        const numStr = match ? match[1].replace(/,/g, '') : '0';
                        stars[`${i}_star`] = parseInt(numStr) || 0;
                    } else {
                        stars[`${i}_star`] = 0;
                    }
                });

                const genreNodes = document.querySelectorAll(".BookPageMetadataSection__genreButton .Button__labelItem");
                const genres = Array.from(genreNodes)
                    .map(el => el.innerText)
                    .filter(t => t !== "...more")
                    .join("|");

                const seriesEl = document.querySelector("h3.Text__italic a");
                const series = seriesEl && seriesEl.href ? seriesEl.href.split('/').pop() : "";

                const pubEl = document.querySelector("[data-testid='publicationInfo']");
                let year = "";
                if (pubEl) {
                    const parts = pubEl.innerText.split(', ');
                    year = parts[parts.length - 1];
                }

                const descEl = document.querySelector("[data-testid='description'] span.Formatted") || 
                               document.querySelector(".DetailsLayoutRightParagraph__widthConstrained span.Formatted");

                return { 
                    stars, 
                    genres, 
                    series, 
                    year, 
                    description: descEl ? descEl.innerHTML : "" 
                };
            }""")
            
            data.update(dom_data['stars'])
            data['genres'] = dom_data['genres']
            data['series'] = dom_data['series']
            data['year'] = dom_data['year']
            data['description'] = self._clean_description(dom_data['description'])

            # Wait for similar books GraphQL response
            wait_attempts = 0
            while not any("getSimilarBooks" in str(p) for p in captured_payloads) and wait_attempts < 5:
                await page.wait_for_timeout(500)
                wait_attempts += 1
            
            similar_books_meta = {}
            for payload in captured_payloads:
                data_block = payload.get("data", {})
                if not data_block:
                    continue
                conn = data_block.get("getSimilarBooks", {})
                edges = conn.get("edges", []) if isinstance(conn, dict) else []
                
                for edge in edges:
                    node = edge.get("node", {})
                    web_url = node.get("webUrl", "")
                    
                    sim_id = None
                    if web_url:
                        match = re.search(r'show/(\d+)', web_url)
                        if match:
                            sim_id = int(match.group(1))
                    
                    if sim_id:
                        work = node.get("work", {})
                        stats = work.get("stats", {}) if work else {}
                        try:
                            rating = float(stats.get("averageRating") or 0)
                            count = int(stats.get("ratingsCount") or 0)
                        except (ValueError, TypeError):
                            rating, count = 0.0, 0
                        
                        similar_books_meta[sim_id] = {"id": sim_id, "r": rating, "c": count}

            final_similar = list(similar_books_meta.values())
            
            # Changed from JSON dumps to Custom Serialize
            data["similar_books"] = serialize_similar(final_similar)
            
            await page.close()
            return data, final_similar

        except Exception:
            await page.close()
            return None, []


def get_latest_export_file() -> Optional[Path]:
    pattern = str(DATA_DIR / "*_goodreads_library_export.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return Path(max(files, key=os.path.getctime))


def calculate_priority_score(avg_rating: float, rating_count: int) -> float:
    """Calculate priority score for crawl queue."""
    if rating_count < 1:
        return -100.0
    log_val = math.log1p(rating_count)
    score = avg_rating - (avg_rating - 3) / log_val
    return score


async def run_crawler_optimized(start_ids: List[int], output_path: Path):
    visited = set()
    queued_set = set()
    pq = []

    # Resume from existing output file
    if output_path.exists():
        print(f"Resuming from {output_path}...")
        try:
            df_iter = pd.read_csv(output_path, usecols=['id', 'similar_books'], chunksize=5000)
            for chunk in df_iter:
                visited.update(chunk['id'].dropna().astype(int).tolist())
                
                # Rebuild frontier from similar books
                for entry in chunk['similar_books'].dropna():
                    # Changed from JSON loads to Custom Parse
                    sim_list = parse_similar(entry)
                    
                    for book_node in sim_list:
                        bid = book_node.get('id')
                        if bid and bid not in visited and bid not in queued_set:
                            score = calculate_priority_score(book_node.get('r', 0), book_node.get('c', 0))
                            heapq.heappush(pq, (-score, bid))
                            queued_set.add(bid)
        except Exception as e:
            print(f"    [Warning] Error reading resume file: {e}")

    # Add start IDs to queue
    for bid in start_ids:
        if bid not in visited and bid not in queued_set:
            heapq.heappush(pq, (-10.0, bid))
            queued_set.add(bid)

    print(f"    Queue Size: {len(pq)} | Visited: {len(visited)}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        scraper = GoodreadsScraper()
        
        context = await browser.new_context(
            viewport={"width": 1000, "height": 800},
            user_agent=USER_AGENT
        )
        
        batch_records = []
        pbar = tqdm(total=len(visited) + len(pq), initial=len(visited), unit="book", smoothing=0.01)
        
        active_tasks = set()
        requests_count = 0
        
        while pq or active_tasks:
            # Fill concurrent task slots
            while len(active_tasks) < CONCURRENCY and pq:
                neg_score, current_id = heapq.heappop(pq)
                if current_id in queued_set:
                    queued_set.remove(current_id)
                if current_id in visited:
                    continue
                
                task = asyncio.create_task(scraper.scrape_book(context, current_id))
                task.set_name(str(current_id))
                active_tasks.add(task)
                
                pbar.set_description(f"Q: {len(pq)} | Act: {len(active_tasks)}")

            if not active_tasks:
                break

            # Wait for any task to complete
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    book_id = int(task.get_name())
                    visited.add(book_id)
                    pbar.update(1)
                    requests_count += 1
                    
                    data, similar_meta = task.result()
                    
                    if data:
                        batch_records.append(data)
                        
                        # Add similar books to queue
                        added_count = 0
                        for sim in similar_meta:
                            sid = sim['id']
                            if sid not in visited and sid not in queued_set:
                                score = calculate_priority_score(sim['r'], sim['c'])
                                heapq.heappush(pq, (-score, sid))
                                queued_set.add(sid)
                                added_count += 1
                        pbar.total += added_count
                        
                except Exception as e:
                    tqdm.write(f"Task Failed: {e}")

            # Save batch periodically
            if len(batch_records) >= 20:
                df = pd.DataFrame(batch_records)
                is_new = not output_path.exists()
                df.to_csv(output_path, mode='a', header=is_new, index=False)
                batch_records = []

            # Rotate browser context for memory management
            if requests_count >= CONTEXT_LIFETIME:
                await context.close()
                context = await browser.new_context(
                    viewport={"width": 1000, "height": 800},
                    user_agent=USER_AGENT
                )
                requests_count = 0

        # Final save
        if batch_records:
            df = pd.DataFrame(batch_records)
            is_new = not output_path.exists()
            df.to_csv(output_path, mode='a', header=is_new, index=False)
        
        await browser.close()
        pbar.close()
        print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--start-ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    
    output_path = DATA_DIR / "goodreads_books_data.csv"
    start_ids = []

    if args.start_ids:
        start_ids = args.start_ids
    else:
        latest_export = get_latest_export_file()
        if args.force_download or not latest_export:
            email = os.getenv("GOODREADS_EMAIL")
            password = os.getenv("GOODREADS_PASSWORD")
            if not email or not password:
                print("Error: GOODREADS_EMAIL and GOODREADS_PASSWORD environment variables required.")
                return
            exporter = GoodreadsExporter(email, password)
            library_df = exporter.download_library()
        else:
            print(f"Using export: {latest_export.name}")
            library_df = pd.read_csv(latest_export)
        
        start_ids = library_df['Book Id'].astype(int).tolist()

    if args.limit:
        start_ids = start_ids[:args.limit]

    asyncio.run(run_crawler_optimized(start_ids, output_path))


if __name__ == "__main__":
    main()