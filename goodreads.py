import argparse
import json
import os
import re
import glob
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from playwright.sync_api import sync_playwright, Page, Response, TimeoutError as PlaywrightTimeoutError

load_dotenv()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


class GoodreadsExporter:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password

    def _login(self, page: Page):
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
        page.fill('input[type="email"]', self.email)
        page.fill('input[type="password"]', self.password)
        page.click('input[type="submit"]')
        
        try:
            page.wait_for_selector(".homePrimaryColumn", timeout=60000)
        except PlaywrightTimeoutError:
            raise Exception("Login failed - check credentials or handle 2FA manually")
    
    def download_library(self) -> pd.DataFrame:
        today_str = datetime.now().strftime("%d%m%y")
        filename = f"{today_str}_goodreads_library_export.csv"
        output_path = DATA_DIR / filename
        
        print("    Starting browser for export (Headed for Login)...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                accept_downloads=True
            )
            page = context.new_page()
            self._login(page)
            page.goto("https://www.goodreads.com/review/import", wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            
            export_button = page.locator(".js-LibraryExport").first
            if export_button.count() > 0:
                export_button.click()
                page.wait_for_timeout(3000)

            max_attempts = 90
            file_list = page.locator(".fileList")
            
            for _ in range(max_attempts):
                if file_list.count() > 0:
                    link = file_list.locator("a").first
                    if link.count() > 0:
                        break
                page.wait_for_timeout(1000)
            else:
                raise TimeoutError("Export generation did not complete in time")
            
            with page.expect_download() as download_info:
                link = file_list.locator("a").first
                link.click()
            
            download = download_info.value
            download.save_as(output_path)
            browser.close()
            
        print(f"    Downloaded new export: {filename}")
        return pd.read_csv(output_path)


class GoodreadsScraper:
    def __init__(self, page: Page):
        self.page = page
        self.captured_payloads = []
        self.interstitial_handled = False  # Flag to track if we've closed the modal
        self.page.on("response", self._handle_network_response)
    
    def _handle_network_response(self, response: Response):
        if self._is_similar_books_request(response):
            try:
                self.captured_payloads.append(response.json())
            except Exception:
                pass
    
    def _is_similar_books_request(self, response: Response) -> bool:
        if "graphql" in response.url and response.request.method == "POST":
            try:
                req_json = response.request.post_data_json
                return isinstance(req_json, dict) and req_json.get("operationName") == "getSimilarBooks"
            except Exception:
                return False
        return False
    
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

    def _handle_interstitial(self):
        """Checks for and closes the 'Sign up' modal only if not yet handled."""
        if self.interstitial_handled:
            return

        # Selector based on the HTML provided
        close_btn = self.page.locator(".Overlay__close button").first
        
        # We use a short timeout because we don't want to wait if it's not there
        if close_btn.count() > 0 and close_btn.is_visible():
            try:
                close_btn.click()
                self.interstitial_handled = True
                # tqdm.write("    [Info] Interstitial modal closed permanently.")
            except Exception:
                pass

    def scrape(self, book_id: int) -> Tuple[Optional[dict], List[int]]:
        self.captured_payloads.clear()
        url = f"https://www.goodreads.com/book/show/{book_id}"
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=45000)
            self._handle_interstitial()

            # Basic book LD-JSON data
            script = self.page.locator('script[type="application/ld+json"]').first
            if script.count() == 0:
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
            if not isinstance(agg_rating, dict):
                agg_rating = {}
            
            data = {
                "id": book_id,
                "title": ld.get("name"),
                "author": self._safe_get_author(ld),
                "avg_rating": agg_rating.get("ratingValue"),
                "total_reviews": agg_rating.get("reviewCount"),
                "pages": ld.get("numberOfPages"),
                "lang": ld.get("inLanguage"),
            }
            
            for star in range(5, 0, -1):
                el = self.page.locator(f"[data-testid='labelTotal-{star}']").first
                count = 0
                if el.count() > 0:
                    text = el.text_content()
                    clean_text = re.sub(r'[^\d]', '', text.split('(')[0])
                    if clean_text:
                        count = int(clean_text)
                data[f"{star}_star"] = count
            
            genre_locs = self.page.locator(".BookPageMetadataSection__genreButton .Button__labelItem")
            genres = [t for t in genre_locs.all_text_contents() if t != "...more"]
            data["genres"] = "|".join(genres) if genres else ""
            
            series_loc = self.page.locator("h3.Text__italic a").first
            data["series"] = ""
            if series_loc.count() > 0:
                href = series_loc.get_attribute("href")
                if href:
                    data["series"] = href.split('/')[-1]
            
            pub_loc = self.page.locator("[data-testid='publicationInfo']").first
            data["year"] = ""
            if pub_loc.count() > 0:
                try:
                    data["year"] = int(pub_loc.text_content().split(', ')[-1])
                except (ValueError, IndexError):
                    pass
            
            desc_loc = self.page.locator("[data-testid='description'] span.Formatted").first
            if desc_loc.count() == 0:
                desc_loc = self.page.locator(".DetailsLayoutRightParagraph__widthConstrained span.Formatted").first
            data["description"] = desc_loc.inner_html() if desc_loc.count() > 0 else ""
            
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
            
            data["similar_books"] = "|".join(map(str, similar_ids))
            return data, list(similar_ids)
        
        except Exception as e:
            tqdm.write(f"Error scraping {book_id}: {e}")
            return None, []


def get_latest_export_file() -> Optional[Path]:
    pattern = str(DATA_DIR / "*_goodreads_library_export.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    
    def extract_date(f_path):
        name = Path(f_path).name
        date_part = name.split('_')[0]
        try:
            return datetime.strptime(date_part, "%d%m%y")
        except ValueError:
            return datetime.min

    latest_file = max(files, key=extract_date)
    return Path(latest_file)


def run_crawler(start_ids: List[int], output_path: Path):
    # 1. Setup Tracking Sets
    visited = set()
    queued_set = set(start_ids)
    queue = deque(start_ids)
    
    # 2. Check for existing data to resume
    if output_path.exists():
        print(f"Resuming from {output_path}...")
        try:
            # Optimize: Read chunks or just necessary columns
            # float_precision='round_trip' helps ensure IDs don't get garbled if read as floats
            existing_df = pd.read_csv(output_path, usecols=['id', 'similar_books'])
            
            # Similar ID extraction
            visited = set(existing_df['id'].dropna().astype(int))
            potential_pool = set(start_ids)
            
            if 'similar_books' in existing_df.columns:
                sim_ids = (
                    existing_df['similar_books']
                    .dropna()
                    .astype(str)
                    .str.split('|')
                    .explode()
                )
                valid_sims = sim_ids[pd.to_numeric(sim_ids, errors='coerce').notnull()].astype(int)
                potential_pool.update(valid_sims.tolist())

            # Rebuild Queue
            queue_items = potential_pool - visited
            queue = deque(list(queue_items))
            queued_set = set(queue) | visited
            
        except Exception as e:
            print(f"    [Warning] Error reading resume file: {e}")
            print("    Starting fresh (or continuing with provided IDs only).")

    # 3. Setup Browser and Scraper
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        scraper = GoodreadsScraper(page)
        
        batch_records = []
        BATCH_SIZE = 50
        books_scraped_session = 0  # Local counter for restarts
        
        DATA_DIR.mkdir(exist_ok=True)

        try:
            # Update total to reflect what we actually know
            with tqdm(total=len(visited) + len(queue), initial=len(visited), unit="book") as pbar:
                while queue:
                    # 4. Resource Management: Restart context every 200 books THIS SESSION
                    if books_scraped_session > 0 and books_scraped_session % 200 == 0:
                        tqdm.write("    [Maintenance] Restarting browser context to clear memory...")
                        page.close()
                        context.close()
                        context = browser.new_context(viewport={"width": 1280, "height": 800})
                        page = context.new_page()
                        scraper = GoodreadsScraper(page) # Re-attach scraper to new page

                    current_id = queue.popleft()
                    
                    # Double check (redundant but safe)
                    if current_id in visited:
                        continue
                    
                    pbar.set_description(f"Scraping {current_id}")
                    
                    # Scrape
                    data, similar_ids = scraper.scrape(current_id)
                    visited.add(current_id)
                    books_scraped_session += 1
                    pbar.update(1)
                    
                    if data:
                        batch_records.append(data)
                        
                        # Queue Update
                        for sid in similar_ids:
                            if sid not in queued_set:
                                queue.append(sid)
                                queued_set.add(sid)
                                pbar.total += 1
                        
                        # 5. Efficient Saving
                        if len(batch_records) >= BATCH_SIZE:
                            df = pd.DataFrame(batch_records)
                            
                            # Check if file exists right now to decide on header
                            file_is_new = not output_path.exists()
                            df.to_csv(output_path, mode='a', header=file_is_new, index=False)
                            
                            batch_records = [] 
                            
        except KeyboardInterrupt:
            tqdm.write("\nStopping... Saving remaining records.")
        except Exception as e:
            tqdm.write(f"\n[Critical Error] {e}")
        finally:
            # Save leftovers
            if batch_records:
                df = pd.DataFrame(batch_records)
                file_is_new = not output_path.exists()
                df.to_csv(output_path, mode='a', header=file_is_new, index=False)
                tqdm.write(f"Saved {len(batch_records)} remaining records.")
            
            browser.close()
            print(f"✓ Finished. Total visited: {len(visited)}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Goodreads book data")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download of a new library export even if one exists"
    )
    parser.add_argument(
        "--start-ids",
        nargs="+",
        type=int,
        help="Manually specify starting book IDs"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of books to scrape"
    )
    
    args = parser.parse_args()
    
    today_str = datetime.now().strftime("%d%m%y")
    output_path = DATA_DIR / f"{today_str}_books_data.csv"

    start_ids = []
    
    if args.start_ids:
        start_ids = args.start_ids
    else:
        latest_export = get_latest_export_file()
        should_download = args.force_download or (latest_export is None)
        
        if should_download:
            email = os.getenv("GOODREADS_EMAIL")
            password = os.getenv("GOODREADS_PASSWORD")
            if not email or not password:
                print("ERROR: Set credentials in .env file.")
                return
            
            print("Initiating library download...")
            exporter = GoodreadsExporter(email, password)
            library_df = exporter.download_library()
        else:
            filename = latest_export.name
            date_part = filename.split('_')[0]
            formatted_date = f"{date_part[:2]}/{date_part[2:4]}/{date_part[4:]}"
            print(f"Using existing export file from: {formatted_date} ({filename})")
            library_df = pd.read_csv(latest_export)
        
        start_ids = library_df['Book Id'].astype(int).tolist()

    if args.limit:
        start_ids = start_ids[:args.limit]

    run_crawler(start_ids, output_path)


if __name__ == "__main__":
    main()