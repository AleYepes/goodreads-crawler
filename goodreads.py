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
from playwright.sync_api import sync_playwright


def download_library(email, password):
    
    def preprocess_library():
        df = pd.read_csv(LIBRARY_PATH)
        df.columns = [col.lower().replace(' ','_') for col in df.columns]
        return df

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1280, "height": 800}, accept_downloads=True)
        page = context.new_page()
        
        # Log in
        page.goto("https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in")
        page.fill('input[type="email"]', email)
        page.fill('input[type="password"]', password)
        page.click('input[type="submit"]')

        # Prep export
        page.wait_for_selector(".homePrimaryColumn", timeout=60000)
        page.goto("https://www.goodreads.com/review/import", wait_until="domcontentloaded")
        export_button = page.locator(".js-LibraryExport").first
        export_button.click()
        
        prepped_export_list = page.locator(".fileList")
        for _ in range(90):
            if prepped_export_list.count() > 0 and prepped_export_list.locator("a").count() > 0:
                break
            page.wait_for_timeout(500)
        
        # Export library
        with page.expect_download() as download_info:
            prepped_export_list.locator("a").first.click()
        download_info.value.save_as(LIBRARY_PATH)

        browser.close()

    return preprocess_library()


def serialize_similar(sim_list):
    # Format: id:rating:count|id:rating:count
    if not sim_list:
        return ""
    return "|".join(f"{item['id']}:{item['r']}:{item['c']}" for item in sim_list)


def parse_similar(encoded_str):
    if pd.isna(encoded_str) or not isinstance(encoded_str, str) or not encoded_str:
        return []
    
    results = []
    for item in encoded_str.split("|"):
        parts = item.split(":")
        try:
            results.append({
                "id": int(parts[0]),
                "r": float(parts[1]),
                "c": int(parts[2])
            })
        except ValueError:
            continue
    return results


class GoodreadsScraper:
    def __init__(self):
        pass

    async def block_media(self, route):
        if route.request.resource_type in ["image", "media"]:
            await route.abort()
        else:
            await route.continue_()

    def safe_get_author(self, ld):
        auth = ld.get("author")
        if isinstance(auth, dict):
            return auth.get("name", "")
        if isinstance(auth, list) and auth:
            item = auth[0]
            if isinstance(item, dict):
                return item.get("name", "")
            return str(item)
        return ""
    
    def clean_description(self, text):
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

    async def scrape_book(self, context, book_id):

        async def handle_response(response):
            if "graphql" in response.url and response.request.method == "POST":
                try:
                    json_body = await response.json()
                    captured_payloads.append(json_body)
                except Exception:
                    pass

        captured_payloads = []
        page = await context.new_page()
        page.on("response", handle_response)
        await page.route("**/*", self.block_media)

        url = f"https://www.goodreads.com/book/show/{book_id}"
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)

            close_btn = page.locator(".Overlay__close button").first
            if await close_btn.is_visible():
                await close_btn.click()

            # Extract JSON-LD metadata
            script_locator = page.locator('script[type="application/ld+json"]').first
            try:
                await script_locator.wait_for(state="attached", timeout=5000)
                content = await script_locator.text_content()
                ld = json.loads(content)
            except Exception:
                await page.close()
                return None, []

            ld = next((item for item in ld if isinstance(item, dict) and item.get("@type") == "Book"), {})
            agg_rating = ld.get("aggregateRating", {})
            data = {
                "id": book_id,
                "title": html.unescape(ld.get("name", "")),
                "author": self.safe_get_author(ld),
                "avg_rating": agg_rating.get("ratingValue"),
                "total_reviews": agg_rating.get("reviewCount"),
                "pages": ld.get("numberOfPages"),
                "lang": ld.get("inLanguage"),
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
            data['description'] = self.clean_description(dom_data['description'])

            # Wait for similar books GraphQL response
            wait_attempts = 0
            while not any("getSimilarBooks" in str(p) for p in captured_payloads) and wait_attempts < WAIT_ATTEMPTS:
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

            page.remove_listener("response", handle_response)
            final_similar = list(similar_books_meta.values())
            data["similar_books"] = serialize_similar(final_similar)
            
            await page.close()
            return data, final_similar

        except Exception:
            await page.close()
            return None, []


def calculate_priority_score(avg_rating, rating_count):
    score = avg_rating - (avg_rating - 3) / math.sqrt(rating_count)
    return score


async def run_crawler_optimized(start_ids, OUTPUT_PATH):
    visited = set()
    queued_set = set()
    pq = []

    # Resume from existing output file
    if OUTPUT_PATH.exists():
        try:
            df_iter = pd.read_csv(OUTPUT_PATH, usecols=['id', 'similar_books'], chunksize=5000)
            for chunk in df_iter:
                visited.update(chunk['id'].dropna().astype(int).tolist())
                
                # Rebuild frontier from similar books
                for entry in chunk['similar_books'].dropna():
                    sim_list = parse_similar(entry)
                    for book_node in sim_list:
                        book_id = book_node.get('id')
                        if book_id and book_id not in visited and book_id not in queued_set:
                            score = calculate_priority_score(book_node.get('r', 0), book_node.get('c', 0))
                            heapq.heappush(pq, (-score, book_id))
                            queued_set.add(book_id)
        except Exception as e:
            print(f"    [Warning] Error reading resume file: {e}")

    # Add start IDs to queue
    for book_id in start_ids:
        if book_id not in visited and book_id not in queued_set:
            heapq.heappush(pq, (-10.0, book_id))
            queued_set.add(book_id)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False) # Running headed is required for GraphQL triggers to fire reliably
        scraper = GoodreadsScraper()
        
        context = await browser.new_context(
            viewport={"width": 1000, "height": 800},
            user_agent=USER_AGENT
        )
        
        batch_records = []
        pbar = tqdm(total=len(visited) + len(pq), initial=len(visited), unit="book", smoothing=0.01)
        
        active_tasks = set()
        
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
                is_new = not OUTPUT_PATH.exists()
                df.to_csv(OUTPUT_PATH, mode='a', header=is_new, index=False)
                batch_records = []

        # Final save
        if batch_records:
            df = pd.DataFrame(batch_records)
            is_new = not OUTPUT_PATH.exists()
            df.to_csv(OUTPUT_PATH, mode='a', header=is_new, index=False)
        
        await browser.close()
        pbar.close()
        print("Done.")


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LIBRARY_PATH = str(DATA_DIR / "goodreads_library_export.csv")
OUTPUT_PATH = str(DATA_DIR / "books.csv")

load_dotenv()
CONCURRENCY = 1
CONTEXT_LIFETIME = 50
WAIT_ATTEMPTS = 20
TIMEOUT_MS = 45000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    if args.force_download or not glob.glob(LIBRARY_PATH):
        email = os.getenv("GOODREADS_EMAIL")
        password = os.getenv("GOODREADS_PASSWORD")
        library_df = download_library(email, password)
    else:
        library_df = pd.read_csv(LIBRARY_PATH)

    start_ids = library_df['Book Id'].astype(int).tolist()
    asyncio.run(run_crawler_optimized(start_ids, OUTPUT_PATH))


if __name__ == "__main__":
    main()