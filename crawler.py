import argparse
import asyncio
import json
import os
import re
import glob
import heapq
import html
import csv
import random
import pandas as pd
import numpy as np
import traceback
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright


def download_library(email, password):

    def preprocess_library():
        df = pd.read_csv(LIBRARY_PATH)
        df.columns = [col.lower().replace(' ','_') for col in df.columns]
        df.to_csv(LIBRARY_PATH, index=False)
        return df

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1280, "height": 800}, user_agent=USER_AGENT, accept_downloads=True)
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
        for _ in range(120):
            if prepped_export_list.count() > 0 and prepped_export_list.locator("a").count() > 0:
                break
            page.wait_for_timeout(500)
        
        # Export library
        with page.expect_download() as download_info:
            prepped_export_list.locator("a").first.click()
        download_info.value.save_as(LIBRARY_PATH)
        browser.close()

    return preprocess_library()


def parse_similar_books_string(encoded_str):
    if not isinstance(encoded_str, str) or not encoded_str:
        return []
    
    similar_books = []
    for item in encoded_str.split("|"):
        book_id, count_adj_rating = item.split(":")
        similar_books.append((int(book_id), float(count_adj_rating)))
    return similar_books


def parse_and_score_similar_books(encoded_str):
    if not isinstance(encoded_str, str) or not encoded_str:
        return []
    
    similar_books = []
    for item in encoded_str.split("|"):
        try:
            parts = item.split(":")
            if len(parts) == 3:
                book_id = int(parts[0])
                avg = float(parts[1])
                count = int(parts[2])
                score = calculate_score(avg, count)
                similar_books.append((book_id, score))
        except ValueError:
            continue
            
    return similar_books


def calculate_score(avg_rating, rating_count):
    # return avg_rating - (avg_rating - 3) / np.sqrt(rating_count + 1)
    # return avg_rating - (avg_rating - 3) / np.log(rating_count + 10)
    return avg_rating - avg_rating / np.log(rating_count + 10)


def prep_crawl_heapq(library_df):
    crawl_queue = {int(book_id): 5.0 for book_id in library_df['book_id'].dropna()} # Max score of 5 to prioritize library seed ids
    scraped_ids = set()

    if OUTPUT_PATH.exists():
        scraped_df = pd.read_csv(OUTPUT_PATH, usecols=['book_id', 'similar_books'], on_bad_lines='skip')
        scraped_ids.update(scraped_df['book_id'].dropna().astype(int))

        for similar_books_str in scraped_df['similar_books'].dropna():
            scored_neighbors = parse_and_score_similar_books(similar_books_str)
            for book_id, score in scored_neighbors:
                if book_id not in scraped_ids:
                    if score >= crawl_queue.get(book_id, 0):
                        crawl_queue[book_id] = score

        for book_id in scraped_ids:
            crawl_queue.pop(book_id, None)

    crawl_queue = [(-rating, book_id) for book_id, rating in crawl_queue.items()]
    heapq.heapify(crawl_queue)

    return crawl_queue, scraped_ids, {book_id for _, book_id in crawl_queue}


async def fetch_book(page, book_id):

    async def handle_response(response):
        if not collecting:
            return
        
        if "graphql" in response.url and response.request.method == "POST":
            try:
                json_body = await response.json()
                captured_payloads.append(json_body)
            except Exception:
                pass

    async def extract_linked_data_basics(page, book_id):
        script_locator = page.locator('script[type="application/ld+json"]').first
        await script_locator.wait_for(state="attached", timeout=PAGE_TIMEOUT_MS)
        content = await script_locator.text_content()
        ld = json.loads(content)

        agg_rating = ld.get("aggregateRating", {})
        title = html.unescape(ld.get("name", ""))
        authors = "|".join(a["name"] for a in ld.get("author", []) if "name" in a)
        data = {
            "book_id": book_id,
            "title": title,
            "authors": authors,
            "avg_rating": agg_rating.get("ratingValue"),
            "review_count": agg_rating.get("reviewCount"),
            "num_pages": ld.get("numberOfPages"),
            "lang": ld.get("inLanguage"),
        }

        return data

    async def extract_dom_data(page, book_data):
        html_content = await page.content()
        soup = BeautifulSoup(html_content, "html.parser")

        # Stars distribution
        stars = {}
        for i in range(1, 6):   
            label = soup.find(attrs={"data-testid": f"labelTotal-{i}"})
            if label:
                text = label.get_text().strip().split()[0]
                text = text.replace(",", "")
                stars[f"{i}_star"] = int(text) if text.isdigit() else 0
            else:
                stars[f"{i}_star"] = 0
        book_data.update(stars)

        # Genres
        genre_nodes = soup.select(".BookPageMetadataSection__genreButton .Button__labelItem")
        genres = [node.get_text() for node in genre_nodes if node.get_text() != "...more"]
        book_data['genres'] = "|".join(genres)

        # Series id
        series_el = soup.select_one("h3.Text__italic a")
        if series_el and series_el.get('href'):
            book_data['series'] = series_el['href'].split('/')[-1]
        else:
            book_data['series'] = ""

        # Year
        pub_el = soup.find(attrs={"data-testid": "publicationInfo"})
        if pub_el:
            parts = pub_el.get_text().split(", ")
            book_data['year'] = parts[-1].strip() if parts else ""
        else:
            book_data['year'] = ""

        # Description
        desc_el = soup.select_one("[data-testid='description'] span.Formatted")
        if not desc_el:
            desc_el = soup.select_one(".DetailsLayoutRightParagraph__widthConstrained span.Formatted")
        if desc_el:
            for br in desc_el.find_all("br"):
                br.replace_with("\n")
            text = desc_el.get_text()
            text = re.sub(r'\n{3,}', '\n\n', text)
            book_data['description'] = text.strip()
        else:
            book_data['description'] = ""

        return book_data

    async def extract_similar_books_json(page, book_data, captured_payloads, collecting):
        wait_attempts = 0
        while not any("getSimilarBooks" in str(p) for p in captured_payloads) and wait_attempts < PAYLOAD_WAIT_ATTEMPTS:
            await page.wait_for_timeout(500)
            wait_attempts += 1
        collecting = False
        
        similar_books = []
        for payload in captured_payloads:
            data_block = payload.get("data", {}).get("getSimilarBooks", {})
            similar_books_raw = data_block.get("edges", [])
            
            for book_edge in similar_books_raw:
                book_node = book_edge.get("node", {})

                web_url = book_node.get("webUrl", "")
                match = re.search(r'show/(\d+)', web_url)
                similar_book_id = int(match.group(1))
                
                stats = book_node.get("work").get("stats", {})
                avg_rating = float(stats.get("averageRating"))
                rating_count = int(stats.get("ratingsCount"))

                similar_books.append(f"{similar_book_id}:{avg_rating}:{rating_count}")

        book_data["similar_books"] = "|".join(similar_books)
        return book_data

    captured_payloads = []
    collecting = True
    page.on("response", handle_response)
    try:
        url = f"https://www.goodreads.com/book/show/{book_id}"
        await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)

        modal_close_btn = page.locator(".Overlay__close button").first
        if await modal_close_btn.is_visible():
            await modal_close_btn.click()

        await page.evaluate(f"window.scrollBy(0, {random.randint(100,200)})")

        book_data = await extract_linked_data_basics(page, book_id)
        book_data = await extract_dom_data(page, book_data)
        book_data = await extract_similar_books_json(page, book_data, captured_payloads, collecting)

        return book_data

    except Exception as e:
        if "Target closed" not in str(e):
             tqdm.write(f"Task Failed for {book_id} -- {e}")
        return None
    finally:
        collecting = False
        page.remove_listener("response", handle_response)
        try:
            await page.goto("about:blank")
        except Exception:
            pass


async def run_crawler(crawl_queue, scraped_book_ids, queued_book_ids):

    async def block_media(route):
        if route.request.resource_type in ["image", "media", "font"]:
            await route.abort()
        else:
            await route.continue_()

    async def fetch_book_wrapper(page, book_id, page_pool):
        try:
            result = await fetch_book(page, book_id)
            return result
        finally:
            await page_pool.put(page)

    file_exists = OUTPUT_PATH.exists()
    field_names = [
        "book_id", "title", "authors", "avg_rating", "review_count", 
        "num_pages", "lang", "1_star", "2_star", "3_star", "4_star", 
        "5_star", "genres", "series", "year", "description", "similar_books"
    ]

    with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()

        pbar = tqdm(total=len(scraped_book_ids) + len(crawl_queue), initial=len(scraped_book_ids), unit='book')

        try:
            while crawl_queue:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=False) # Seems that running headed is required for GraphQL triggers to fire
                    context = await browser.new_context(user_agent=USER_AGENT)
                    await context.route("**/*", block_media)

                    page_pool = asyncio.Queue()
                    for _ in range(CONCURRENCY):
                        pg = await context.new_page()
                        page_pool.put_nowait(pg)

                    active_tasks = set()
                    window_processed_count = 0
                    while crawl_queue or active_tasks:
                        while (crawl_queue and not page_pool.empty() and window_processed_count < RESTART_THRESHOLD):
                            _, current_id = heapq.heappop(crawl_queue)
                            if current_id in scraped_book_ids:
                                continue

                            page = page_pool.get_nowait()
                            task = asyncio.create_task(fetch_book_wrapper(page, current_id, page_pool))
                            active_tasks.add(task)
                        if not active_tasks:
                            break

                        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            try:
                                window_processed_count += 1
                                book_data = task.result()
                                if book_data:
                                    writer.writerow(book_data)
                                    f.flush()
                                    scraped_book_ids.add(book_data['book_id'])
                                    pbar.update(1)
                                
                                    scored_neighbors = parse_and_score_similar_books(book_data.get('similar_books'))
                                    added_count = 0
                                    for similar_book_id, score in scored_neighbors:
                                        if similar_book_id not in scraped_book_ids and similar_book_id not in queued_book_ids:
                                            heapq.heappush(crawl_queue, (-score, similar_book_id))
                                            queued_book_ids.add(similar_book_id)
                                            added_count += 1
                                    pbar.total += added_count
                            except Exception as e:
                                tqdm.write(f"\nError post-processing task: {e}")
                                traceback.print_exc()
                            
                    if window_processed_count >= RESTART_THRESHOLD:
                        await asyncio.sleep(1)

        finally:
            pbar.close()
            for task in active_tasks:
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            await browser.close()


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LIBRARY_PATH = DATA_DIR / "goodreads_library_export.csv"
OUTPUT_PATH = DATA_DIR / "books.csv"

load_dotenv()
CONCURRENCY = 3
PAYLOAD_WAIT_ATTEMPTS = 20
PAGE_TIMEOUT_MS = 10000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
RESTART_THRESHOLD = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", action="store_true")
    args = parser.parse_args()

    if args.fd or not glob.glob(str(LIBRARY_PATH)):
        email = os.getenv("GOODREADS_EMAIL")
        password = os.getenv("GOODREADS_PASSWORD")
        library_df = download_library(email, password)
    else:
        library_df = pd.read_csv(LIBRARY_PATH)

    crawl_queue, scraped_book_ids, queued_book_ids = prep_crawl_heapq(library_df)
    try:
        asyncio.run(run_crawler(crawl_queue, scraped_book_ids, queued_book_ids))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()