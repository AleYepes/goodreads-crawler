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
        page.wait_for_selector(".js-LibraryExport", timeout=10000)

        export_button = page.locator(".js-LibraryExport").first
        while True:
            if export_button.is_visible():
                export_button.click()
                break
            page.wait_for_timeout(500)
        
        prepped_export_list = page.locator(".fileList")
        for _ in range(240):
            if prepped_export_list.count() > 0 and prepped_export_list.locator("a").count() > 0:
                break
            page.wait_for_timeout(500)
        
        # Export library
        with page.expect_download() as download_info:
            list = prepped_export_list.locator("a").first
            list.click()
        download_info.value.save_as(LIBRARY_PATH)
        browser.close()

    return preprocess_library()


def parse_and_score_similar_books(encoded_str, scoring_func):
    if not isinstance(encoded_str, str) or not encoded_str:
        return []
    
    similar_books = []
    for item in encoded_str.split("|"):
        try:
            parts = item.split(":")
            book_id, avg, count = int(parts[0]), float(parts[1]), int(parts[2])
            similar_books.append((book_id, scoring_func(avg, count)))
        except ValueError:
            continue
    return similar_books


def filter_save_file():
    if OUTPUT_PATH.exists():
        try:
            book_df = pd.read_csv(OUTPUT_PATH, on_bad_lines='skip')
            star_cols = [col for col in book_df.columns if col.endswith('star')]
            int_cols = ['book_id', 'review_count', 'num_pages', 'author_followers', 
                        'want_to_read', 'author_num_books', 'currently_reading'] + star_cols
            for col in int_cols:
                book_df[col] = pd.to_numeric(book_df[col], errors='coerce').astype('Int64')
            book_df['year'] = pd.to_numeric(book_df['year'], errors='coerce').astype('Int16')

            # book_df.dropna(subset=['book_id']) 
            book_df = book_df[~book_df['similar_books'].isna()]

            temp_path = OUTPUT_PATH.with_suffix('.tmp')
            book_df.to_csv(temp_path, index=False)
            temp_path.replace(OUTPUT_PATH)
        except Exception as e:
            print(f"Error cleaning file: {e}")
            traceback.print_exc()


def prep_crawl_heapq(library_df, scoring_func):
    crawl_queue = {int(book_id): 9e7 for book_id in library_df['book_id'].dropna()} # Prioritize library seed ids
    scraped_ids = set()

    if OUTPUT_PATH.exists():
        scraped_df = pd.read_csv(OUTPUT_PATH, usecols=['book_id', 'similar_books'], on_bad_lines='skip')
        scraped_ids.update(scraped_df['book_id'].dropna().astype(int))

        for similar_books_str in scraped_df['similar_books'].dropna():
            for book_id, score in parse_and_score_similar_books(similar_books_str, scoring_func):
                if book_id not in scraped_ids:
                    crawl_queue[book_id] = max(score, crawl_queue.get(book_id, 0))

        for book_id in scraped_ids:
            crawl_queue.pop(book_id, None)

    crawl_queue = [(-rating, book_id) for book_id, rating in crawl_queue.items()]
    heapq.heapify(crawl_queue)

    return crawl_queue, scraped_ids, {book_id for _, book_id in crawl_queue}


async def fetch_book(page, book_id):

    async def handle_response(response):       
        if collecting and "graphql" in response.url and response.request.method == "POST":
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
        return {
            "book_id": book_id,
            "title": title,
            "authors": authors,
            "avg_rating": agg_rating.get("ratingValue"),
            "review_count": agg_rating.get("reviewCount"),
            "num_pages": ld.get("numberOfPages"),
            "lang": ld.get("inLanguage"),
        }


    async def extract_dom_data(page, book_data):
        html_content = await page.content()
        soup = BeautifulSoup(html_content, "html.parser")

        # Stars distribution
        for i in range(1, 6):   
            label = soup.find(attrs={"data-testid": f"labelTotal-{i}"})
            text = label.get_text().strip().split()[0].replace(",", "") if label else "0"
            book_data[f"{i}_star"] = int(text) if text.isdigit() else 0

        # Genres
        try:
            if await page.query_selector('button[aria-label="Show all items in the list"]'):
                await page.click('button[aria-label="Show all items in the list"]')
                await page.wait_for_timeout(100) 
                html_content = await page.content()
                soup = BeautifulSoup(html_content, "html.parser")
        except Exception:
            pass

        genre_nodes = soup.select(".BookPageMetadataSection__genreButton .Button__labelItem")
        genres = [node.get_text() for node in genre_nodes if node.get_text() != "...more"]
        book_data['genres'] = "|".join(genres)

        # Series id
        series_el = soup.select_one("h3.Text__italic a")
        book_data['series'] = series_el['href'].split('/')[-1] if series_el and series_el.get('href') else ""

        # Year
        pub_el = soup.find(attrs={"data-testid": "publicationInfo"})
        book_data['year'] = pub_el.get_text().split(", ")[-1].strip() if pub_el else ""

        # Description
        desc_el = soup.select_one("[data-testid='description'] span.Formatted") or \
                  soup.select_one(".DetailsLayoutRightParagraph__widthConstrained span.Formatted")
        if desc_el:
            for br in desc_el.find_all("br"):
                br.replace_with("\n")
            book_data['description'] = re.sub(r'\n{3,}', '\n\n', desc_el.get_text()).strip()
        else:
            book_data['description'] = ""

        # Currently reading 
        reading_el = soup.find(attrs={"data-testid": "currentlyReadingSignal"})
        if reading_el:
            text = reading_el.get_text()
            match = re.search(r'(\d+)', text.replace(",", ""))
            book_data['currently_reading'] = int(match.group(1)) if match else 0
        else:
            book_data['currently_reading'] = 0

        # Want to read
        wtr_el = soup.find(attrs={"data-testid": "toReadSignal"})
        if wtr_el:
            text = wtr_el.get_text()
            match = re.search(r'(\d+)', text.replace(",", ""))
            book_data['want_to_read'] = int(match.group(1)) if match else 0
        else:
            book_data['want_to_read'] = 0

        # Author name
        author_name_el = soup.find(attrs={"data-testid": "name"})
        book_data['primary_author'] = author_name_el.get_text().strip() if author_name_el else ""

        # Author stats
        author_stats_el = soup.select_one(".FeaturedPerson__infoPrimary .Text__subdued")
        
        book_data['author_num_books'] = 0
        book_data['author_followers'] = 0
        if author_stats_el:
            stats_text = author_stats_el.get_text(separator=" ", strip=True)
            
            books_match = re.search(r'([\d,]+)\s*books', stats_text)
            if books_match:
                book_data['author_num_books'] = int(books_match.group(1).replace(",", ""))

            # Author follower count
            followers_match = re.search(r'([\d,kKmM\.]+)\s*followers', stats_text)
            if followers_match:
                val = followers_match.group(1).lower().replace(",", "")
                if 'k' in val:
                    val = float(val.replace('k', '')) * 1e3
                elif 'm' in val:
                    val = float(val.replace('m', '')) * 1e6
                book_data['author_followers'] = int(val)

        return book_data

    async def extract_similar_books_json(page, book_data, captured_payloads, collecting):
        wait_attempts = 0
        while not any("getSimilarBooks" in str(p) for p in captured_payloads) and wait_attempts < PAYLOAD_WAIT_ATTEMPTS:
            await page.wait_for_timeout(500)
            wait_attempts += 1
        collecting = False
        
        similar_books = []
        for payload in captured_payloads:
            for book_edge in payload.get("data", {}).get("getSimilarBooks", {}).get("edges", []):
                book_node = book_edge.get("node", {})
                match = re.search(r'show/(\d+)', book_node.get("webUrl", ""))
                if match:
                    stats = book_node.get("work", {}).get("stats", {})
                    similar_books.append(f"{match.group(1)}:{stats.get('averageRating')}:{stats.get('ratingsCount')}")

        book_data["similar_books"] = "|".join(similar_books)
        return book_data
    
    async def close_modal(page):
        try:
            close_btn = page.locator(".Overlay__close button, [aria-label='Close']").first
            if await close_btn.is_visible():
                await close_btn.click(timeout=2000)
        except Exception:
            pass

    await close_modal(page)
    captured_payloads = []
    collecting = True
    page.on("response", handle_response)
    try:
        url = f"https://www.goodreads.com/book/show/{book_id}"
        await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
        await page.evaluate(f"window.scrollBy(0, {random.randint(100,200)})")
        await close_modal(page)

        book_data = await extract_linked_data_basics(page, book_id)
        book_data = await extract_dom_data(page, book_data)
        book_data = await extract_similar_books_json(page, book_data, captured_payloads, collecting)

        return book_data

    except Exception as e:
        if "Target closed" not in str(e):
             tqdm.write(f"Failed {book_id} -- {e}")
        return None
    finally:
        collecting = False
        page.remove_listener("response", handle_response)
        try:
            await page.goto("about:blank")
        except Exception:
            pass


async def run_crawler(library_df):

    async def block_media(route):
        if route.request.resource_type in ["image", "media", "font"]:
            await route.abort()
        else:
            await route.continue_()

    async def fetch_wrapper(page_pool, page, book_id):
        try:
            return await fetch_book(page, book_id)
        finally:
            page_pool.put_nowait(page)

    field_names = [
        "book_id", "title", "authors", "avg_rating", "review_count", 
        "num_pages", "lang", "1_star", "2_star", "3_star", "4_star", 
        "5_star", "genres", "series", "year", "description", "similar_books",
        "primary_author", "author_followers", "want_to_read", 
        "author_num_books", "currently_reading"
    ]

    cycle = 0
    while True:
        scoring_func = SCORING_FUNCTIONS[cycle % len(SCORING_FUNCTIONS)]
        filter_save_file()
        crawl_queue, scraped_ids, queued_ids = prep_crawl_heapq(library_df, scoring_func)

        if not crawl_queue:
            break

        file_exists = OUTPUT_PATH.exists()
        with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            if not file_exists:
                writer.writeheader()

            pbar = tqdm(total=len(scraped_ids) + len(crawl_queue), initial=len(scraped_ids), unit='book')

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False) # Seems that running headed is required for GraphQL triggers to fire
                context = await browser.new_context(user_agent=USER_AGENT)
                await context.route("**/*", block_media)

                page_pool = asyncio.Queue()
                for _ in range(CONCURRENCY):
                    page_pool.put_nowait(await context.new_page())

                active_tasks = set()
                processed = 0
                try:
                    while (crawl_queue or active_tasks) and processed < RESTART_THRESHOLD:
                        while crawl_queue and not page_pool.empty() and processed < RESTART_THRESHOLD:
                            _, book_id = heapq.heappop(crawl_queue)
                            if book_id in scraped_ids:
                                continue

                            page = page_pool.get_nowait()
                            task = asyncio.create_task(fetch_wrapper(page_pool, page, book_id))
                            active_tasks.add(task)

                        if not active_tasks:
                            break

                        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            try:
                                processed += 1
                                book_data = task.result()
                                if book_data:
                                    writer.writerow(book_data)
                                    f.flush()
                                    scraped_ids.add(book_data['book_id'])
                                    pbar.update(1)
                                
                                    added = 0
                                    for similar_id, score in parse_and_score_similar_books(book_data.get('similar_books', ''), scoring_func):
                                        if similar_id not in scraped_ids and similar_id not in queued_ids:
                                            heapq.heappush(crawl_queue, (-score, similar_id))
                                            queued_ids.add(similar_id)
                                            added += 1
                                    pbar.total += added
                            except Exception as e:
                                tqdm.write(f"\nError post-processing task: {e}")
                                traceback.print_exc()

                finally:
                    pbar.close()
                    for task in active_tasks:
                        task.cancel()
                    if active_tasks:
                        await asyncio.gather(*active_tasks, return_exceptions=True)
                    await browser.close()
                    await asyncio.sleep(1)

        cycle += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fd", action="store_true")
    args = parser.parse_args()

    if args.fd or not glob.glob(str(LIBRARY_PATH)):
        email = os.getenv("GOODREADS_EMAIL")
        password = os.getenv("GOODREADS_PASSWORD")
        library_df = download_library(email, password)
    else:
        library_df = pd.read_csv(LIBRARY_PATH)

    try:
        asyncio.run(run_crawler(library_df))
    except KeyboardInterrupt:
        pass


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LIBRARY_PATH = DATA_DIR / "goodreads_library_export.csv"
OUTPUT_PATH = DATA_DIR / "books.csv"

load_dotenv()
CONCURRENCY = 3
PAYLOAD_WAIT_ATTEMPTS = 20
PAGE_TIMEOUT_MS = 20000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
RESTART_THRESHOLD = 200

SCORING_FUNCTIONS = [
    # lambda avg_rating, rating_count: avg_rating - avg_rating / np.sqrt(rating_count + 1),
    # lambda avg_rating, rating_count: rating_count,
    lambda avg_rating, rating_count: avg_rating - avg_rating / np.log10(rating_count + 10),
    lambda avg_rating, rating_count: rating_count,
]

if __name__ == "__main__":
    main()