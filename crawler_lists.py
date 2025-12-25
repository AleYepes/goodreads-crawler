import re
import csv
import time
from playwright.sync_api import sync_playwright

# Format: https://www.goodreads.com/review/list/75706676?print=true&sort=date_read&view=reviews
LIST_IDS = [75706676, 104945614, 124720847, 91998392, 34518408, 105258888, 129155685, 113185438,
 166997642, 174792571, 13737030, 27115955, 24885719, 51281420, 104343033, 65139494, 115764833]

OUTPUT_FILE = "data/friend_ratings.csv"

RATING_MAP = {
    "it was amazing": 5,
    "really liked it": 4,
    "liked it": 3,
    "it was ok": 2,
    "did not like it": 1,
}

def clean_text(text):
    if text:
        return text.strip().replace("\n", "")
    return ""

def scrape_goodreads():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = context.new_page()

        with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["list_id", "book_id", "title", "rating", "num_pages", "date_read", "date_added"])

            for list_id in LIST_IDS:
                url = f"https://www.goodreads.com/review/list/{list_id}?print=true&sort=rating&view=reviews"
                page.goto(url)
                print(f"Scraping {list_id}")
                page_num = 1
                while True:
                    try:
                        page.wait_for_selector("#booksBody", timeout=10000)
                    except:
                        print(f"Could not find book table for {list_id}. Skipping or finished.")
                        break

                    rows = page.query_selector_all("tr.bookalike.review")
                    for row in rows:
                        try:
                            # 1. Title
                            title_el = row.query_selector(".field.title a")
                            title = clean_text(title_el.inner_text()) if title_el else "Unknown"

                            # 2. Book ID
                            # href format: /book/show/20527133-superintelligence
                            href = title_el.get_attribute("href") if title_el else ""
                            book_id_match = re.search(r'/book/show/(\d+)', href)
                            book_id = book_id_match.group(1) if book_id_match else "Unknown"

                            # 3. Rating
                            rating_el = row.query_selector(".field.rating .staticStars")
                            rating_text = rating_el.get_attribute("title") if rating_el else ""
                            rating = RATING_MAP.get(rating_text, 0) # Default to 0 if unrated

                            # 4. Num Pages
                            pages_el = row.query_selector(".field.num_pages .value")
                            raw_pages = pages_el.text_content() if pages_el else ""
                            num_pages = re.sub(r"[^\d]", "", raw_pages)

                            # 5. Date Read
                            date_read_el = row.query_selector(".field.date_read .date_read_value")
                            date_read = clean_text(date_read_el.inner_text()) if date_read_el else ""

                            # 6. Date Added
                            date_added_el = row.query_selector(".field.date_added span")
                            date_added = ""
                            if date_added_el:
                                date_added = date_added_el.get_attribute("title") 
                                if not date_added:
                                    date_added = clean_text(date_added_el.inner_text())

                            # Write row
                            writer.writerow([list_id, book_id, title, rating, num_pages, date_read, date_added])

                        except Exception as e:
                            print(f"Error extracting row: {e}")
                            continue

                    next_button = page.query_selector("a.next_page")
                    if next_button and "disabled" not in next_button.get_attribute("class"):
                        with page.expect_navigation():
                            next_button.click()
                        print(f'    P{page_num}')
                        page_num += 1
                        
                        time.sleep(1)
                    else:
                        print(f'Finished {list_id}')
                        break

        browser.close()

if __name__ == "__main__":
    scrape_goodreads()