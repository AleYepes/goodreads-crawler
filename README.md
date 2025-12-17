# Goodreads Crawler & Ranker

An async crawler and recommender system that finds good books to read.

## Setup 
1. **Save Books to Your Goodreads Library:**

    <img src="assets/button.png" alt="alt text" width="300">
   
   The books in your library seed the crawler.<br>
   Books marked as 'Read' are used to train your personal model.

2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

3. **Configure Credentials:**
   Create a `.env` file in the root:
   ```ini
   GOODREADS_EMAIL=your_email@example.com
   GOODREADS_PASSWORD=your_password
   ```

## Usage
1. **Crawl Goodreads:**
    ```bash
    python3 crawler.py
    ```
    *Optional Flag*: `--fd` prompts a fresh download of your Goodreads library.

2. **Train & Get Recommendations**
    ```bash
    python3 recommend.py
    ```
    Outputs a ranked list of books tailored to your tastes.