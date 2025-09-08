import asyncio
import os
import logging
import time
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler
from firecrawl import Firecrawl
from bs4 import BeautifulSoup
from pymongo import MongoClient, errors
from pymongo.errors import ConnectionFailure
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

ROOT_URL = "https://www.startech.com.bd/"
MONGO_URI = os.getenv("MONGODB_URI") 
if not MONGO_URI:
    raise ValueError("MONGODB_URI environment variable not set")

logging.info("Using MongoDB URI: %s", MONGO_URI.split('@')[0] + '@' + MONGO_URI.split('@')[1].split('/')[0])

DB_NAME = "startech"
COLLECTION_NAME = "products"


MAX_WORKERS = 3  # Reduced number of concurrent threads
BATCH_SIZE = 50  # Smaller batch size for MongoDB inserts
REQUEST_DELAY = 2  # Delay between requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for MongoDB operations
RETRY_DELAY = 5  # Delay between retries in seconds


def init_mongodb():
    for attempt in range(MAX_RETRIES):
        try:
            
            client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=30000,  # Increased timeout
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True,
                w='majority',  # Ensure writes are acknowledged by majority
                journal=True,  # Wait for journal commit
            )
            
            
            
            
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            
          
            collection.create_index([("url", 1)], unique=True, background=True)
            logging.info(f"Initialized collection {DB_NAME}.{COLLECTION_NAME}")
            
            return collection
        except (ConnectionFailure, errors.ServerSelectionTimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                logging.warning(f"MongoDB Atlas connection attempt {attempt + 1} failed: {str(e)}")
                logging.warning(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logging.error("Failed to connect to MongoDB Atlas after all retries")
                logging.error(f"Last error: {str(e)}")
                raise
        except Exception as e:
            logging.error(f"Unexpected error connecting to MongoDB Atlas: {str(e)}")
            if "SSL: CERTIFICATE_VERIFY_FAILED" in str(e):
                logging.error("SSL Certificate verification failed. Check your MongoDB Atlas connection string or SSL settings.")
            raise


collection = init_mongodb()

firecrawl = Firecrawl()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_html(url: str) -> str:
    """Fetch page HTML with Crawl4AI, fallback to FireCrawl."""
    await asyncio.sleep(REQUEST_DELAY)  # Add delay between requests
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, use_playwright=True)
            if result and result.html:
                return result.html
    except Exception as e:
        logging.warning(f"Crawl4AI failed for {url}: {e}")

    try:
        # Fixed FireCrawl params
        result = firecrawl.crawl(url)
        if isinstance(result, dict) and "html" in result:
            return result["html"]
        return result
    except Exception as e:
        logging.error(f"FireCrawl failed for {url}: {e}")
        raise  # Let the retry decorator handle it

async def discover_categories(root_url: str):
    """Extract root categories and first-level subcategories only."""
    logging.info(f"==== Extracting categories from homepage: {root_url}")
    html = await fetch_html(root_url)
    if not html:
        logging.error("Failed to fetch homepage.")
        return []

    soup = BeautifulSoup(html, "html.parser")
    categories = set()
    root_categories = set()

    # Extract root categories from main navigation
    for a in soup.select("nav.navbar .nav-item > a.nav-link"):
        href = a.get("href", "").strip()
        if not href or href == "#" or href == "/":
            continue
        
        full_url = urljoin(root_url, href)
        path = urlparse(full_url).path.strip("/")
        if "/" not in path:  # Root category
            root_categories.add(full_url)
            categories.add(full_url)

    # Extract first-level subcategories from dropdowns
    for a in soup.select("nav.navbar .dropdown-menu a"):
        href = a.get("href", "").strip()
        if not href or href == "#" or href == "/":
            continue

        full_url = urljoin(root_url, href)
        path = urlparse(full_url).path.strip("/")
        path_parts = path.split("/")
        
        # Only keep first-level subcategories
        if len(path_parts) == 2 and any(root_cat.endswith(path_parts[0]) for root_cat in root_categories):
            categories.add(full_url)

    filtered = sorted(list(categories))
    logging.info(f"Found {len(filtered)} categories:")
    for cat in filtered:
        logging.info(f"  - {cat}")
    return filtered

async def scrape_category(url: str):
    """Scrape all products from a category page with their details."""
    logging.info(f"==== Scraping category page: {url}")
    html = await fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    products = []

    # Get category name from URL
    category = urlparse(url).path.strip("/").split("/")[0]

    # Try different product card selectors used by StarTech
    for product in soup.select(".p-item, .product-layout"):
        try:
            # Extract product details
            title_elem = product.select_one(".p-item-name, .product-name, h4 a")
            price_elem = product.select_one(".p-item-price, .price-new, .price")
            status_elem = product.select_one(".p-item-stock, .stock-status, .status")
            url_elem = product.select_one("h4 a, .p-item-name a, .product-name a")

            if not title_elem:
                continue

            # Get product URL
            if url_elem:
                product_url = urljoin(url, url_elem["href"])
            else:
                # Try to find URL from other possible elements
                url_elem = product.select_one("a")
                if not url_elem:
                    continue
                product_url = urljoin(url, url_elem["href"])
            
            # Clean up and extract text content
            title = title_elem.text.strip()
            price = price_elem.text.strip() if price_elem else "N/A"
            status = status_elem.text.strip() if status_elem else "N/A"
            
            # Create product data
            product_data = {
                "url": product_url,
                "category": category,
                "title": title,
                "price": price,
                "status": status,
                "scraped_from": url
            }
            
            products.append(product_data)
            logging.debug(f"Found product: {title}")
            
        except Exception as e:
            logging.error(f"Error extracting product data: {e}")
            continue

    logging.info(f"Found {len(products)} products in category: {url}")
    return products

def save_products(products):
    if not products:
        return 0
        
    try:
        # Update with upsert to avoid duplicates based on URL
        inserted_count = 0
        for product in products:
            result = collection.update_one(
                {"url": product["url"]},  # query
                {"$set": product},        # update
                upsert=True               # insert if not exists
            )
            if result.upserted_id or result.modified_count:
                inserted_count += 1
                
        if inserted_count > 0:
            logging.info(f"Successfully inserted/updated {inserted_count} products in MongoDB.")
        return inserted_count
        
    except Exception as e:
        logging.error(f"MongoDB insert error: {e}")
        return 0

async def process_category_chunk(chunk, semaphore):
    """Process a chunk of categories with concurrency control"""
    processed_products = []
    for category in chunk:
        async with semaphore:  # Limit concurrent requests
            try:
                products = await scrape_category(category)
                if products:
                    processed_products.extend(products)
                logging.info(f"Processed category: {category}")
            except Exception as e:
                logging.error(f"Error processing category {category}: {e}")
            # Add delay between requests
            await asyncio.sleep(REQUEST_DELAY)
    return processed_products

async def main():
    categories = await discover_categories(ROOT_URL)
    if not categories:
        logging.error("No categories found.")
        return

    all_products = []
    total_inserted = 0
    processed = 0
    
    # Create semaphore to limit concurrent connections
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    
    # Process categories in smaller chunks
    chunk_size = MAX_WORKERS * 2  # Process chunks twice the size of workers
    for i in range(0, len(categories), chunk_size):
        category_chunk = categories[i:i + chunk_size]
        
        # Process chunk
        try:
            products = await process_category_chunk(category_chunk, semaphore)
            if products:
                all_products.extend(products)
                processed += len(category_chunk)
                
                # Save products in batches
                if len(all_products) >= BATCH_SIZE:
                    inserted = save_products(all_products)
                    if inserted:
                        total_inserted += inserted
                    all_products = []
                    
                logging.info(f"Progress: {processed}/{len(categories)} categories processed. "
                           f"Total products inserted so far: {total_inserted}")
                
        except Exception as e:
            logging.error(f"Error processing chunk: {e}")
        
        # Add delay between chunks
        await asyncio.sleep(1)

    # Save any remaining products
    if all_products:
        inserted = save_products(all_products)
        if inserted:
            total_inserted += inserted

    logging.info(f"Scraping completed! Total products inserted: {total_inserted}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        