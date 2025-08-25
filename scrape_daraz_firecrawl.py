import os
import re
import json
import time
import random
import asyncio
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pymongo import MongoClient
from bs4 import BeautifulSoup
from firecrawl import FirecrawlApp

load_dotenv()


class Product(BaseModel):
    product_title: str = Field(..., description="Name of the product")
    current_price: str = Field(..., description="Current price in BDT (e.g., '৳ 12,399')")
    original_price: str | None = Field(None, description="Original price in BDT if discounted")
    product_url: str = Field(..., description="Product detail page URL")
    is_free_delivery: bool = Field(default=False, description="Whether delivery is offered for free")


# Helpers

def to_abs(url: str | None, base: str) -> str | None:
    if not url:
        return None
    if url.startswith("//"):
        return "https:" + url
    if bool(urlparse(url).netloc):
        return url
    return urljoin(base, url)

def text_or_none(el):
    return el.get_text(strip=True) if el else None

def normalize_price(s: str | None) -> str | None:
    if not s:
        return None
    price_pattern = r'(?:৳|Tk\.?)\s*([\d,]+(?:,\d+)*)'
    match = re.search(price_pattern, str(s))
    if not match:
        return None
    price_str = match.group(1)
    return f"৳ {price_str}"

def parse_float_or_none(s: str | None) -> float | None:
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    try:
        return float(m.group(1)) if m else None
    except:
        return None

def unique_by_url(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        k = r.get("product_url")
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

async def main():
    base_url = "https://www.daraz.com.bd/smartphones/"
    max_pages = 10  
    test_url = "https://docs.firecrawl.dev"  

   
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client["daraz_scraping"]
    products3_collection = db["products3"]  

   

    all_products: List[Dict[str, Any]] = []

    
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    
    print("\n==== Testing Firecrawl with test URL ====")
    test_result = app.scrape(test_url, formats=["html", "markdown"])
    print(f"Test result: html length = {len(getattr(test_result, 'html', ''))}, markdown length = {len(getattr(test_result, 'markdown', ''))}")
    if not getattr(test_result, 'html', '') and not getattr(test_result, 'markdown', ''):
        print("Error: Firecrawl test failed. Check API key or network.")

    for page_num in range(1, max_pages + 1):
        url = base_url if page_num == 1 else f"{base_url}?page={page_num}"
        print(f"\n==== Scraping page {page_num}: {url}")

        try:
           
            crawl_result = app.scrape(
                url,
                formats=["html", "markdown"],
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.7",
                    "Connection": "keep-alive"
                }
            )

            
            print(f"Debug: crawl_result type: {type(crawl_result)}, attributes: {dir(crawl_result)}")
            if hasattr(crawl_result, "metadata"):
                print(f"Debug: metadata: {getattr(crawl_result, 'metadata', {})}")

            
            html_content = getattr(crawl_result, "html", "")
            markdown_content = getattr(crawl_result, "markdown", "")
            if not html_content and not markdown_content:
                print("Warning: No content extracted from Firecrawl response.")

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            page_products = []

            # Extract product cards
            product_cards = soup.select('div[data-qa-locator="product-item"], div[data-spm="sku"] > div, .gridItem--Yd0sa, .box--ujueT, .c2prKC, li, div')
            found = 0
            for card in product_cards:
                a = card.select_one('a[title], a[href*="/products/"], a[href*="-i"]')
                if not a:
                    continue
                href = a.get("href") or ""
                if not ("/products/" in href or re.search(r"-i\d+\.html", href or "")):
                    continue
                title = (a.get("title") or a.get_text(strip=True) or "")[:500]
                price_el = card.select_one('.price, .price--NVB62, .c3gUW0, .c13VH6, .pdp-price, span[class*="price"], .ooOXD')
                current_price = normalize_price(text_or_none(price_el) or card.get_text(" ", strip=True))
                if not title or not current_price:
                    continue
                img_el = card.select_one('img[data-qa-locator="product-image"], img')
                img = img_el.get("src") or img_el.get("data-src") if img_el else None
                original_el = card.select_one('del, .price-original, .discount-original, .c1hkC1')
                original_price = normalize_price(text_or_none(original_el))
                rating_el = card.select_one('[data-qa-locator="product-rating"], .rating, span[class*="rating"]')
                rating = parse_float_or_none(text_or_none(rating_el))
                free_text = card.get_text(" ", strip=True).lower()
                free_delivery = "free delivery" in free_text

                page_products.append({
                    "product_title": title,
                    "current_price": current_price,
                    "original_price": original_price,
                    "rating": rating,
                    "product_url": href,
                    "product_img": img,
                    "is_free_delivery": free_delivery,
                    "pagination_url": url,
                })
                found += 1
            print(f"Soup fallback found {found} products")

            # Normalize and validate
            cleaned: List[Dict[str, Any]] = []
            for p in page_products:
                p["product_url"] = to_abs(p.get("product_url"), url)
                p["product_img"] = to_abs(p.get("product_img"), url)
                p["current_price"] = normalize_price(p.get("current_price"))
                p["original_price"] = normalize_price(p.get("original_price"))
                if isinstance(p.get("rating"), str):
                    p["rating"] = parse_float_or_none(p["rating"])

                try:
                    validated = Product(**p).model_dump()  
                    cleaned.append(validated)
                except ValidationError as ve:
                    print(f"Skip invalid product (validation): {p.get('product_title')} -> {ve.errors()}")

            # Deduplicate by URL
            before = len(cleaned)
            cleaned = unique_by_url(cleaned)
            after = len(cleaned)
            print(f"Validated {after}/{before} products on page {page_num}")

            all_products.extend(cleaned)

            # Stop if last page
            next_page = soup.select_one('li[title="Next Page"]')
            is_last_page = not next_page or next_page.get("aria-disabled") == "true"
            if is_last_page:
                print("Reached last page (or Next disabled). Stopping pagination.")
                break

           
            await asyncio.sleep(random.uniform(2.0, 4.0))

        except Exception as e:
            print(f"ERROR scraping page {page_num}: {e}")
            break

    
    final_products = unique_by_url(all_products)
    print(f"\nTotal unique products to insert: {len(final_products)}")

    if final_products:
        try:
            result = products3_collection.insert_many(final_products)
            print(f"✓ Inserted {len(result.inserted_ids)} products into products3")
            print(f"✓ Total documents in products3: {products3_collection.count_documents({})}")
        except Exception as e:
            print(f"Mongo insert error: {e}")
    else:
        print("✗ No valid products found to store")

    client.close()

if __name__ == "__main__":
    asyncio.run(main())