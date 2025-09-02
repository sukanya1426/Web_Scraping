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
from crawl4ai import AsyncWebCrawler
from firecrawl import FirecrawlApp

load_dotenv()

class Product(BaseModel):
    product_title: str = Field(..., description="Name of the product")
    current_price: str = Field(..., description="Current price in BDT (e.g., '৳ 12,399')")
    original_price: str | None = Field(None, description="Original price in BDT if discounted")
    product_url: str = Field(..., description="Product detail page URL")
    product_img: str | None = Field(None, description="URL of the product image")
    is_free_delivery: bool = Field(default=False, description="Whether delivery is offered for free")
    source: str = Field(..., description="Which crawler was used (crawl4ai or firecrawl)")

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

async def scrape_with_crawl4ai(url: str, crawler: AsyncWebCrawler) -> List[Dict[str, Any]]:
    """Scrape a single page using Crawl4AI"""
    try:
        result = await crawler.arun(
            url=url,
            use_playwright=True,
            word_count_threshold=1,
            bypass_cache=True,
            playwright_options={
                "headless": True,
                "slow_mo": 50,
                "viewport": {"width": 1366, "height": 2000},
            },
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.7",
            },
            js_code="""
                async function wait(ms) { return new Promise(r => setTimeout(r, ms)); }
                
                (function dismissPopups() {
                    const texts = ["accept", "agree", "ok", "got it"];
                    document.querySelectorAll("button, a").forEach(btn => {
                        const t = (btn.innerText || "").toLowerCase();
                        if (texts.some(x => t.includes(x))) {
                            try { btn.click(); } catch(_e){}
                        }
                    });
                })();
                
                for (let i=0; i<6; i++) {
                    window.scrollTo(0, document.body.scrollHeight);
                    await wait(1200);
                }

                function firstNonEmpty(...els) {
                    for (const el of els) {
                        if (!el) continue;
                        const t = (el.textContent || "").trim();
                        if (t) return t;
                        const v = el.getAttribute?.("content") || el.getAttribute?.("title") || el.getAttribute?.("alt");
                        if (v && v.trim()) return v.trim();
                    }
                    return "";
                }

                const containers = Array.from(document.querySelectorAll(
                    '[data-qa-locator="product-item"], .gridItem--Yd0sa, .box--ujueT, .c2prKC, li, div'
                ));

                const products = [];
                const seen = new Set();

                containers.forEach(card => {
                    const a = card.querySelector('a[title], a[href*="/products/"], a[href*="-i"]');
                    if (!a) return;
                    const href = a.getAttribute('href') || '';
                    if (!href.includes("/products/") && !/-i\\d+\\.html/i.test(href)) return;

                    if (seen.has(href)) return;
                    seen.add(href);

                    const title = firstNonEmpty(
                        a,
                        card.querySelector('.title, .title--wFj93, [data-qa-locator="product-title"], img[alt]')
                    );
                    
                    const priceEl = card.querySelector(
                        '.price, .price--NVB62, .c3gUW0, .c3gUW0 span, .c13VH6, span[class*="price"], .pdp-price, .ooOXD'
                    );
                    const priceNow = priceEl ? priceEl.textContent.trim() : '';
                    
                    if (!title || !priceNow) return;

                    const img = card.querySelector('img')?.getAttribute('src') || '';
                    const freeDelivery = (card.textContent || "").toLowerCase().includes("free delivery");

                    products.push({
                        product_title: title,
                        current_price: priceNow,
                        product_url: href,
                        product_img: img || null,
                        is_free_delivery: freeDelivery,
                        source: "crawl4ai"
                    });
                });

                const holder = document.createElement('script');
                holder.id = '__SCRAPED__';
                holder.type = 'application/json';
                holder.textContent = JSON.stringify({ products });
                document.body.appendChild(holder);

                return JSON.stringify({ products });
            """
        )

        if not result.html:
            return []

        soup = BeautifulSoup(result.html, "html.parser")
        script = soup.select_one("script#__SCRAPED__")
        
        if script and script.string and script.string.strip():
            try:
                payload = json.loads(script.string.strip())
                products = payload.get("products", [])
                if products:
                    print(f"Successfully extracted {len(products)} products from JS data")
                    return products
                else:
                    print("No products found in JS data")
            except json.JSONDecodeError as e:
                print(f"Failed to parse JS-injected data: {e}")

        # Fallback to soup parsing if JS data not available
        products = []
        cards = soup.select('[data-qa-locator="product-item"], .gridItem--Yd0sa, .box--ujueT, .c2prKC, li, div')
        
        for card in cards:
            try:
                a = card.select_one('a[title], a[href*="/products/"], a[href*="-i"]')
                if not a:
                    continue

                href = a.get("href", "")
                if not ("/products/" in href or re.search(r"-i\d+\.html", href)):
                    continue

                title = (a.get("title") or a.get_text(strip=True))[:500]
                price_el = card.select_one('.price, .price--NVB62, .c3gUW0, .c13VH6, span[class*="price"], .pdp-price, .ooOXD')
                current_price = normalize_price(text_or_none(price_el) or card.get_text(" ", strip=True))

                if not title or not current_price:
                    continue

                img_el = card.select_one('img[data-qa-locator="product-image"], img')
                img = img_el.get("src") or img_el.get("data-src") if img_el else None

                free_text = card.get_text(" ", strip=True).lower()
                free_delivery = "free delivery" in free_text

                products.append({
                    "product_title": title,
                    "current_price": current_price,
                    "product_url": href,
                    "product_img": img,
                    "is_free_delivery": free_delivery,
                    "source": "crawl4ai"
                })

            except Exception as e:
                print(f"Error parsing product card: {e}")
                continue

        return products

    except Exception as e:
        print(f"Crawl4AI error: {e}")
        return []

async def scrape_with_firecrawl(url: str, app: FirecrawlApp) -> List[Dict[str, Any]]:
    """Scrape a single page using FireCrawl"""
    try:
        crawl_result = app.scrape(
            url,
            formats=["html"],
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.7",
            }
        )

        html_content = getattr(crawl_result, "html", "")
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        products = []

        cards = soup.select('div[data-qa-locator="product-item"], div[data-spm="sku"] > div, .gridItem--Yd0sa, .box--ujueT, .c2prKC, li, div')
        for card in cards:
            try:
                a = card.select_one('a[title], a[href*="/products/"], a[href*="-i"]')
                if not a:
                    continue

                href = a.get("href", "")
                if not ("/products/" in href or re.search(r"-i\d+\.html", href)):
                    continue

                title = (a.get("title") or a.get_text(strip=True))[:500]
                price_el = card.select_one('.price, .price--NVB62, .c3gUW0, .c13VH6, .pdp-price, span[class*="price"], .ooOXD')
                current_price = normalize_price(text_or_none(price_el) or card.get_text(" ", strip=True))

                if not title or not current_price:
                    continue

                img_el = card.select_one('img[data-qa-locator="product-image"], img')
                img = img_el.get("src") or img_el.get("data-src") if img_el else None

                original_el = card.select_one('del, .price-original, .discount-original, .c1hkC1')
                original_price = normalize_price(text_or_none(original_el))

                free_text = card.get_text(" ", strip=True).lower()
                free_delivery = "free delivery" in free_text

                products.append({
                    "product_title": title,
                    "current_price": current_price,
                    "original_price": original_price,
                    "product_url": href,
                    "product_img": img,
                    "is_free_delivery": free_delivery,
                    "source": "firecrawl"
                })

            except Exception as e:
                print(f"Error parsing product card: {e}")
                continue

        return products

    except Exception as e:
        print(f"FireCrawl error: {e}")
        return []

async def main():
    base_url = "https://www.daraz.com.bd/smartphones/"
    max_pages = 10
    current_page = 1
    all_products: List[Dict[str, Any]] = []

    
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client["daraz_scraping"]
    combined_collection = db["products_combined"]

    # Initialize crawlers
    async with AsyncWebCrawler(verbose=True) as crawler:
        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

        while current_page <= max_pages:
            url = base_url if current_page == 1 else f"{base_url}?page={current_page}"
            print(f"\n==== Scraping page {current_page}: {url}")

            # Try Crawl4AI first
            print("Attempting with Crawl4AI...")
            products = await scrape_with_crawl4ai(url, crawler)
            
            # Add a small delay before potential fallback
            await asyncio.sleep(random.uniform(1.0, 2.0))

            # If Crawl4AI fails or returns no products, fallback to FireCrawl
            if not products:
                print(f"Crawl4AI returned {len(products) if products else 0} products. Falling back to FireCrawl...")
                products = await scrape_with_firecrawl(url, app)
                print(f"FireCrawl returned {len(products) if products else 0} products")

            if not products:
                print("Both crawlers failed. Stopping pagination.")
                break

            # Process products
            cleaned: List[Dict[str, Any]] = []
            for p in products:
                p["product_url"] = to_abs(p.get("product_url"), url)
                p["product_img"] = to_abs(p.get("product_img"), url)

                try:
                    validated = Product(**p).model_dump()
                    cleaned.append(validated)
                except ValidationError as ve:
                    print(f"Skip invalid product: {p.get('product_title')} -> {ve.errors()}")

            # Deduplicate
            before = len(cleaned)
            cleaned = unique_by_url(cleaned)
            after = len(cleaned)
            print(f"Validated {after}/{before} products on page {current_page}")

            all_products.extend(cleaned)
            current_page += 1

            # Add delay between requests
            await asyncio.sleep(random.uniform(2.0, 4.0))

    # Final deduplication and storage
    final_products = unique_by_url(all_products)
    print(f"\nTotal unique products to insert: {len(final_products)}")

    if final_products:
        try:
            result = combined_collection.insert_many(final_products)
            print(f"✓ Inserted {len(result.inserted_ids)} products into products_combined")
            print(f"✓ Total documents in collection: {combined_collection.count_documents({})}")
        except Exception as e:
            print(f"MongoDB insert error: {e}")
    else:
        print("✗ No valid products found to store")

    client.close()

if __name__ == "__main__":
    asyncio.run(main())
