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

load_dotenv()


class Product(BaseModel):
    product_title: str = Field(..., description="Name of the product")
    current_price: str = Field(..., description="Current price in BDT (e.g., '৳ 12,399')")
    product_url: str = Field(..., description="Product detail page URL")
    is_free_delivery: bool = Field(default=False, description="Whether delivery is free")



# Helpers

def to_abs(url: str | None, base: str) -> str | None:
    if not url:
        return None
    # Some Daraz links may start with // or be relative
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
    scroll_rounds = 6         
    scroll_wait_ms = 1200     # per round

   
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client["daraz_scraping"]
    products2_collection = db["products2"]  

    

    all_products: List[Dict[str, Any]] = []

    async with AsyncWebCrawler(verbose=True) as crawler:
        for page_num in range(1, max_pages + 1):
            url = base_url if page_num == 1 else f"{base_url}?page={page_num}"
            print(f"\n==== Scraping page {page_num}: {url}")

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
                        "Connection": "keep-alive",
                    },
                    # We inject scraped JSON into the DOM as <script id="__SCRAPED__" type="application/json">...</script>
                    js_code=f"""
                        async function wait(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

                        // Try dismissing popups (cookie/region)
                        (function dismissPopups() {{
                            const texts = ["accept", "agree", "ok", "got it"];
                            document.querySelectorAll("button, a").forEach(btn => {{
                                const t = (btn.innerText || "").toLowerCase();
                                if (texts.some(x => t.includes(x))) {{
                                    try {{ btn.click(); }} catch(_e){{}}
                                }}
                            }});
                        }})();

                        // Scroll to load products
                        for (let i=0; i<{scroll_rounds}; i++) {{
                            window.scrollTo(0, document.body.scrollHeight);
                            await wait({scroll_wait_ms});
                        }}

                        function firstNonEmpty(...els) {{
                            for (const el of els) {{
                                if (!el) continue;
                                const t = (el.textContent || "").trim();
                                if (t) return t;
                                const v = el.getAttribute?.("content") || el.getAttribute?.("title") || el.getAttribute?.("alt");
                                if (v && v.trim()) return v.trim();
                            }}
                            return "";
                        }}

                        function getPriceFrom(node) {{
                            if (!node) return "";
                            // Common price classes on Daraz vary; try a few
                            const cand = node.querySelector(
                                '.price, .price--NVB62, .c3gUW0, .c3gUW0 span, .c13VH6, span[class*="price"], .pdp-price, .ooOXD'
                            );
                            if (cand) return cand.textContent.trim();
                            // Fallback: any text node with BDT-like numbers
                            const text = node.innerText || "";
                            const m = text.match(/[৳Tk\\.]*\\s?[\\d,]+(?:\\.\\d+)?/);
                            return m ? m[0] : "";
                        }}

                        function getOriginalFrom(node) {{
                            if (!node) return "";
                            const cand = node.querySelector('del, .price-original, .discount-original, .c1hkC1');
                            return cand ? cand.textContent.trim() : "";
                        }}

                        function getRatingFrom(node) {{
                            if (!node) return "";
                            const cand = node.querySelector('[data-qa-locator="product-rating"], .rating, .rating--qP8Rm, span[class*="rating"]');
                            if (cand) return cand.textContent.trim();
                            // Some cards show stars count like "4.8"
                            const text = node.innerText || "";
                            const m = text.match(/(\\d+(?:\\.\\d+)?)(?=\\s*\\/\\s*5|\\s*stars?)/i) || text.match(/\\b(\\d\\.\\d)\\b/);
                            return m ? m[1] : "";
                        }}

                        function getImgFrom(node) {{
                            if (!node) return "";
                            const img = node.querySelector('img[data-qa-locator="product-image"], img, img.lazyload, img.loadable');
                            if (!img) return "";
                            return img.getAttribute("src") || img.getAttribute("data-src") || img.getAttribute("data-ks-lazyload") || "";
                        }}

                        function looksLikeProductURL(href) {{
                            if (!href) return false;
                            return href.includes("/products/") || /-i\\d+\\.html/i.test(href);
                        }}

                        const containers = Array.from(document.querySelectorAll(
                            '[data-qa-locator="product-item"], .gridItem--Yd0sa, .box--ujueT, .c2prKC, li, div'
                        ));

                        const products = [];
                        const seen = new Set();

                        containers.forEach(card => {{
                            // Find the anchor that links to the product detail
                            const a = card.querySelector('a[title], a[href*="/products/"], a[href*="-i"]');
                            if (!a) return;
                            const href = a.getAttribute('href') || '';
                            if (!looksLikeProductURL(href)) return;

                            // Dedup by href within page
                            if (seen.has(href)) return;
                            seen.add(href);

                            const title = firstNonEmpty(
                                a,
                                card.querySelector('.title, .title--wFj93, [data-qa-locator="product-title"], img[alt]')
                            );
                            const priceNow = getPriceFrom(card);
                            if (!title || !priceNow) return; // must have both

                            const img = getImgFrom(card);

                            const rating = getRatingFrom(card);
                            const freeDelivery = !!(card.innerText || "").toLowerCase().includes("free delivery");

                            products.push({{
                                product_title: title,
                                current_price: priceNow,
                                rating: rating || null,
                                product_url: href,
                                product_img: img || null,
                                is_free_delivery: freeDelivery,
                                pagination_url: window.location.href
                            }});
                        }});

                        // Detect if there's a "Next" and whether it's disabled
                        let isLastPage = true;
                        const nextLi = document.querySelector('li[title="Next Page"], .ant-pagination-next');
                        if (nextLi) {{
                            const disabled = nextLi.getAttribute('aria-disabled');
                            isLastPage = disabled === 'true';
                        }}

                        // Inject result into DOM so backend can read it from HTML
                        const payload = {{ products, isLastPage }};
                        const holder = document.createElement('script');
                        holder.id = '__SCRAPED__';
                        holder.type = 'application/json';
                        holder.textContent = JSON.stringify(payload);
                        document.body.appendChild(holder);

                        // Also return something in case runtime supports it
                        JSON.stringify(payload);
                    """,
                )

                # Save HTML + Markdown for debugging
                html_path = f"daraz_page_{page_num}.html"
                md_path = f"daraz_products_page_{page_num}.md"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(result.html or "")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(result.markdown or "")
                print(f"Saved HTML -> {html_path} | Markdown -> {md_path}")

                # Pull injected JSON from the HTML
                soup = BeautifulSoup(result.html or "", "html.parser")
                script = soup.select_one("script#__SCRAPED__")
                page_products: List[Dict[str, Any]] = []
                is_last_page = True

                if script and script.text.strip():
                    try:
                        payload = json.loads(script.text)
                        page_products = payload.get("products", []) or []
                        is_last_page = bool(payload.get("isLastPage", True))
                        print(f"JS extracted {len(page_products)} products (isLastPage={is_last_page})")
                    except json.JSONDecodeError:
                        print("WARN: Could not decode JS payload; falling back to soup-only parsing.")
                else:
                    print("WARN: No injected JSON found; attempting soup-only parsing.")

                # If JS didn’t capture, attempt soup parse (rendered HTML)
                if not page_products:
                    cards = soup.select('[data-qa-locator="product-item"], .gridItem--Yd0sa, .box--ujueT, .c2prKC, li, div')
                    found = 0
                    for card in cards:
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
                        
                    
                        rating_el = card.select_one('[data-qa-locator="product-rating"], .rating, span[class*="rating"]')
                        rating = parse_float_or_none(text_or_none(rating_el))
                        free_text = card.get_text(" ", strip=True).lower()
                        free_delivery = "free delivery" in free_text

                        page_products.append({
                            "product_title": title,
                            "current_price": current_price,
                            
                            "rating": rating,
                            "product_url": href,
                            "product_img": img,
                            "is_free_delivery": free_delivery,
                            "pagination_url": url,
                        })
                        found += 1
                    print(f"Soup fallback found {found} products")
                    # keeping is_last_page True unless we detect otherwise

                # Normalize and validate
                cleaned: List[Dict[str, Any]] = []
                for p in page_products:
                    p["product_url"] = to_abs(p.get("product_url"), url)
                    p["product_img"] = to_abs(p.get("product_img"), url)
                    p["current_price"] = normalize_price(p.get("current_price"))
                   
                    # Ratings: if came as string, coerce to float
                    if isinstance(p.get("rating"), str):
                        p["rating"] = parse_float_or_none(p["rating"])

                    # Validate with Pydantic
                    try:
                        validated = Product(**p).model_dump()
                        cleaned.append(validated)
                    except ValidationError as ve:
                        print(f"Skip invalid product (validation): {p.get('product_title')} -> {ve.errors()}")

                # Deduplicate by URL (within page and across pages)
                before = len(cleaned)
                cleaned = unique_by_url(cleaned)
                after = len(cleaned)
                print(f"Validated {after}/{before} products on page {page_num}")

                all_products.extend(cleaned)

                # Stop if last page
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
           
            result = products2_collection.insert_many(final_products)
            print(f"✓ Inserted {len(result.inserted_ids)} products into products2")
            print(f"✓ Total documents in products2: {products2_collection.count_documents({})}")
        except Exception as e:
            print(f"Mongo insert error: {e}")
    else:
        print("✗ No valid products found to store")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())