from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import json, time, random, sys

# url
BASE_URL = "https://cookpad.com/"
START_URL = "https://cookpad.com/id"

HEADLESS = True                     # show the page (False) or not (True)
MAX_CATEGORIES = None               # number of category want to fetch
MAX_RECIPES_PER_CATEGORY = 100      # number of recipe each category
MAX_LOAD_ROUNDS = 80                # max number of reload to get recipe data
PLATEAU_ROUNDS = 5                  # max number of retry to get additional recipe data
NETWORK_PAUSE = (0.7, 1.3)          # random delay number
OUTFILE = "cookpad_recipe_all.json" # output filename

# browser identifier
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/123.0.0.0 Safari/537.36")

# concat base url and href url
def abs_url(href: str) -> str:
    return urljoin(BASE_URL, href)

# sleep function
def polite_wait():
    time.sleep(random.uniform(*NETWORK_PAUSE))

# get list of url category
def get_category_links(page):
    # get href or link of each category
    links = []
    for a in page.locator("a.uppercase[href^='/id/cari/']").all():
        href = a.get_attribute("href")
        if href:
            links.append(abs_url(href))
    
    # prevent duplicate
    seen, out = set(), []
    for u in links:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

# get list of url recipes
def get_recipe_links(page):
    # get href of each recipe
    links = []
    for a in page.locator("a.block-link__main[href^='/id/resep/']").all():
        href = a.get_attribute("href")
        if href:
            links.append(abs_url(href))
    
    # prevent duplicate
    seen, out = set(), []
    for u in links:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

# load more data in category page
def load_more(page) -> bool:
    btn = page.locator("#search-recipes-pagination a[rel='next']")
    if btn.count() == 0:
        # there is no button
        return False
    try:
        if btn.is_visible():
            # expand data if available
            btn.click(timeout=2000)
            return True
    except Exception:
        return False
    return False

# scroll to bottom of the page to trigger load more data
def trigger_scroll(page):
    page.evaluate("window.scrollTo(0, document.body.scrollHeight);")

# get recipe data as much as possible per category
def get_recipe_data(ctx, category_url):
    # open page
    page = ctx.new_page()
    print(f"Kategori: {category_url}")
    page.goto(category_url, wait_until="domcontentloaded", timeout=60_000)

    # init variable
    plateau = 0
    seen_links = []

    # get as much data as possible use load more or trigger in one page
    for _ in range(MAX_LOAD_ROUNDS):
        before = page.locator("a.block-link__main[href^='/id/resep/']").count()
        
        # load more data by click load more or trigger from scroll
        clicked = load_more(page)
        if not clicked:
            trigger_scroll(page)
        polite_wait()

        # wait if there is new data from load more
        try:
            page.wait_for_function(
                "prev => document.querySelectorAll(\"a.block-link__main[href^='/id/resep/']\").length > prev",
                arg=before,
                timeout=3500
            )
        except PWTimeoutError:
            pass
        
        # get recipe link and check if there is additional recipe data
        # assume there is no additional data if plateau === PLATEAU_ROUNDS
        current = get_recipe_links(page)
        if len(current) > len(seen_links):
            seen_links = current
            plateau = 0
            print(f"Total recipe: {len(seen_links)}")
        else:
            plateau += 1

        # limit fetch data if already achieve max data needed
        if len(seen_links) >= MAX_RECIPES_PER_CATEGORY:
            print("Reach limit max recipe per category")
            break
        if plateau >= PLATEAU_ROUNDS:
            print("Stopped, there is no new item")
            break

    page.close()
    return seen_links[:MAX_RECIPES_PER_CATEGORY]

# extract recipe data from html
def parse_data(html: str, url: str):
    soup = BeautifulSoup(html, "lxml")

    data = None
    # locate the json-ld that contain a recipe data
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            obj = json.loads(tag.string) if tag.string else None
        except Exception:
            continue
        if not obj:
            continue

        candidates = []
        if isinstance(obj, dict):
            candidates = [obj]
            if "@graph" in obj and isinstance(obj["@graph"], list):
                candidates.extend(obj["@graph"])
        elif isinstance(obj, list):
            candidates = obj

        for node in candidates:
            if isinstance(node, dict) and node.get("@type") in ("Recipe", ["Recipe"]):
                data = node
                break
        if data:
            break
    
    # if there is no recipe in json-ld
    if not data:
        title = soup.select_one("h1")
        return {
            "url": url,
            "title": title.get_text(strip=True) if title else None,
            "image": None,
            "ingredients": [],
            "steps": []
        }

    # parse url image
    def norm_image(img):
        if isinstance(img, str):
            return img
        if isinstance(img, list) and img:
            if isinstance(img[0], str):
                return img[0]
            if isinstance(img[0], dict) and "url" in img[0]:
                return img[0]["url"]
        if isinstance(img, dict) and "url" in img:
            return img["url"]
        return None

    # parse recipe instruction into list of {text, images[]}
    def parse_steps(instr):
        steps = []
        if isinstance(instr, str):
            steps.append({"text": instr.strip(), "images": []})
        elif isinstance(instr, list):
            for it in instr:
                if isinstance(it, str):
                    steps.append({"text": it.strip(), "images": []})
                elif isinstance(it, dict):
                    txt = (it.get("text") or it.get("name") or "").strip()
                    imgs = it.get("image")
                    imgs_out = []
                    if isinstance(imgs, str):
                        imgs_out = [imgs]
                    elif isinstance(imgs, list):
                        imgs_out = [x for x in imgs if isinstance(x, str)]
                    steps.append({"text": txt, "images": imgs_out})
        return steps

    return {
        "url": data.get("url") or url,
        "title": data.get("name"),
        "image": norm_image(data.get("image")),
        "ingredients": data.get("recipeIngredient") or [],
        "steps": parse_steps(data.get("recipeInstructions"))
    }

# get html of recipe data
def fetch_recipe_data(ctx, url, referer="https://cookpad.com/id"):
    headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id,en-US;q=0.8,en;q=0.6",
        "Referer": referer,
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    backoff = 1.0
    last_status = None
    last_snippet = None

    # iterate to retry if failed to fetch
    for attempt in range(4):
        # get html data of the url
        resp = ctx.request.get(url, headers=headers, timeout=20000)
        last_status = resp.status
        if resp.ok:
            # parse the html data
            html = resp.text()
            return parse_data(html, url)
        
        # get text for return data
        try:
            txt = resp.text() or ""
            last_snippet = txt[:300]
        except Exception:
            last_snippet = None

        # retry for 400/403/429/500+
        if resp.status in (400, 403, 429) or resp.status >= 500:
            time.sleep(backoff)
            backoff *= 1.8
            continue
        break

    # return status if error
    err = f"{last_status} Bad Response"
    if last_snippet:
        err += f" | body: {last_snippet!r}"
    return {"url": url, "error": err}

def main():
    all_results = []

    with sync_playwright() as p:
        # launch page and go to start url
        browser = p.chromium.launch(headless=HEADLESS)
        ctx = browser.new_context(
            locale="id-ID",
            user_agent=UA,
            viewport={"width": 1366, "height": 900}
        )
        page = ctx.new_page()
        page.goto(START_URL, wait_until="domcontentloaded", timeout=60_000)

        # scrape category url from main page
        category_urls = get_category_links(page)

        # cut category if needed
        if MAX_CATEGORIES:
            category_urls = category_urls[:MAX_CATEGORIES]
        print(f"Total Category: {len(category_urls)}")

        # iterate of each category
        total_seen = set() # prevent duplication
        for idx, cat in enumerate(category_urls, 1):
            print(f"({idx}/{len(category_urls)}) For Category : {cat}")
            # get recipe url for each category
            recipe_urls = get_recipe_data(ctx, cat)
            category = cat.split('/')[-1] # get category from url
            print(f"Total Data: {len(recipe_urls)}")

            # iterate from recipe data
            for i, url in enumerate(recipe_urls, 1):
                # skip duplicate urk 
                if url in total_seen:
                    continue
                # add url and wait
                total_seen.add(url)
                polite_wait()

                # get needed data from recipe
                data = fetch_recipe_data(ctx, url, referer=cat)
                data['category'] = category
                all_results.append(data)

                # logging
                if i % 10 == 0:
                    print(f"{i} data succesfully collected")

        # close all pages
        browser.close()

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Done, Total data saved : {len(all_results)} in {OUTFILE}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user")
        sys.exit(1)