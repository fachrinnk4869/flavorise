from playwright.sync_api import sync_playwright, Playwright
from rich import print
import json

BASE_URL = "https://cookpad.com/"

def run(playwright: Playwright):
    start_url = "https://cookpad.com/id"
    chrome = playwright.chromium
    browser = chrome.launch(headless=False)
    page = browser.new_page()
    page.goto(start_url)
    
    while True:
        for link in page.locator(
            "a[class='uppercase']"
        ).all():
            category_page = browser.new_page(base_url=BASE_URL)
            url = link.get_attribute("href")
            print(url.split('/')[-1])
            if url is not None:
                category_page.goto(url)
            else:
                category_page.close()

            while True:
                for data in category_page.locator(
                    "a[class='block-link__main']"
                ).all():
                    detail_page = browser.new_page(base_url=BASE_URL)
                    detail_url = data.get_attribute("href")
                    
                    if detail_url is not None:
                        detail_page.goto(detail_url)
                    else:
                        detail_page.close()

                    data = detail_page.locator("script[type='application/ld+json']").all()[1].text_content()
                    detail_page.close()

                category_page.close()

with sync_playwright() as playwright:
    run(playwright)