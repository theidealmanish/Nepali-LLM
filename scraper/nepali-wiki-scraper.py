from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from bs4 import BeautifulSoup
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options()
# chrome_options.add_argument("--headless")  # Commented out to display the browser
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
# Optional: Set window size for better visibility
chrome_options.add_argument("window-size=1200,600")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

wiki_url = "https://ne.wikipedia.org/wiki/"

try:
    driver.get(wiki_url)

    # Wait until the page is fully loaded
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "a"))
    )

    # Collect all valid links first to avoid stale elements
    a_tags = driver.find_elements(By.TAG_NAME, "a")
    
    # Helper function to detect if text is entirely Nepali
    def is_fully_nepali(text):
        for char in text:
            if not ('\u0900' <= char <= '\u097F' or char.isspace() or char in ['।', '॥', ',', '.', '!', '?', '॥']):
                return False
        return len(text) > 0  # Ensure that text is not empty

    # Extract unique Nepali links
    links = []
    for a_tag in a_tags:
        link = a_tag.get_attribute('href')
        text = a_tag.text.strip()

        if link and 'ne.wikipedia.org/wiki/' in link and is_fully_nepali(text):
            if link not in links:
                links.append(link)

    data = []
    visited_links = set()
    serial_no = 1

    for link in links:
        if link in visited_links:
            continue
        visited_links.add(link)

        try:
            driver.get(link)
            
            # Wait until the content is loaded
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "p"))
            )

            # Scrape the article content
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            paragraphs = soup.find_all('p')
            article_text = " ".join([
                p.get_text(strip=True) 
                for p in paragraphs 
                if is_fully_nepali(p.get_text(strip=True))
            ])

            if article_text:  # Ensure there's some Nepali text
                data.append({
                    "Id": serial_no,
                    "Article Link": link,
                    "Text": article_text
                })
                serial_no += 1

        except Exception as e:
            print(f"Error processing {link}: {e}")
            continue

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV with UTF-8 encoding to handle Nepali characters
    df.to_csv("nepali_wikipedia_articles.csv", index=False, encoding='utf-8')

    print("Scraping completed successfully.")

except Exception as main_e:
    print(f"An error occurred: {main_e}")

finally:
    driver.quit()
