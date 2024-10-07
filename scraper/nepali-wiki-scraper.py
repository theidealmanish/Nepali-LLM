from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import csv
import time
import logging
from selenium.common.exceptions import StaleElementReferenceException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('selenium')

# Set up Selenium WebDriver for Google Chrome using ChromeDriverManager
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--remote-debugging-port=9222')

# Initialize the WebDriver
driver = webdriver.Chrome(service=service, options=options)

visited_urls = set()
data = []

def crawl(url):
    try:
        # Visit the URL
        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        # Extract the full page content
        page_content = driver.find_element(By.TAG_NAME, 'body').text.strip()
        data.append((len(data) + 1, page_content))  # ID and content

        # Find all <a> tags on the page
        a_tags = driver.find_elements(By.TAG_NAME, 'a')

        for a_tag in a_tags:
            link = a_tag.get_attribute('href')
            text = a_tag.text.strip()

            # Check if the link is valid and not visited yet
            if link and link.startswith("https://ne.wikipedia.org/wiki/") and link not in visited_urls:
                visited_urls.add(link)  # Mark this URL as visited
                
                # Recursively crawl the new link
                crawl(link)

    except StaleElementReferenceException:
        logger.warning("StaleElementReferenceException caught. Retrying...")
        crawl(url)  # Retry crawling the same URL

# Start crawling from the Nepali Wikipedia main page
start_url = "https://ne.wikipedia.org/wiki/"
crawl(start_url)

# Write collected data to CSV file
with open('./data/data2.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Text'])  # Header row
    writer.writerows(data)  # Write all collected data

# Close the WebDriver
driver.quit()