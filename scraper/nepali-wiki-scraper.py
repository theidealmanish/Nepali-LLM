from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import logging
from selenium.common.exceptions import StaleElementReferenceException  # {{ edit_1 }}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('selenium')

# Set up Selenium WebDriver for Google Chrome using ChromeDriverManager
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Initialize the WebDriver
driver = webdriver.Chrome(service=service, options=options)

visited_urls = set()
data = []

def crawl(url):
    visited_urls.add(url)  # Mark this URL as visited
    driver.get(url)

    # Wait for the page to load and body tag to be present
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    
    # Extract the full page content
    page_content = driver.find_element(By.TAG_NAME, 'body').text.strip()
    data.append((len(data) + 1, page_content))  # ID and content

    # Find all <a> tags on the page
    a_tags = driver.find_elements(By.TAG_NAME, 'a')

    for a_tag in a_tags[:5]:
        link = a_tag.get_attribute('href')
      

# Start crawling from the Nepali Wikipedia main page
start_url = "https://www.onlinekhabar.com"
crawl(start_url)

# Write collected data to CSV file
with open('../data/data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Text'])  # Header row
    writer.writerows(data)  # Write all collected data


# Close the WebDriver
driver.quit()
