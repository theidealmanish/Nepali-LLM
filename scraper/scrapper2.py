from selenium import webdriver
from selenium.webdriver.common.by import By
import csv

from selenium import webdriver
from webdriver_manager.microsoft import EdgeChromiumDriverManager

driver = webdriver.Edge(EdgeChromiumDriverManager().install())
browser_name = driver.capabilities["browserName"]
print(f"Browser Name: {browser_name}")

# Set up Selenium WebDriver (make sure you have the correct driver installed, like ChromeDriver)
driver = webdriver.Edge(executable_path='./msedgedriver.exe')  # Update the path to your WebDriver
url = "https://example.com"  # Replace with the website you want to scrape
driver.get(url)

# Extract all <a> tags and <p> tags
a_tags = driver.find_elements(By.TAG_NAME, 'a')
p_tags = driver.find_elements(By.TAG_NAME, 'p')

# Open CSV file to write the data
with open('scraped_data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Tag Type', 'Text Content', 'URL or Other Info'])  # Header row

    # Write data from <a> tags
    for a_tag in a_tags:
        writer.writerow(['a', a_tag.text, a_tag.get_attribute('href')])

    # Write data from <p> tags
    for p_tag in p_tags:
        writer.writerow(['p', p_tag.text, 'N/A'])

# Close the WebDriver
driver.quit()
