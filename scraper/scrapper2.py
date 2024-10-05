# For microsoft edge

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
import csv

# Set up Selenium WebDriver for Microsoft Edge using EdgeChromiumDriverManager
service = Service(EdgeChromiumDriverManager().install())
driver = webdriver.Edge(service=service)

url = "https://ne.wikipedia.org/wiki/%E0%A4%85_%E0%A4%AC%E0%A5%8D%E0%A4%B0%E0%A4%BF%E0%A4%AB_%E0%A4%B9%E0%A4%BF%E0%A4%B8%E0%A5%8D%E0%A4%9F%E0%A5%8D%E0%A4%B0%E0%A5%80_%E0%A4%85%E0%A4%AB_%E0%A4%9F%E0%A4%BE%E0%A4%87%E0%A4%AE"  # Replace with the website you want to scrape
driver.get(url)



p_tags = driver.find_elements(By.TAG_NAME, 'p')


# Open CSV file to write the data
with open('../data/scraped_data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Tag Type', 'Text Content', 'URL or Other Info'])  # Header row

   
   
    # Write data from <p> tags
    for p_tag in p_tags:
        writer.writerow([ p_tag.text, 'N/A'])

# Close the WebDriver
driver.quit()
