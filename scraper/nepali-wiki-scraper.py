import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to scrape a single Wikipedia page
def scrape_wikipedia_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the main content (usually in <p> tags)
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])

    return text.strip()

# List of URLs to scrape (you can add more)
urls = [
    'https://ne.wikipedia.org/wiki/नेपाल',
    'https://ne.wikipedia.org/wiki/काठमाडौं',
    'https://ne.wikipedia.org/wiki/नेपालको_संस्कृति'
]

# Create a DataFrame to store the results
data = []

for i, url in enumerate(urls):
    text = scrape_wikipedia_page(url)
    data.append({'ID': i + 1, 'Text': text})

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('nepali_wikipedia_data.csv', index=False, encoding='utf-8-sig')

print("Data has been scraped and saved to nepali_wikipedia_data.csv")