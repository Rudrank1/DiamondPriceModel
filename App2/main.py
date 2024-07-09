import requests
from bs4 import BeautifulSoup

def scrape_and_print_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    print("Scraped content: " + soup)

url = 'http://myapp:5000'
scrape_and_print_content(url)