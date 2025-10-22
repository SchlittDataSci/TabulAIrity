import requests
import re

from bs4 import BeautifulSoup
from random import choice, uniform
from time import sleep



def scrapePageText(url: str) -> str:
    """
    Fetches a webpage's HTML, parses it with BeautifulSoup, and extracts
    clean, sentence-structured text content.

    This function is designed to strip away HTML tags, scripts, styles, and
    non-prose text (like navigation links or footers) to return a clean
    plain text version of the main content.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: A string containing the cleaned, plain text version of the page,
             or an error message if the page cannot be fetched or parsed.
    """
    sleep(uniform(0.5,1.5))
    
    try:
        requestHeaders = [{'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
                          {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"},
                          {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15'}]

        response = requests.get(url, headers=choice(requestHeaders), timeout=10)
        response.raise_for_status()
        htmlContent = response.text

        soup = BeautifulSoup(htmlContent, 'html.parser')
        for scriptOrStyle in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            scriptOrStyle.decompose()

        pageText = soup.get_text()
        textLines = (line.strip() for line in pageText.splitlines())

        proseLines = []
        for line in textLines:
            if line and line.count(' ') > 2:
                proseLines.append(line)

        cleanedText = '\n'.join(proseLines)
        cleanedText = re.sub(r'\n\s*\n', '\n\n', cleanedText)

        return cleanedText.strip()

    except requests.exceptions.RequestException as e:
        return f"Error: Could not retrieve the webpage. Please check the URL and your connection. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"