"""Utilities to scrape Kijiji ads"""
import pandas as pd
import requests
import structlog
from typing import List 

logger = structlog.getLogger(__name__)

class KijijiScrape:
    """Class to interact with Kijiji html content"""

    def __init__(self, soup):
        """Instantiate

        Args:
            soup: raw html content
        """

        self.soup = soup
    
    def extract_image_cards_url(self) -> List:
        """Parse image url from raw html content

        Returns:
            list: list of image urls
        """
        image_urls = []
        picture_elements = self.soup.find_all("picture")

        for picture in picture_elements:
            img_tag = picture.find("img")
            if img_tag and "data-src" in img_tag.attrs:
                image_url = img_tag["data-src"]
                image_urls.append(image_url)
        return image_urls
    
    def extract_ad_url(self) -> List:
        """Parse the url to the detailed ad page

        Returns:
            list: list of detail ad urls
        """
        ad_ids = []
        ad_urls = []
        ad_titles = []
        detail_urls = self.soup.find_all("a", class_="title")

        href_list = [link["href"] for link in detail_urls]
        title_text = [element.get_text(strip=True) for element in detail_urls]
        for href, title in zip(href_list, title_text):
            ad_urls.append('https://www.kijiji.ca' + href)
            ad_ids.append(href.split('/')[-1])
            ad_titles.append(title)
        return ad_urls, ad_ids, ad_titles

    def extract_ad_summary(self) -> list:
        """Parse the ad summary for each card

        Returns:
            list: list of ad summary
        """
        ad_descriptions = []
        description_elements = self.soup.find_all(class_="description")

        description_texts = [element.get_text(strip=True) for element in description_elements]
        for description_text in description_texts:
            description_text = description_text.replace('...', '')
            ad_descriptions.append(description_text)

        return ad_descriptions

    def download_image_card(self, image_filename, image_link):
        """Download jpg image of each ad card

        Args:
            image_filename: Unique name of ad
            image_link: URL path to image
        """
        response = requests.get(image_link)

        if response.status_code == 200:
            with open(image_filename, "wb") as file:
                file.write(response.content)
        else:
            logger.info(f"Failed to download the image: {image_filename}")
            