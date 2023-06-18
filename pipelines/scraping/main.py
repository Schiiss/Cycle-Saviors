"""Run scraping functions and store images and metadata"""
import requests
from tqdm import tqdm
import yaml
import structlog
import pandas as pd
from src.kijiji_utils import KijijiScrape
from bs4 import BeautifulSoup
from pathlib import Path
import os

logger = structlog.getLogger(__name__)

# user defined configurations for scraping kijiji
with open("config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

cities = config_data['kijiji']['cities']
max_pages = config_data['kijiji']['max_pages']
base_url = config_data['kijiji']['base_url']

# empty list to append dataframe of scraped information
df_list = []
for city, value in cities.items():
    for category, category_code in value.items():
        logger.info(f"Scraping data for category: {category} in location: {city}")

        for page_num in tqdm(range(0, max_pages)):
            url = base_url.format(category, city, page_num, category_code) 
            response = requests.get(url)

            if response.status_code == 200:
                html_content = response.content
                try:
                    soup = BeautifulSoup(html_content, "html.parser")

                    KijijiObj = KijijiScrape(soup)

                    image_urls = KijijiObj.extract_image_cards_url()
                    ad_urls, ad_ids, ad_titles = KijijiObj.extract_ad_url()
                    ad_summaries = KijijiObj.extract_ad_summary()
                    ad_categories = [category] * len(ad_ids)
                    ad_locations = [city] * len(ad_ids)

                    df = pd.DataFrame([ad_ids, ad_locations, ad_categories, ad_urls, ad_titles, ad_summaries, image_urls]).T
                    df.columns = ['ad_id', 'ad_location', 'ad_category', 'ad_link', 'ad_title', 'ad_description', 'image_link']
                    df_list.append(df)

                except:
                    logger.info(f'Error: Could not retrieve content from {url}')

            else:
                logger.info("Error: Failed to retrieve the web page.")

# create dataframe of information to export
concatenated_df = pd.concat(df_list, ignore_index=True)
concatenated_df.to_csv(Path(config_data['kijiji']['output_csv']))

# use extracted image links to download images
if not os.path.exists('../../data/images'):
    os.makedirs('../../data/images')
else:
    pass

base_image_path = config_data['kijiji']['output_images']
logger.info('Downloading images from image card links')
for i in tqdm(range(len(concatenated_df))):
    row = concatenated_df.iloc[i]
    image_filename = base_image_path.format(row['ad_id'])
    try:
        KijijiObj.download_image_card(image_filename, row['image_link'])
    except:
        logger.info(f"Could not download image: {row['image_link']}")
