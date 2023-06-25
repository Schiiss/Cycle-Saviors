"""Extract data from online marketplaces

Resources:
https://bikeindex.org/documentation/api_v3
"""
from tqdm import tqdm
import yaml
import structlog
from src.bike_index_utils import BikeIndex
import os
from src.io_utils import download_image_card
from tqdm import tqdm

logger = structlog.getLogger(__name__)

# user defined configurations
with open("config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

#######################################################################
# BIKE INDEX DATA
#######################################################################
cities = config_data['bike-index']['cities']
max_pages = config_data['bike-index']['max_pages']
base_url = config_data['bike-index']['base_url']

metadata_list = []
for city in cities:
    logger.info(f"Requesting Bike-Index data for location: {city}")

    BikeIndexObj = BikeIndex(city)

    for page_num in tqdm(range(1, max_pages)):
        url = base_url.format(page_num, city) 

        response = BikeIndexObj.call_bike_index_api(url)

        for data in response['bikes']:
            metadata = {}

            # if an image thumbnail exists then parse
            if data['thumb']:

                try:
                    data = BikeIndexObj.clean_bike_index_response(data)
                    metadata = BikeIndexObj.create_bike_index_metadata(data)
                    metadata_list.append(metadata)

                    image_filepath = os.path.join(config_data['bike-index']['output_images'], str(data["id"]) + '.jpg')
                    download_image_card(image_filepath, data['thumb'])

                except:
                    print(f'Could not parse {data["id"]}')

BikeIndexObj.save_bike_index_json(config_data['bike-index']['output_json'], metadata_list)
