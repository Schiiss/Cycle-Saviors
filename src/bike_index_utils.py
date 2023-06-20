"""Utilities to get bike index metadata"""
import requests
import json
from datetime import datetime

class BikeIndex:
    """Class to request Bike Index content"""

    def __init__(self, city):
        """Instantiate

        Args:
            city: location of api request
        """        
        self.city = city

    def call_bike_index_api(self, url):
        """_summary_

        Args:
            url (_type_): _description_

        Returns:
            _type_: _description_
        """
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            return None

    def clean_bike_index_response(self, data):
        """_summary_

        Args:
            data (_type_): _description_
        """
        for key, value in data.items():
            if not data[key]:
                data[key] = 'None'
        
        return data
            
    def create_bike_index_metadata(self, data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        list_date = int(datetime.fromtimestamp(data['date_stolen']).strftime('%Y%m%d'))

        metadata = {
            "source_id": str(data['id']),
            "list_date": list_date,
            "ad_location": self.city,
            "ad_link": data['url'],
            "image_link": data['thumb'],
            "ad_title": data['title'],
            "ad_description": data['description'],
            "brand": data['manufacturer_name'],
            "model": data['frame_model'],
            "color": data['frame_colors'],
            "source": 'Bike Index'
        }

        return metadata

    def save_bike_index_json(self, filepath, metadata_list):
        """_summary_

        Args:
            metadata_list (_type_): _description_
        """
        data = {"metadata": metadata_list}

        # Write the data to the JSON file
        with open(filepath, 'w') as json_file:
            json.dump(data, json_file)
