"""Utilities to help with writing and reading"""
import requests
import structlog

logger = structlog.getLogger(__name__)

def download_image_card(image_filename, image_link):
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
        