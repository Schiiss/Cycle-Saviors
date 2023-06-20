"""Utilities to  interact with Pinecone"""
import os
import json
from PIL import Image
from tqdm import tqdm
import structlog

logger = structlog.getLogger(__name__)

class PineconeTools:
    """Class to upsert data to Pinecone"""

    def __init__(self):
        """instantiate"""

    def load_metadata(self, filepath):
        """_summary_

        Args:
            data (_type_): _description_
        """
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)

        metadata = data['metadata']

        return metadata
    
    def batch_upsert(self, filepath, data, index, model):
        """_summary_

        Args:
            filepath (_type_): _description_
            data (_type_): _description_
        """
        i = 0
        upserts = []
        for idx in tqdm(range(0, len(data))):

            metadata = data[idx]
            image_filepath = os.path.join(filepath, str(metadata["source_id"]) + '.jpg')
            embedding = model.encode(Image.open(image_filepath)).tolist()

            upserts.append({
                'id': str(idx),
                'values': embedding,
                'metadata': metadata
            })    
            
            i += 1
            if i % 250 == 0:
                index.upsert(upserts)
                logger.info(f'Upsert count: {i}')
                upserts = []

        # this will index last few documents when i % 1000 != 0
        if len(upserts) > 0:
            index.upsert(upserts)
            logger.info('Upserting final batch')
