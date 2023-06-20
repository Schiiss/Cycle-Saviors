"""Upsert data to Pinecone index

Resources:
https://github.com/pinecone-io/examples/blob/master/search/hybrid-search/ecommerce-search/ecommerce-search.ipynb
https://huggingface.co/sentence-transformers/clip-ViT-B-32
https://docs.pinecone.io/docs/image-similarity-search
"""
import yaml
import structlog
from src.pinecone_utils import PineconeTools
import pinecone
from sentence_transformers import SentenceTransformer
import os

logger = structlog.getLogger(__name__)

# user defined configurations
with open("config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = config_data['pinecone']['index_name']
index = pinecone.Index(index_name)

embedding_model_name = config_data['pinecone']['embedding_model_name']
model = SentenceTransformer(embedding_model_name)

PineconeObj = PineconeTools()

for key, value in config_data['pinecone']['data'].items():
    logger.info(f'Loading data for marketplace: {key} from {value["json_data"]}')
    metadata = PineconeObj.load_metadata(value["json_data"])

    logger.info(f'Upserting batches of data to Pinecone')
    PineconeObj.batch_upsert(value["image_data"], metadata, index, model)
