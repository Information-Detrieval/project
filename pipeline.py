from pinecone import Pinecone, ServerlessSpec
import asyncio
from pipeline import Pipeline
from scrape import WebScraper
import os
import pickle

class DataPipeline(Pipeline):
    def __init__(self, websites):
        self.websites = websites
        self.initializePinecone()
    
    def initializePinecone(self):
        api_key = self.PINECONE_API_KEY
        pc = Pinecone(api_key=api_key)
        try:
            if self.PINECONE_INDEX_NAME not in pc.list_indexes().names():
                pc.create_index(
                    name=self.PINECONE_INDEX_NAME,
                    dimension=1536,
                    metric="euclidean",
                    spec=ServerlessSpec(cloud="gcp", region="us-central1"),
                )
            pinecone_index = pc.Index(self.PINECONE_INDEX_NAME)
            print("Pine Cone initialized successfully")
            return pinecone_index
        except:
            print("Pine Cone initialization failed")
        

    async def web_scraping(self):
        web_scraper = WebScraper(self.websites)
        web_scraper.scrape_websites()
        return web_scraper.websites

    def run(self):
        pipeline = DataPipeline(self.websites)
        asyncio.run(pipeline.run(
            pipeline.web_scraping(),
            pipeline.store_data_in_pinecone_db(pipeline.websites)
        ))


if __name__ == "__main__":
    websites = [
        "https://www.iiitd.ac.in/dhruv",
        # "https://iiitd.ac.in/",
        # "https://www.geeksforgeeks.org/",
        # "https://www.google.com/",
    ]

    DataPipeline(websites).run()
