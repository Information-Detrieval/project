import os.path

import os
import pickle
import time
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from scrape import WebScraper
import pprint

class DataPipeline():
    def __init__(self, websites):
        self.pkl_dir = os.path.join(os.getcwd(), "website_data" , "pkl")
        self.OPENAI_API_KEY = "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"
        self.PINECONE_API_KEY = "8a73267f-d64d-4d53-a5ae-0a241afd5517"
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["PINECONE_API_KEY"] = self.PINECONE_API_KEY
        self.PERSIST_DIR = os.path.join(os.getcwd(), "storage")
        self.pinecone_index = self.initialize_pinecone()
        self.websites = websites

    def scrape_websites(self):
        return WebScraper(self.websites).scrape_websites()

    def initialize_pinecone(self):
        pc = Pinecone(api_key=self.PINECONE_API_KEY)
        try:
            index_name = "detrieval"

            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="gcp", region="us-central1"),
                )

            return pc.Index(index_name)

        except Exception as e:
            print(f"Pine Cone initialization failed: {e}")


    def initialize_index(self, documents, vector_store):
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=200)
        Settings.embed_model = embed_model
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_overlap=200,
                    chunk_size=1024,
                ),
                embed_model,
            ],
            vector_store=vector_store,
        )
        if os.path.exists(self.PERSIST_DIR):
            storage_context = StorageContext.from_defaults(
                persist_dir=self.PERSIST_DIR, vector_store=vector_store
            )
            index = load_index_from_storage(storage_context)
            return index
        else:
            pipeline.run(documents=documents)
            index = VectorStoreIndex.from_vector_store(vector_store)
            index.storage_context.persist(persist_dir=self.PERSIST_DIR)
            return index

    def run_query(self, query_str):
        llm = OpenAI(model="gpt-3.5-turbo-0125", api_key=self.OPENAI_API_KEY)
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        if os.path.exists(os.path.join(os.getcwd(), "website_data", "pkl", "documents.pkl")):
            with open(os.path.join(os.getcwd(), "website_data", "pkl", "documents.pkl"), "rb") as f:
                documents = pickle.load(f)
        else:
            documents = SimpleDirectoryReader(
                os.path.join(os.getcwd(), "website_data", "txt"),
                recursive=True,
            ).load_data()
            with open(os.path.join(os.getcwd(), "website_data", "pkl", "documents.pkl"), "wb") as f:
                documents = pickle.dump(documents, f)

        index = self.initialize_index(documents, vector_store)
        retriever = VectorIndexRetriever(index, similarity_top_k=3)
        retrieved_nodes = retriever.retrieve(query_str)
        return retrieved_nodes


if __name__ == "__main__":
    websites = ["https://www.iiitd.ac.in/dhruv"]
    temp = DataPipeline(websites)

    query_str = "What are the dining facilities in IIITD?"
    adi = temp.run_query(query_str)
    
    for i in adi:
        print(i)
        print(i.metadata)
    # print(type(adi[0]))
    print(len(adi))



# pinecone_index = self.initialize_pinecone()
# vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# documents = self.load_or_cache_documents("./cached_documents.pkl")
# documents = self.load_or_cache_documents("./cached_documents.pkl")
# documents = self.load_or_cache_documents("./cached_documents.pkl")
# documents = self.load_or_cache_documents("./cached_documents.pkl")
