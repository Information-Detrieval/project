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


class DataPipeline:
    def __init__(self):
        self.pkl_dir = os.path.join(os.getcwd(), "website_data" , "pkl")
        self.OPENAI_API_KEY = "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"
        self.PINECONE_API_KEY = "8a73267f-d64d-4d53-a5ae-0a241afd5517"
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["PINECONE_API_KEY"] = self.PINECONE_API_KEY
        self.PERSIST_DIR = os.path.join(os.getcwd(), "storage")
        self.pinecone_index = self.initialize_pinecone()

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
        with open ("website_data\pkl\www.iiitd.ac.in_dhruv.txt", "rb") as f:
            documents = f.read()
        index = self.initialize_index(documents, vector_store)
        retriever = VectorIndexRetriever(index, similarity_top_k=10)
        retrieved_nodes = retriever.retrieve(query_str)
        return (retrieved_nodes)


if __name__ == "__main__":
    temp = DataPipeline()

    query_str = "Who is Dhruv Kumar"
    print(temp.run_query(query_str))



# pinecone_index = self.initialize_pinecone()
# vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# documents = self.load_or_cache_documents("./cached_documents.pkl")
# documents = self.load_or_cache_documents("./cached_documents.pkl")
# documents = self.load_or_cache_documents("./cached_documents.pkl")
# documents = self.load_or_cache_documents("./cached_documents.pkl")