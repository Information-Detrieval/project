import os.path

import os
import pickle
import time
import json
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
import pandas as pd

path = os.getcwd()

class DataPipeline():
    def __init__(self):
        self.pkl_dir = os.path.join(path, "website_data" , "pkl")
        self.OPENAI_API_KEY = "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"
        self.PINECONE_API_KEY = "8a73267f-d64d-4d53-a5ae-0a241afd5517"
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["PINECONE_API_KEY"] = self.PINECONE_API_KEY
        self.PERSIST_DIR = os.path.join(path, "storage")
        self.pinecone_index = self.initialize_pinecone()
        self.websites = ""

    def scrape_websites(self, websites):
        return WebScraper(websites).scrape_websites()

    def scrape_sitemap(self, sitemap):
        return WebScraper(self.websites).scraped_sitemap(sitemap)

    def initialize_pinecone(self):
        pc = Pinecone(api_key=self.PINECONE_API_KEY)
        try:
            index_name = "detrieval"

            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="gcp-starter", region="Iowa (us-central1)"),
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
        def extract_metadata(filename):
            with open(os.path.join(path, "website_data", "json", filename), "r") as f:
                metadata = json.load(f)
            return {"title": metadata.get("title", ""), "url": metadata.get("url", "")}
        
        filename_fn = extract_metadata
        llm = OpenAI(model="gpt-3.5-turbo-0125", api_key=self.OPENAI_API_KEY)
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        if os.path.exists(os.path.join(path, "website_data", "pkl", "documents.pkl")):
            with open(os.path.join(path, "website_data", "pkl", "documents.pkl"), "rb") as f:
                documents = pickle.load(f)
        else:
            documents = SimpleDirectoryReader(
                os.path.join(path, "website_data", "json"),
                file_metadata=filename_fn,
                recursive=True,
            ).load_data()
            with open(os.path.join(path, "website_data", "pkl", "documents.pkl"), "wb") as f:
                pickle.dump(documents, f)

        # # Generate vectors for each document and add them to Pinecone along with metadata
        # for doc in documents:
        #     print(doc)
        #     text = doc.get("text", "")
        #     metadata = {"title": doc["title"], "url": doc["url"]}
        #     vector = llm.generate_vecs(text)
        #     # vector_store.upsert(items=[vector], ids=[doc["url"]], metadata=[metadata])
        #     self.pinecone_index.upsert(vectors=[{
        #         "id": doc["url"],
        #         "values": vector,
        #         "metadata": metadata
        #     }])

        index = self.initialize_index(documents, vector_store)
        retriever = VectorIndexRetriever(index, similarity_top_k=3)
        retrieved_nodes = retriever.retrieve(query_str)
        return retrieved_nodes



if __name__ == "__main__":
    websites = ["https://www.iiitd.ac.in/dhruv"]
    df = pd.read_csv("QnA-Website2.csv")
    temp = DataPipeline()
    # temp.scrape_sitemap("law.xml")
    temp.run_query("What is the capital of India?")
    # new_rows = []

    # for index, row in df.iterrows():
    #     print(index)
    #     ground_truth_doc = row['Text File']
    #     query = row['Question']
    #     query_answer = row['Answer']
    #     retreived_docs = temp.run_query(query)
    #     r_doc1 = retreived_docs[0].metadata['file_path'][57:]
    #     r_doc2 = retreived_docs[1].metadata['file_path'][57:]
    #     r_doc3 = retreived_docs[2].metadata['file_path'][57:]

    #     new_row = [query, ground_truth_doc, r_doc1, r_doc2, r_doc3]
    #     new_rows.append(new_row)
    #     # break

    # new_df = pd.DataFrame(new_rows, columns=['Question', 'Text_File', 'Retrieved_document_1', 'Retrieved_document_2', 'Retrieved_document_3'])
    # new_df.to_csv("QnR-Taj.csv", index=False)





# pinecone_index = self.initialize_pinecone()
# vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# documents = self.load_or_cache_documents("./cached_documents.pkl")
# documents = self.load_or_cache_documents("./cached_documents.pkl")
# documents = self.load_or_cache_documents("./cached_documents.pkl")
# documents = self.load_or_cache_documents("./cached_documents.pkl")
