import datetime
import os.path

import os
import pickle
import time
import json
from typing import List
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET

import langchain
import requests
from bs4 import BeautifulSoup
from langchain import hub
from langchain.chains.combine_documents.stuff import StuffDocumentsChain, create_stuff_documents_chain
from langchain.chains.openai_functions import create_structured_output_chain, create_openai_fn_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    Settings,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
from pydantic.v1 import Field

from scrape import WebScraper
import pandas as pd
import shutil
from PIL import Image
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as langchainVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

langchain.verbose = True
langchain.debug = True

path = os.getcwd()


def scrape_websites(websites):
    scraped_data = WebScraper(websites).scrape_websites()
    return scraped_data


class DataPipeline():
    def __init__(self, name):
        self.name = name
        self.pkl_dir = os.path.join(path, "website_data", "pkl")
        self.OPENAI_API_KEY = "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"
        self.PINECONE_API_KEY = "8a73267f-d64d-4d53-a5ae-0a241afd5517"
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["PINECONE_API_KEY"] = self.PINECONE_API_KEY
        self.PERSIST_DIR = os.path.join(path, "storage", self.name)
        self.pinecone_index = self.initialize_pinecone()
        self.websites = ""
        self.index = None

    def form_sitemap(self, websites):
        unique_domains = list(set([urlparse(website).netloc for website in websites]))
        print("Unique domains:", unique_domains)
        mapping = {urlparse(website).netloc: website for website in websites}

        temp_json = json.load(open("mapping.json", "r"))
        mapping.update(temp_json)
        print("adi",mapping )
        with open("mapping.json", "w") as f:
            json.dump(mapping, f)

        mapping = json.load(open("mapping.json", "r"))
        print(mapping)

        # create the sitemap for each domain and url and store it in a xml file
        for domain in unique_domains:
            print(domain)
            sitemap = self.create_sitemap(mapping[domain], 1)

            # Create the root element
            urlset = ET.Element('urlset', {
                'xmlns': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                'xsi:schemaLocation': 'http://www.sitemaps.org/schemas/sitemap/0.9 '
                                      'http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd'
            })

            # Add each URL to the sitemap
            for url in sitemap:
                url_element = ET.SubElement(urlset, 'url')
                loc_element = ET.SubElement(url_element, 'loc')
                loc_element.text = url
                lastmod_element = ET.SubElement(url_element, 'lastmod')
                lastmod_element.text = datetime.datetime.now().isoformat()

            # Write the XML data to the file
            tree = ET.ElementTree(urlset)
            path_sitemap = os.path.join(path, "website_data", "sitemaps", domain.split(".")[0] + ".xml")
            print(path_sitemap)
            tree.write(path_sitemap, encoding='utf-8', xml_declaration=True)
            return path_sitemap

    def create_sitemap(self, base_url, depth) -> List[str]:
        # print(depth)
        if depth <= 0:
            return [base_url]

        sitemap = [base_url]
        try:
            response = requests.get(base_url)
            # print("page content", response)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)  # Ensure href attribute exists
            print('LINK COUNT:', len(links), 'DEPTH:', depth)
            for link in links:  # Limit the number of links to follow
                url = link['href']  # Use dictionary-like access for href
                if url.startswith('/'):
                    url = urlparse(base_url).netloc + url
                if url.startswith('http') or url.startswith('www'):  # Skip mailto or javascript links
                    # Resolve relative links
                    # url = urljoin(base_url, url)
                    # Check if the URL is still on the same domain
                    netloc1 = urlparse(url).netloc
                    netloc2 = urlparse(base_url).netloc
                    if netloc1.startswith("www."):
                        netloc1 = netloc1[4:]
                    if netloc2.startswith("www."):
                        netloc2 = netloc2[4:]
                    if netloc1 == netloc2:
                        print('Following:', url)
                        sitemap += self.create_sitemap(url, depth - 1)
                    else:
                        print(urlparse(url).netloc + ' != ' + urlparse(base_url).netloc)
                else:
                    print('Skipping:', url)
        except requests.exceptions.RequestException:
            pass

        # Remove duplicates
        # sitemap = list(set(sitemap))
        # Save sitemap to a file
        with open("sitemap.xml", 'w') as f:
            json.dump(sitemap, f)

        return sitemap

    def scrape_sitemap(self, sitemap):
        return WebScraper(self.websites).scraped_sitemap(sitemap)

    def initialize_pinecone(self):
        pc = Pinecone(api_key=self.PINECONE_API_KEY)
        try:
            index_name = self.name

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

    def initialize_documents(self, type_path):
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)

        def extract_metadata(filename):
            json_path = os.path.join(path, "website_data", "meta_data",
                                     str(os.path.basename(filename)).replace(".txt", ".json"))
            print(json_path)
            with open(json_path, "r") as f:
                metadata = json.load(f)
            return {"title": metadata.get("title", ""), "url": metadata.get("url", "")}

        filename_fn = extract_metadata

        if os.path.exists(os.path.join(path, "website_data", "pkl", str(type_path + "_documents.pkl"))):
            with open(os.path.join(path, "website_data", "pkl", str(type_path + "_documents.pkl")), "rb") as f:
                documents = pickle.load(f)
        else:
            documents = SimpleDirectoryReader(
                os.path.join(path, "website_data", str(type_path)),
                file_metadata=filename_fn,
                recursive=True,
            ).load_data()
            with open(os.path.join(path, "website_data", "pkl", str(type_path) + "_documents.pkl"), "wb") as f:
                pickle.dump(documents, f)
        self.index = self.initialize_index(documents, vector_store)

    def run_query(self, query_str):
        llm = OpenAI(model="gpt-3.5-turbo-0125", api_key=self.OPENAI_API_KEY)
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
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

        retriever = VectorIndexRetriever(self.index)
        retrieved_nodes = retriever.retrieve(query_str)
        return retrieved_nodes

    def invoke(self, query, convo):

        pipeline_img = DataPipeline("img")
        pipeline_img.initialize_documents("imgs_txt")
        img_output = pipeline_img.run_query(query)
        if len(img_output) > 0 and img_output[0].score < 0.55:
            img_output = []
        llm = ChatOpenAI(
            openai_api_key=self.OPENAI_API_KEY,
            model_name='gpt-3.5-turbo',
            temperature=0.0
        )

        retrieval_qa_chat_prompt = ChatPromptTemplate.from_template("""Answer the following question based on
        provided context only. Past user queries are also provided. Provide detailed answers and provide output in html, along with appropriate use 
        of headings, bold and italic text:
            
            <context>
            {context}
            </context>
            
            Question: {input}
            
            Past user queries:
            {convo}
            
            """ + ("" if len(img_output) == 0 else """
            Also, add the Image URL to the appropriate place in the output only if the provided Image Description is relevant to the provided question. Do not add the image if Image Description is not relevant to the question:
                Image Description: {img_desc}
                Image URL: {img_url}
            """))

        # combine_docs_chain = create_stuff_documents_chain(
        #     llm,
        #     retrieval_qa_chat_prompt
        # )

        # model_name = 'text-embedding-ada-002'
        # embed = OpenAIEmbeddings(
        #     model=model_name,
        #     openai_api_key=self.OPENAI_API_KEY
        # )

        # text_field = "url"  # the metadata field that contains our context

        # vector_store = PineconeVectorStore(self.pinecone_index, embed, text_field)

        # retriever = vector_store.as_retriever()
        # retriever = VectorIndexRetriever(self.index)
        # retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        chain = retrieval_qa_chat_prompt | llm

        context = self.run_query(query)

        if len(context) == 0:
            return {
                "answer": "No relevant documents found."
            }
        else:
            context = "\n----------------------------------------------------------\n".join(
                list(map(lambda x: json.dumps(
                    {"title": x.metadata['title'], "url": x.metadata['url'], "content": x.text},
                    indent=4
                ), context)))

        invocation_input = {"input": query, "context": context, "convo": convo}
        if len(img_output) > 0:
            invocation_input["img_url"] = img_output[0].metadata['url']
            invocation_input["img_desc"] = img_output[0].text

        reply = chain.invoke(invocation_input)
        return {"answer":reply.content}


def img_ir_pre(image_website):
    json_name = "images_info.json"
    with open(json_name, "r") as f:
        img_json_file = json.load(f)

    for entry in img_json_file:
        title = entry["title"]
        url = entry["url"]
        text = entry["text"]
        image_path = os.path.join(image_website, url)
        text_path = os.path.join(image_website, url[:len(url) - 4] + ".txt")

        if os.path.exists(image_path) and os.path.exists(text_path):
            shutil.copy(image_path, os.path.join("website_data/imgs", url))
            shutil.copy(text_path, os.path.join("website_data/img_txt", title + ".txt"))
            with open(os.path.join("website_data/img_json", title + ".json"), "w") as json_file:
                json.dump(entry, json_file, indent=4)


# For processing images
# image_website = "images_india"
# img_ir_pre(image_website)


if __name__ == "__main__":
    # websites = ["https://www.iiitd.ac.in/dhruv"]
    # df = pd.read_csv("Combined-QnA.csv")
    # img_db = DataPipeline(name="img")
    # # temp.scrape_websites(["https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-166a-punishment-for-non-recording-of-information-/"])
    # # temp.scrape_sitemap("news.xml")
    # # temp.scrape_sitemap("law.xml")
    # img_db.initialize_documents("img_txt")
    # obj = img_db.run_query(
    #     "I want to see a terracotta image")
    # for i in range(3):
    #     print(obj[i].metadata['url'])
    #     img = Image.open(os.path.join(path, "website_data", "imgs", str(obj[i].metadata['url'])))
    #     img = img.convert("RGB")
    #     img.show()

    text_db = DataPipeline("txt")
    text_db.initialize_documents("txt")
    print(text_db.generativeQnA(
        "What is the punishment for a public servant unlawfully buying or bidding for property under Section 169 of "
        "the IPC?"))
    new_rows = []

    # data = DataPipeline("txt")
    # data.initialize_documents("txt")
    # print(generativeQnA(data, "What is the punishment for a public servant unlawfully buying or bidding for property under Section 169 of the IPC"))

    # for index, row in df.iterrows():
    #     print(index)
    #     ground_truth_doc = row['Text File']
    #     query = row['Question']
    #     query_answer = row['Answer']
    #     retreived_docs = temp.run_query(query)
    #     print(retreived_docs[0])
    #     # print(retreived_docs[0].metadata['url'][57:])
    #     r_doc1 = retreived_docs[0].metadata['url']
    #     r_doc2 = retreived_docs[1].metadata['url']
    #     r_doc3 = retreived_docs[2].metadata['url']

    #     new_row = [query, ground_truth_doc, r_doc1, r_doc2, r_doc3]
    #     new_rows.append(new_row)
    #     # break

    # new_df = pd.DataFrame(new_rows, columns=['Question', 'Text_File', 'Retrieved_document_1', 'Retrieved_document_2', 'Retrieved_document_3'])
    # new_df.to_csv("Combined-QnR.csv", index=False)
