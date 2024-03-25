import re
from lxml import etree

import requests
import pickle
from bs4 import BeautifulSoup
import os

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)
import xml.etree.ElementTree as ET


class WebScraper:
    def __init__(self, websites):
        self.websites = websites
        self.html_dir = os.path.join(os.getcwd(), "website_data", "html")
        self.pkl_dir = os.path.join(os.getcwd(), "website_data", "txt")
        self.mapping_file = "mapping.pkl"

        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.pkl_dir, exist_ok=True)

    def _get_filename(self, website):
        return website.split("://")[1].replace("/", "_")

    def _write_to_file(self, filepath, content):
        with open(filepath,  "w", encoding="utf-8") as f:
            f.write(content)

    def scrape_websites(self):
        mapping = {}
        for website in self.websites:
            print(f"Scraping {website}...")
            try:
                r = requests.get(website)
                soup = BeautifulSoup(r.text, 'html.parser')
                data = soup.get_text()
                dom = etree.HTML(str(soup))

                filename = self._get_filename(website)
                html_filepath = os.path.join(self.html_dir, f"{filename}.html")
                pkl_filepath = os.path.join(self.pkl_dir, f"{filename}.txt")

                # self._write_to_file(pkl_filepath+".txt", data)
                # remove repeated blank lines, but retain single new lines
                txt = re.sub('\n{2,}', '\n', data)
                if website.startswith("https://www.latestlaws.com"):
                    x_path = '//*[@id="content-area"]/div/div/div[2]/div[2]/div[1]/div[3]/p'
                    txt = dom.xpath(x_path)[0].text

                self._write_to_file(html_filepath, r.text)
                self._write_to_file(pkl_filepath, txt)

                mapping[website] = f"{filename}.pkl"

            except Exception as e:
                print(f"Error scraping {website}: {e}")

        # with open(self.mapping_file, "wb") as f:
        #     pickle.dump(mapping, f)

        return self.html_dir, self.pkl_dir


    def get_html(self, website):
        filename = self._get_filename(website)
        filepath = os.path.join(self.html_dir, f"{filename}.html")
        try:
            with open(filepath, "r") as f:
                html_content = f.read()
                print(f"HTML for {website}:\n{html_content}")
        except FileNotFoundError:
            print(f"HTML file for {website} not found.")

    def get_data(self, website):
        filename = self._get_filename(website)
        filepath = os.path.join(self.pkl_dir, f"{filename}.pkl")
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                print(f"Data for {website}:\n{data}")
        except FileNotFoundError:
            print(f"Data file for {website} not found.")

    def get_mapping(self):
        try:
            with open(self.mapping_file, "rb") as f:
                mapping = pickle.load(f)
                print("Mapping:")
                for website, filename in mapping.items():
                    print(f"{website} -> {filename}")
        except FileNotFoundError:
            print("Mapping file not found.")

    # extract data from XML sitemap and call scrape

    def scraped_sitemap(self, sitemap_file):
        try:
            with open(sitemap_file, "r", encoding="utf-8") as f:
                sitemap_xml = f.read()
                root = ET.fromstring(sitemap_xml)

                # Define the namespace
                ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

                # Use the namespace in the XPath expression
                urls = [url.text for url in root.findall(".//ns:loc", namespaces=ns)]

                self.websites = urls
                self.scrape_websites()

        except FileNotFoundError:
            print(f"Sitemap file not found: {sitemap_file}")




if __name__ == "__main__":
    websites = ["https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-170-personating-a-public-servant/"]
    scraper = WebScraper(websites)
    scraper.scrape_websites()
    # scraper.scraped_sitemap("sitemap.xml")

    # with open("website_data/pkl/iiitd_ac_in_dhruv.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     print(data)
