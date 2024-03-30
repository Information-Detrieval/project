import json
import re
import ssl

import nltk
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

from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


def preprocess_text(text: str) -> str:
    text = text.lower()

    tokens = word_tokenize(text)

    # NOTE: Assuming all non-alphanumeric characters are punctuation
    tokens = [word for word in tokens if word.isalnum()]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# NLTK MacOS SSL error fix
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')


class WebScraper:
    def __init__(self, websites):
        self.websites = websites
        self.html_dir = os.path.join(os.getcwd(), "website_data", "html")
        self.pkl_dir = os.path.join(os.getcwd(), "website_data", "txt")
        self.json_dir = os.path.join(os.getcwd(), "website_data", "json")
        self.mapping_file = "mapping.pkl"

        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.pkl_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

    def _get_filename(self, website):
        return website.split("://")[1].replace("/", "_")

    def _write_to_file(self, filepath, content):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def scrape_websites(self):
        mapping = {}
        for website in self.websites:
            print(f"Scraping {website}...")
            try:
                r = requests.get(website)
                raw_html = r.text

                soup = BeautifulSoup(raw_html, 'html.parser')
                title = preprocess_text(soup.title.string)

                if website.startswith("https://www.latestlaws.com"):
                    selector = "#content-area > div > div > div.col-md-6.order-1.order-sm-1.order-md-2 > div:nth-child(4) > div:nth-child(1) > div.page-content.actdetail.act-single-page"
                    soup = soup.select(selector)[0]

                data = soup.get_text()
                filename = self._get_filename(website)
                html_filepath = os.path.join(self.html_dir, f"{filename}.html")
                pkl_filepath = os.path.join(self.pkl_dir, f"{filename}.txt")
                json_filepath = os.path.join(self.json_dir, f"{filename}.json")

                # self._write_to_file(pkl_filepath+".txt", data)
                # remove repeated blank lines, but retain single new lines
                txt = re.sub('\n{2,}', '\n', data)

                self._write_to_file(html_filepath, r.text)
                self._write_to_file(pkl_filepath, txt)
                self._write_to_file(json_filepath, json.dumps({
                    "title": title,
                    "url": website,
                    "text": txt
                }, indent=4))

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
    websites = [
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-166a-punishment-for-non-recording-of-information-/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-169-public-servant-unlawfully-buying-or-bidding-for-property/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-167-public-servant-farming-an-incorrect-document-with-intent-to-cause-injury/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-131-abetting-mutiny-or-attempting-to-seduce-a-soldier-sailor-or-airman-from-his-duty/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-132-abetment-of-mutiny-if-mutiny-is-committed-in-consequence-thereof/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-133-abetment-of-assault-by-soldier-sailor-or-airman-on-his-superior-officer-when-in-execution-of-his-office/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-134-abetment-of-such-assault-if-the-assault-is-committed/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-135-abetment-of-desertion-of-soldier-sailor-or-airman/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-136-harbouring-deserter/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-137-deserter-concealed-on-board-merchant-vessel-through-negligence-of-master/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-138-abetment-of-act-of-insubordination-by-soldier-sailor-or-airman/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-139-persons-subject-to-certain-acts/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-140-wearing-garb-or-carrying-token-used-by-soldier-sailor-or-airman/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-141-unlawful-assembly/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-142-being-member-of-unlawful-assembly/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-143-punishment/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-144-joining-unlawful-assembly-armed-with-deadly-weapon/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-145-joining-or-continuing-in-unlawful-assembly-knowing-it-has-been-commanded-to-disperse/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-146-rioting/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-147-punishment-for-rioting/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-148-rioting-armed-with-deadly-weapon/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-149-every-member-of-unlawful-assembly-guilty-of-offence-committed-in-prosecution-of-common-object/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-150-hiring-or-conniving-at-hiring-of-persons-to-join-unlawful-assembly/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-151-knowingly-joining-or-continuing-in-assembly-of-five-or-more-persons-after-it-has-been-commanded-to-disperse/",
        "https://www.latestlaws.com/bare-acts/central-acts-rules/ipc-section-152-assaulting-or-obstructing-public-servant-when-suppressing-riot-etc-/",
    ]
    scraper = WebScraper(websites)
    scraper.scrape_websites()
    # scraper.scraped_sitemap("sitemap.xml")

    # with open("website_data/pkl/iiitd_ac_in_dhruv.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     print(data)
