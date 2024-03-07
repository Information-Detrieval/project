import requests
import pickle
from bs4 import BeautifulSoup
import os


class WebScraper:
    def __init__(self, websites):
        self.websites = websites

    def scrape_websites(self):
        try:
            for website in self.websites:
                r = requests.get(website)
                soup = BeautifulSoup(r.text, 'html.parser')
                data = soup.get_text()

                filename = website.split("://")[1].replace("/", "_")
                filepath = os.path.join(os.getcwd(),"website_data\html" , f"{filename}.html")

                with open(filepath, "w") as f:
                    # remove trainling spaces inbetwewn the code 
                    f.write(r.text)

                filepath = os.path.join(os.getcwd(),"website_data\pkl" , f"{filename}.pkl")

                with open(filepath, "wb") as f:
                    # remove trainling spaces and extra empty lines
                    data = data.replace("\n", " ")
                    pickle.dump(data, f)

        except Exception as e:
            print(e)

    def get_html(self, website):
        try:
            r = requests.get(website)
            filename = website.split("://")[1].replace("/", "_")
            filepath = os.path.join(os.getcwd(), f"{filename}.html")

            with open(filepath, "w") as f:
                f.write(r.text)
        except Exception as e:
            print(e)

    def get_data(self, website):
        try:
            r = requests.get(website)
            soup = BeautifulSoup(r.text, 'html.parser')
            data = soup.get_text()
            filename = website.split("://")[1].replace("/", "_")
            filepath = os.path.join(os.getcwd(), f"{filename}.pkl")

            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(e)

    def get_mapping(self):
        mapping = {}
        for website in self.websites:
            filename = website.split("://")[1].replace("/", "_")
            mapping[website] = f"{filename}.pkl"
        with open("mapping.pkl", "wb") as f:
            pickle.dump(mapping, f)

    

if __name__ == "__main__":
    websites = [
        "https://www.iiitd.ac.in/dhruv"
    ]
    scraper = WebScraper(websites)
    scraper.scrape_websites()
    scraper.get_mapping()

    # load the pickle and print the data inside it 
    with open("mapping.pkl", "rb") as f:
        mapping = pickle.load(f)

    for website, filename in mapping.items():
        scraper.get_data(website)
        print(f"Data for {website} saved to {filename}")
        
    with open ("www.iiitd.ac.in_dhruv.pkl", "rb") as f:
        data = pickle.load(f)
        print(data)


