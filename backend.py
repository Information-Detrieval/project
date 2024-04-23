import json
import os
import pickle
from urllib.parse import urlparse

from flask import Flask, request, jsonify
from pipeline import DataPipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

pipeline_text = DataPipeline("txt")
pipeline_img = DataPipeline("img")

path = os.getcwd()


@app.route('/start_sitemap', methods=['POST'])
def start_sitemap():
    """
        Starts the sitemap creation and initializes the documents'
    """
    websites = request.get_json()['websites']
    print("in start_sitemap")
    # delete Scraping
    import shutil
    if os.path.exists("mapping.json"):
        mapping = json.load(open("mapping.json", "r"))
        for website in websites:
            print("Debug", mapping.keys(), urlparse(website).netloc )
            if urlparse(website).netloc not in mapping.keys():
                if os.path.isdir("storage"):
                    shutil.rmtree("storage")
                    os.remove("website_data/pkl/txt_documents.pkl")
                    files = os.listdir("website_data/meta_data")

                    # Iterate over each file and delete it
                    for file in files:
                        file_path = os.path.join("website_data/meta_data", file)
                        try:
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file: {file_path}, Error: {e}")


                    files = os.listdir("website_data/txt")

                    # Iterate over each file and delete it
                    for file in files:
                        file_path = os.path.join("website_data/txt", file)
                        try:
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file: {file_path}, Error: {e}")

                    files = os.listdir("website_data/sitemaps")

                    # Iterate over each file and delete it
                    for file in files:
                        file_path = os.path.join("website_data/sitemaps", file)
                        try:
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file: {file_path}, Error: {e}")

                # os.system("rm -rf storage/")
                # os.system("rm -rf website_data/pkl")
                # os.system("mkdir -p website_data/pkl")
                sitemap_path = pipeline_text.form_sitemap(websites)
                print(sitemap_path)
                pipeline_text.scrape_sitemap(sitemap_path)
                # iterate over the files of the sitemaps
                # for i in os.listdir(os.path.join(path, "website_data", "sitemaps")):
                #     pipeline_text.scrape_sitemap(i)
                # pipeline_img.initialize_documents("img_txt")
                pipeline_text.initialize_documents("txt")
    return jsonify({})


@app.route('/scrape_websites', methods=['POST'])
def scrape_websites():
    websites = request.get_json()['websites']
    print("Current website: ", websites)
    scrape_websites(websites)
    # pipeline_text.initialize_documents("txt")
    return jsonify({})


@app.route('/scrape_sitemap', methods=['POST'])
def scrape_sitemap():
    os.system("rm -rf storage/")
    os.system("rm -rf website_data/pkl")
    os.system("mkdir -p website_data/pkl")
    website = request.get_json()['website']
    result = pipeline_text.scrape_sitemap(website)
    return jsonify(result)


@app.route('/run_query', methods=['POST', ])
def run_query():
    if request.method == 'POST':
        query_str = request.get_json()
        print(query_str['query_str'])
        result = pipeline_text.run_query(query_str['query_str'])
        print(result)
        reply = None
        try:
            reply = result['result']
        except Exception as e:
            reply = result[0].metadata['html'] if 'html' in result[0].metadata else result[0].text
            url = result[0].metadata['url']

            reply = f"<a href='{url}'>{url}</a><br/><br/>{reply}"
        return jsonify({"data": reply})


@app.route('/')
def home():
    return "Welcome to DeRetrival Backend API!"


if __name__ == '__main__':
    app.run(port=8000, debug=True)
