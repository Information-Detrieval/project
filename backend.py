import os

from flask import Flask, request, jsonify
from pipeline import DataPipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

pipeline_text = DataPipeline("txt")
pipeline_img = DataPipeline("img")

path = os.getcwd()


@app.route('/scrape_websites', methods=['POST'])
def scrape_websites():
    websites = request.get_json()['websites']
    # delete Scraping/
    # os.system("rm -rf storage/")
    # os.system("rm -rf website_data/pkl")
    # os.system("mkdir -p website_data/pkl")
    # result = pipeline_text.scrape_websites(websites)
    # iterate over the files of the sitemaps
    # for i in os.listdir(os.path.join(path, "website_data", "sitemaps")):
    #     pipeline_text.scrape_sitemap(i)
    # pipeline_img.initialize_documents("img_txt")
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
