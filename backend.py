from flask import Flask, request, jsonify
from pipeline import DataPipeline

app = Flask(__name__)
pipeline_text = DataPipeline("txt")
pipeline_img = DataPipeline("img")


@app.route('/scrape_websites', methods=['POST'])
def scrape_websites():
    websites = request.json['websites']
    result = pipeline_text.scrape_websites(websites)
    return jsonify(result)


@app.route('/scrape_sitemap', methods=['POST'])
def scrape_sitemap():
    sitemap = request.json['sitemap']

    result = pipeline_text.scrape_sitemap(sitemap)
    return jsonify(result)


@app.route('/run_query', methods=['POST'])
def run_query():
    if request.method == 'POST':
        query_str = request.get_json()
        print(query_str['query_str'])
        result = pipeline_text.run_query(query_str['query_str'])
        print(result)
        return jsonify({"data": str(result)})


@app.route('/')
def home():
    return "Welcome to DeRetrival Backend API!"


if __name__ == '__main__':
    app.run(debug=True)
