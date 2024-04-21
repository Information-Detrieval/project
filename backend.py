from flask import Flask, request, jsonify
from pipeline import DataPipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

pipeline = DataPipeline()


@app.route('/scrape_websites', methods=['POST'])
def scrape_websites():
    websites = request.get_json()['websites']
    result = pipeline.scrape_websites(websites)
    return jsonify({})


@app.route('/scrape_sitemap', methods=['POST'])
def scrape_sitemap():
    sitemap = request.json['sitemap']
    result = pipeline.scrape_sitemap(sitemap)
    return jsonify(result)


@app.route('/run_query', methods=['POST', ])
def run_query():
    if request.method == 'POST':
        query_str = request.get_json()
        print(query_str['query_str'])
        result = pipeline.run_query(query_str['query_str'])
        print(result)
        reply = result[0].metadata['html']
        url = result[0].metadata['url']

        reply = f"<a href='{url}'>{url}</a><br/><br/>{reply}"
        return jsonify({"data": reply})


@app.route('/')
def home():
    return "Welcome to DeRetrival Backend API!"



if __name__ == '__main__':
    app.run(port=8000, debug=True)

