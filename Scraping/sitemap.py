import requests
from xml.etree import ElementTree
from urllib.parse import urljoin

# Define the maximum depth of recursion
MAX_DEPTH = 1  # You can adjust this based on your needs

def fetch_sitemap_content(url):
    """
    Fetches the content of a sitemap URL.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        #print(f"Failed to fetch sitemap: {e}")
        return None

def parse_sitemap(content, base_url, depth=0):
    """
    Parses sitemap content to extract URLs. If the sitemap contains other sitemaps (.xml), it fetches them recursively.
    Stops recursion at a defined maximum depth to prevent infinite loops.
    """
    if depth > MAX_DEPTH:
        #print(f"Reached maximum recursion depth of {MAX_DEPTH}.")
        return []
    
    try:
        sitemap = ElementTree.fromstring(content)
        namespace = {'ns': sitemap.tag.split('}')[0].strip('{')}
        
        urls = []
        # Check if this is a sitemap index or a regular sitemap
        for loc_tag in sitemap.findall('.//ns:loc', namespaces=namespace):
            url = urljoin(base_url, loc_tag.text)
            if 'sitemapindex' in sitemap.tag or url.endswith('.xml'):
                # If this is another sitemap, fetch and parse it recursively
                sitemap_content = fetch_sitemap_content(url)
                if sitemap_content:
                    urls.extend(parse_sitemap(sitemap_content, base_url, depth + 1))
            else:
                # Otherwise, add the URL to the list
                urls.append(url)
        return urls
    except ElementTree.ParseError as e:
        print(f"Failed to parse XML: {e}")
        return []

def get_sitemap_urls(url):
    """
    Fetches and parses the sitemap of a given website, including handling sitemap indexes and nested sitemaps.
    """
    sitemap_url = url.rstrip('/') + '/sitemap.xml'
    sitemap_content = fetch_sitemap_content(sitemap_url)
    if sitemap_content:
        return parse_sitemap(sitemap_content, url)
    else:
        return []

website_url = 'https://www.latestlaws.com/'
sitemap_urls = get_sitemap_urls(website_url)
print(f"URLs found in the sitemap: {sitemap_urls}")