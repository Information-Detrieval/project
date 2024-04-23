import requests
from xml.etree import ElementTree as ET
from urllib.parse import urljoin

MAX_DEPTH = 1


def fetch_sitemap_content(url):
    """
    Fetches the content of a sitemap URL.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        return None


def add_sitemap_or_url(parent_element, url, is_sitemap=False):
    """
    Adds a sitemap or URL element to the parent element.
    """
    if is_sitemap:
        sitemap_element = ET.SubElement(parent_element, 'sitemap')
        ET.SubElement(sitemap_element, 'loc').text = url
    else:
        url_element = ET.SubElement(parent_element, 'url')
        ET.SubElement(url_element, 'loc').text = url


def parse_sitemap_to_xml(content, base_url, parent_element, depth=0):
    """
    Parses sitemap content and constructs an XML structure with URLs.
    """
    if depth > MAX_DEPTH:
        return

    try:
        sitemap = ET.fromstring(content)
        namespace = {'ns': sitemap.tag.split('}')[0].strip('{')}

        # Check if this is a sitemap index or a regular sitemap
        for loc_tag in sitemap.findall('.//ns:loc', namespaces=namespace):
            url = urljoin(base_url, loc_tag.text)
            if 'sitemapindex' in sitemap.tag or url.endswith('.xml'):
                # If another sitemap, fetch and parse recursively
                sitemap_content = fetch_sitemap_content(url)
                if sitemap_content:
                    add_sitemap_or_url(parent_element, url, is_sitemap=True)
                    child_element = ET.SubElement(parent_element, 'sitemap')
                    parse_sitemap_to_xml(sitemap_content, base_url, child_element, depth + 1)
            else:
                # Add URL to the XML structure
                add_sitemap_or_url(parent_element, url)
    except ET.ParseError as e:
        print(f"Failed to parse XML: {e}")


def get_sitemap_as_xml(url):
    """
    Fetches and parses the sitemap of a given website into an XML structure.
    """
    root = ET.Element('sitemapindex', xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    sitemap_url = url.rstrip('/') + '/sitemap.xml'
    sitemap_content = fetch_sitemap_content(sitemap_url)
    if sitemap_content:
        parse_sitemap_to_xml(sitemap_content, url, root)
        # Format the XML string
        return ET.tostring(root, encoding='unicode', method='xml')
    else:
        return ''


if __name__ == "__main__":
    website_url = 'https://www.latestlaws.com/'

    sitemap_url = website_url.rstrip('/') + '/sitemap.xml'
    print(requests.get(sitemap_url).content)  #REAL SITEMAP OF THE GIVEN WEBSITE

    sitemap_xml = get_sitemap_as_xml(website_url)
    print(f"XML representation of the sitemap:\n{sitemap_xml}")  #PARSED AND CREATED
