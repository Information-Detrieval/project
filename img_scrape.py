import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from urllib.parse import urljoin, unquote
from io import BytesIO
import re
import json

# The URL of the webpage you want to scrape
url = 'https://en.wikipedia.org/wiki/India'


def call(url):
    # Folder where the images and their alt texts will be stored
    parent_folder = 'website_data'
    images_folder = os.path.join(parent_folder, 'imgs')
    images_text_folder = os.path.join(parent_folder, 'imgs_txt')
    images_json_folder = os.path.join(parent_folder, 'meta_data')

    os.makedirs(parent_folder, exist_ok=True)

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(images_text_folder, exist_ok=True)
    os.makedirs(images_json_folder, exist_ok=True)

    # Use requests to fetch the content of the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Create a Markdown file to store the scraped text
    with open('Scraping/scraped_content_india.md', 'w', encoding='utf-8') as md_file:
        for element in soup.find_all(['p', 'h1', 'h2', 'h3']):
            # Assuming you want to capture paragraphs and headers
            if element.name.startswith('h'):
                md_file.write(f"\n# {element.get_text()}\n\n")  # Header with markdown format
            else:
                md_file.write(f"{element.get_text()}\n\n")  # Paragraph

    #to download the image and store the alt text/related text of the image

    titles_set = set()
    images_info = []
    for img_tag in soup.find_all('img'):

        img_url = urljoin(url, img_tag['src'])
        img_response = requests.get(img_url)

        # Get the alt text
        alt_text = img_tag.get('alt')
        if not alt_text:  # Checks if alt_text is None or empty
            alt_text = " "
            # Use the 'src' attribute as the fallback
            # src_text = img_tag.get('src')
            # srcset_text = img_tag.get('srcset')
            # if not src_text:
            #     alt_text = srcset_text
            #     if not srcset_text:
            #         alt_text = " "
            # else:
            #     alt_text = src_text
            # # if (srcset_text and src_text):
            # #     alt_text = src_text + " " + srcset_text
            # print(alt_text)

        title = img_tag.get('title')
        if not title:
            image_name = os.path.basename(img_url)
            clean_title = re.sub(r'\d+px', '', unquote(os.path.splitext(image_name)[0]))
            title = clean_title.replace('_', ' ').replace('-', ' ').strip()
            # If the title ends with '.svg', remove it
            if title.endswith('.svg'):
                title = title[:-4]
        else:
            image_name = os.path.basename(img_url)

        #print(title)
        # Ensure title uniqueness
        original_title = title
        count = 1
        while title in titles_set:
            count += 1
            title = f"{original_title} ({count})"
        titles_set.add(title)

        image_name = os.path.basename(img_url)

        data = {
            "title": title,
            "url": img_url,
            # "full_url": img_url,
            "text": alt_text
        }

        images_info.append(data)

        image_path = os.path.join(images_folder, image_name)
        with open(image_path, 'wb') as img_file:
            img_file.write(img_response.content)

        #saving the alt text in a separate file with the same name but .txt extension
        alt_text_path = os.path.join(images_text_folder, f"{os.path.splitext(image_name)[0]}.txt")
        with open(alt_text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(alt_text+" "+title.split(".")[0])

        image_info_path = os.path.join(images_json_folder, f"{os.path.splitext(image_name)[0]}.json")
        with open(image_info_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    with open('Scraping/images_info.json', 'w', encoding='utf-8') as json_file:
        json.dump(images_info, json_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    call(url)
    print("Scraping done")
