# IR Project

Backend of the Information Detrevial

## Prerequisites

- Python 3.8 or higher
- pip

## Installation

1. Clone the repository Run: `git clone git@github.com:Information-Detrieval/project.git` 
2. Change directory to the project root directory `cd project`
3. Create a virtual environment
   - On Windows, run: `python -m venv env`
   - On Unix or MacOS, run: `python3 -m venv .venv`
   - This will create a virtual environment named `env`
4. Activate the virtual environment
   - On Windows, run: `.env\Scripts\activate`
   - On Unix or MacOS, run: `source .venv/bin/activate`
5. Install the required packages
   - Run: `pip install -r requirements.txt`
6. project Structure:
    - Website_data folder - consists of the extracted content from websites in form of 
        - html
        - json {"title":"", "url": "", "text": ""}
        - txt
        - pickle
7. To run the code with NEW Vectors and NEW data
    - Delete the storage folder
    - Delete the pkl/documents.pkl file
8. Add the new sitemap locaiton `scrape_sitemap("law.xml")` line of the code.

8. Run `python backend.py`. You would see a loading bar in the console that shows that the vectors have been upserterd. If not please redo the step 7.
