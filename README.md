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

8. Run `python backend.py`. You might see a loading bar in the console that shows that the vectors have been upserted.

9. Goto `chrome://extensions/` in your browser and enable developer mode. Click on `Load unpacked` and select the `extension` repo (in the same Github org as this repo).

10. Open a new tab and click on the extension icon. You should see the extension popup alongwith the chat UI.
