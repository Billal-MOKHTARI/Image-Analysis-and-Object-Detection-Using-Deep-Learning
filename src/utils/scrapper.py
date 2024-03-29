import pandas as pd
import numpy as np
from lxml import html
import requests
from bs4 import BeautifulSoup

def image_metadata_scrapper(URL, save_path = None, **kwargs):

    index = kwargs.get("index", True)

    def dict_to_dataframe(data):
        # Initialize a list to store dictionaries for each row
        rows = []
        
        # Find all unique keys across all records
        all_keys = set()
        for record in data.values():
            all_keys.update(record.keys())
        
        # Create a dictionary for each row
        for key, record in data.items():
            row = {}
            for nested_key in all_keys:
                row[nested_key] = record.get(nested_key, None)
            rows.append(row)
        
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(rows, index=data.keys())
        
        return df

    main_page = requests.get(URL)
    main_tree = html.fromstring(main_page.content)

    main_page_content = main_page.content
    soup = BeautifulSoup(main_page_content, 'html.parser')

    # Find all 'a' tags within 'div' tags having class 'dbox'
    links = soup.select('div.dbox a[href]')

    # Extract href attribute from each 'a' tag
    sublinks = [link['href'] for link in links]

    # Create a dictionary of records
    data = dict()

    for sublink in sublinks:
        sub_page = requests.get(sublink)
        sub_tree = html.fromstring(sub_page.content)

        sub_page_content = sub_page.content
        soup = BeautifulSoup(sub_page_content, 'html.parser')

        metadata = soup.select("ul.details")[:-1]

        keys = list()
        values = list()
        for md in metadata:
            soup = BeautifulSoup(f"<html>{md}<\html>", 'html.parser')
            keys.extend([e.get_text().replace(':', '', 1) for e in soup.select("span.dtab")])
            values.extend([e.get_text() for e in soup.select("span.dvalue")])

        arg = keys.index("File Name")
        filename = values[arg]

        # Remove the file name from the lists
        keys.pop(arg)
        values.pop(arg)

        doc = dict(zip(keys, values))
        data[filename] = doc

    data = dict_to_dataframe(data)

    if save_path is not None:
        data.to_csv(save_path, index = index)

    return data
            
        


# URL = "https://www.pic2map.com/photos-sgrumn.html"
# image_metadata_scrapper(URL, "/workspaces/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/image_metadata.csv")

