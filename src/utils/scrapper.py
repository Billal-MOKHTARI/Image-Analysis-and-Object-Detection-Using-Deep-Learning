import pandas as pd
import numpy as np
from lxml import html
import requests
from bs4 import BeautifulSoup

from PIL import Image
from PIL.ExifTags import TAGS
import subprocess

def pic2map_scrapper(URL, save_path = None, **kwargs):

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
            
        

def get_exif_data(folder_path, save_path, metadata_extractor="./src/scripts/extract_metadata.sh"):
    # Execute the shell script to extract metadata
    try:
        subprocess.run(['pwd'], check=True)
        subprocess.run (["chmod", "+x", metadata_extractor], check=True)
        subprocess.run([metadata_extractor, folder_path, save_path], check=True)
        print("Metadata extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Metadata extraction failed with exit code {e.returncode}.")

get_exif_data("data/test", "data/output/metadata/image_metadata.csv")