#!/bin/env python3

'''
This script is used to load all process all of the XML text files
into a dictionary of {dataset split -> {file_code -> file_text}}. 
It uses the processing functions defined in read_xml
'''

import os
import re
import xml.etree.ElementTree as ET
import logging
from typing import *
from read_xml import get_xml_document, resolve_path

# TODO: Make sure documents consistent include or exlude places

Path = str

logging.basicConfig(
    level=logging.INFO, 
    format='(%(levelname)s|(%(asctime)s)) %(message)s',
    handlers=[
        logging.FileHandler("preprocess/preprocess.log")
    ]
)
logger = logging.getLogger()

def parse_metadata_file(file: Path) -> Dict[str, str]:
    '''Go through the metadata file and acquire a dictionary of file
        codes to text.
        Args:
            - file: the metadata .xml file of the test split 
    '''
    with open(file, 'r') as document:
        doc_as_string = document.read()
    documents_list = ET.fromstring(doc_as_string)
    doc_ids = documents_list.findall(".//doc")
    dictionary = {}
    for doc_id in doc_ids:
        id = doc_id.get("id")
        try:
            dictionary[id] = get_xml_document(id)
            # print(id, " ".join(dictionary[id].split()[0:10]), "...")
        except FileNotFoundError:
            logger.warning("Couldn't load file ID: {}. Issue with path: {}?".format(id, resolve_path(id)))
    return dictionary


def get_dataset_dict(
        root_directory: Path = '~/dropbox/22-23/575x/Data/Documents'
    ) -> Dict[str, Dict[Path, str]]:
    '''Obtain the data from the directory provided'''
    if re.match(r'^~', root_directory):
        root_directory = os.path.join(
            os.path.expanduser("~"), root_directory[2:]
        )
    dictionary = {}
    for (dirpath, _, fnames) in os.walk(root_directory):
        for file in fnames:
            if file.endswith(".xml"):
                train_test_split = os.path.basename(dirpath)
                file_path = os.path.join(dirpath, file)
                dictionary[train_test_split] = parse_metadata_file(file_path)
    return dictionary


if __name__ == '__main__':
    print("Reading files")
    get_dataset_dict()