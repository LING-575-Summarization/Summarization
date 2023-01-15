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

Path = str

logging.basicConfig(
    level=logging.INFO, 
    datefmt="%m/%d %H:%M:%S"
)
logger = logging.getLogger()


def parse_metadata_file(file: Path, docset: str) -> Dict[str, Dict[str, str]]:
    '''
    Go through the metadata file and acquire a dictionary of file codes to text.
    Args:
        - file: the metadata .xml file of the test split 
    '''
    with open(file, 'r') as document:
        doc_as_string = document.read()
    documents_list = ET.fromstring(doc_as_string)
    doc_ids = documents_list.findall(".//{}".format(docset))
    dictionary = {}   
    errors = 0    
    for doc_id in doc_ids:
        id_for_list = doc_id.get("id")
        dictionary[id_for_list] = {}
        print(doc_id.tag, id_for_list)
        for document in doc_id.findall("doc"):
            id = document.get("id")
            try:
                dictionary[id_for_list][id] = get_xml_document(id)
                print("\t", id_for_list, id, (" ".join(dictionary[id_for_list][id].replace("\n", " ").split()[0:10])), "...")
            except FileNotFoundError:
                logger.warning(
                    "Couldn't load file ID: {}. Issue with path: {}?".format(id, resolve_path(id))
                )
                errors += 1
    if errors == 0:
        logger.info("Done with {}! Good news! No errors found!".format(file))
    else:
        logger.info("Done with {}! Uh oh! Ran into {} errors...".format(file, str(errors)))
    return dictionary


def read_xml_files_from_directory(
        root_directory: Path = '~/dropbox/22-23/575x/Data/Documents',
        docset: str = "docsetA"
    ) -> Dict[str, Dict[str, Dict[Path, str]]]:
    '''
    Obtain the data from the directory provided containing datasets splits.
    Arguments:
        - root_directory: Path to the directory containing train/test/dev splits.
    '''
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
                dictionary[train_test_split] = parse_metadata_file(file_path, docset)
    return dictionary


def write_outputs(dictionary: Dict[str, Dict[Path, str]], output_dir: Path):
    '''
    Unravel the dictionary output and create directories with files for each document
    in a docset
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in dictionary:
        for docset in dictionary[file]:
            docset_dir = os.path.join(output_dir, docset)
            if not os.path.exists(docset_dir):
                os.mkdir(docset_dir)
            for doc, string in dictionary[file][docset].items():
                file_path = os.path.join(docset_dir, doc)
                # TODO: Put tokenizer here!
                # NOTE: Paragraphs are not yet split into separate sentences for each line
                #       so that needs to be done here as well.
                #       This seems like a decent sentence tokenizer: https://www.nltk.org/api/nltk.tokenize.html?highlight=tokenize#nltk.tokenize.sent_tokenize
                with open(file_path, 'w') as outfile:
                    outfile.write(string)
    logger.info("Successfully wrote dictionary to files")


if __name__ == '__main__':
    no_fmt, default_fmt = '%(message)s', '(%(levelname)s|%(asctime)s) %(message)s'
    hndlr = logging.FileHandler("src/preprocess/preprocess.log")
    hndlr.setFormatter(logging.Formatter(no_fmt))
    logger.handlers.clear()
    logger.addHandler(hndlr)
    from datetime import datetime
    _now = datetime.now()
    now = [_now.day, _now.month, _now.hour, _now.minute, _now.second]
    now = tuple(map(lambda x: str(x) if x>9 else "0" + str(x), now))
    logger.info("\n======= Script session %s/%s %s:%s:%s =======\n" % now)
    for hndlr in logger.handlers:
        hndlr.setFormatter(logging.Formatter(default_fmt))
    dictionary = read_xml_files_from_directory()
    write_outputs(dictionary, "outputs/D2")