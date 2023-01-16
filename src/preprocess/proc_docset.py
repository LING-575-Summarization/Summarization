#!/bin/env python3

"""
This script is used to load all process all of the XML text files
into a dictionary of {dataset split -> {file_code -> file_text}}.
It uses the processing functions defined in read_xml
"""

import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List

from get_data_path import resolve_path
from tokenizer import read_by_corpus_type

Path = str

logging.basicConfig(
    level=logging.INFO,
    datefmt="%m/%d %H:%M:%S"
)
logger = logging.getLogger()


def get_data_dir(file: Path) -> Dict[str, List[Tuple[str, str, int, int]]]:
    """
    Go through the metadata file and acquire a dictionary of file codes to text.
    Args:
        - file: the metadata .xml file of the test split
    """
    with open(file, 'r') as document:
        doc_as_string = document.read()
    documents_list = ET.fromstring(doc_as_string)
    # Only read docsetA per spec
    topic_nodes = documents_list.findall("topic")
    path_dict = {}
    errors = 0
    for topic_node in topic_nodes:
        category = topic_node.get("category")
        docset_node = topic_node.find("docsetA")
        docset_id = docset_node.get("id")
        path_dict[docset_id] = []
        for document in docset_node.findall("doc"):
            doc_id = document.get("id")
            try:
                data_path, corpus_category = resolve_path(doc_id)
                path_dict[docset_id].append((data_path, doc_id, corpus_category, category))
            except FileNotFoundError:
                logger.warning(
                    "Couldn't load file ID: {}. Issue with path: {}?".format(doc_id, resolve_path(doc_id))
                )
                errors += 1
    if errors == 0:
        logger.info("Done with {}! Good news! No errors found!".format(file))
    else:
        logger.info("Done with {}! Uh oh! Ran into {} errors...".format(file, str(errors)))
    return path_dict


def write_outputs(path_dict: Dict[str, List[Tuple[str, str, int, int]]], output_dir: Path):
    """
    Unravel the dictionary output and create directories with files for each document
    in a docset
    """
    os.makedirs(output_dir, exist_ok=True)
    for docset, value in path_dict.items():
        docset_dir = os.path.join(output_dir, docset)
        if not os.path.exists(docset_dir):
            os.mkdir(docset_dir)
        for data_path, doc_id, corpus_type, category_id in value:
            output_path = os.path.join(docset_dir, doc_id)
            read_by_corpus_type(data_path, doc_id, category_id, corpus_type, output_path)
    logger.info("Successfully wrote dictionary to files")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Missing arguments, make sure you contain <input_xml_file> <output_dir>")
        exit(-1)
    else:
        input_xml_file = sys.argv[1]
        output = sys.argv[2]
        no_fmt, default_fmt = '%(message)s', '(%(levelname)s|%(asctime)s) %(message)s'
        hndlr = logging.FileHandler("src/preprocess/preprocess.log")
        hndlr.setFormatter(logging.Formatter(no_fmt))
        logger.handlers.clear()
        logger.addHandler(hndlr)
        from datetime import datetime

        _now = datetime.now()
        now = [_now.day, _now.month, _now.hour, _now.minute, _now.second]
        now = tuple(map(lambda x: str(x) if x > 9 else "0" + str(x), now))
        logger.info("\n======= Script session %s/%s %s:%s:%s =======\n" % now)
        for hndlr in logger.handlers:
            hndlr.setFormatter(logging.Formatter(default_fmt))
        dict = get_data_dir(input_xml_file)
        write_outputs(dict, output)