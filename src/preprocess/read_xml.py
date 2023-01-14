#!/bin/env python3

'''
This file contains functions to read and process
XML files. Note that the functions are designed to read
different document structures and formats, e.g., some document IDs have
underscores, while others don't.

Documents from training and devtest are found in the ACQUAINT corpus
Documents from training and evaltest are found /corpora/LDC/LDC10E12/12/TAC_2010_KBP_Source_Data
'''

from lxml import etree
import xml.etree.ElementTree as ET
import os
import re
from typing import *

# to include path in typing
Path = str


CORPUS_PATHS = {
    "AQUAINT": "/corpora/LDC/LDC02T31/",
    "AQUAINT2": "/corpora/LDC/LDC08T25/data/",
    "TAC2011": "/corpora/LDC/LDC10E12/12/TAC_2010_KBP_Source_Data/data/2009/nw"
}


def parse_aquaint(path: Path, doc_id: str, parser: etree.XMLParser) -> str:
    '''Parse an invalid XML in AQUAINT corpus by adding XML headers and then
        going through the file to collect sentences
        Arguments:
            - Path: path to file to open
            - doc_id: the document ID to find and extract from
            - parser: an lxml parser to flexibly open the file
    '''
    with open(path, 'r') as document:
        doc_as_string = "<XML>" + document.read() + "</XML>"
    documents_tree = etree.fromstring(doc_as_string, parser=parser)
    for child in documents_tree.iterchildren():
        doc = child[0]
        if doc.text.strip() == doc_id:
            for info in child.iterchildren():
                if info.tag == "BODY":
                    for text in info.iterchildren():
                        if text.tag == "TEXT":
                            text = " ".join(
                                [p.text for p in text.iter()]
                            )
                            text = re.sub(r'\s+', ' ', text)
                            text = re.sub(r'(^\s+|\s+$)', '', text)
                            return text


def parse_tac(path: Path, doc_id: str, parser: etree.XMLParser) -> str:
    '''Parse an invalid XML file in AQUAINT corpus by adding XML headers and then
        going through the file to collect sentences
        Arguments:
            - Path: path to file to open
            - doc_id: the document ID to find and extract from
            - parser: an lxml parser to flexibly open the file
    '''
    with open(path, 'r') as document:
        doc_as_string = "<XML>" + document.read() + "</XML>"
    documents_tree = etree.fromstring(doc_as_string, parser=parser)
    doc = next(documents_tree.iterchildren())
    for child in doc.iterchildren():
        if child.tag == 'DOCID':
            assert doc_id in child.text
        elif child.tag == 'BODY':
            print([x.tag for x in child.iterchildren()])
            for text in child.iterchildren():
                if text.tag == "TEXT":
                    text = " ".join(
                        [p.text for p in text.iter()]
                    )
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'(^\s+|\s+$)', '', text)
                    return text


def parse_aquaint2(path: Path, doc_id: str) -> str:
    '''Parse a valid XML file in AQUAINT2 corpus
        Arguments:
            - Path: path to file to open
            - doc_id: the document ID to find and extract from
    '''
    with open(path, 'r') as document:
        doc_as_string = document.read()
    documents_tree = ET.fromstring(doc_as_string)
    doc_nos = documents_tree.findall("DOC")
    for doc_no in doc_nos:
        if doc_no.get('id') != doc_id:
            pass
        else:
            text = doc_no.find('TEXT')
            text = " ".join([p.text for p in text])
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'(^\s+|\s+$)', '', text)
            return text


def get_xml_document(doc_id: str) -> Path:
    '''Find the path for the file based on the following heuristics:
            If <= 2000, then search in AQUAINT (parse_invalid)
            If >= 2000 and <= 2006.03 then search in AQUAINT2 (parse_valid)
            If >= 2006.03 then search in TAC2011 (parse_invalid)
        Each element of the if/else clause is used to resolve the paths
        to the files given their different naming conventions
    Returns: The text in the file as a string
    Arguments:
        - doc_id: longer string corresponding to the document to be searched
            in a particular corpus e.g. NYT_ENG_20050112.0012
    NOTE: some doc_ids do not contain _
    CITATION: https://stackoverflow.com/q/38853644/ helped problem solve opening xml file
    '''
    regex = re.compile(r'([A-Za-z]*|[A-Za-z]*_[A-Za-z]*)_?(\d+\.\d+)')
    parsed_doc = re.match(regex, doc_id)
    news_org, doc = parsed_doc.group(1), parsed_doc.group(2)
    time_period = int(doc.split(".")[0])
    parser = etree.XMLParser(recover=True)
    if time_period <= 20009999:
        year = str(int(round(time_period/10000, 1)))
        news_org_f = "XIN" if news_org == "XIE" else news_org
        news_org_f = news_org_f + "_ENG" if news_org_f != "NYT" else news_org_f
        file = "".join([str(time_period), "_", news_org_f])
        path = os.path.join(
            CORPUS_PATHS["AQUAINT"], 
            news_org.lower(), 
            year,
            file
        )
        return parse_aquaint(path, doc_id, parser)
    elif time_period > 20009999 and time_period <= 20060399:
        year = str(int(round(time_period/100, 1)))
        file = "".join(
            [news_org.lower(), "_", year, ".xml"]
        )
        path = os.path.join(
            CORPUS_PATHS["AQUAINT2"], 
            news_org.lower(),
            file
        )
        return parse_aquaint2(path, doc_id)
    else:
        directory = doc.split(".")[0]
        file = "".join(
            [news_org, "_", doc, ".LDC2009T13", ".sgm"]
        )
        path = os.path.join(
            CORPUS_PATHS["TAC2011"], 
            news_org.lower(),
            directory,
            file
        )
        return parse_tac(path, doc_id, parser)


if __name__ == '__main__':
    import sys
    arg = sys.argv[1]
    print(get_xml_document(arg))