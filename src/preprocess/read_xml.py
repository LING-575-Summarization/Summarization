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


def replace_spaces(text: str) -> str:
    '''
    Helper function that replaces all spacing in a text
    with single spaces (and corrects any spacing at the beginning)
    '''
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(^\s+|\s+$)', '', text)
    return text


def parse_aquaint(
        news_org: str, 
        time_period: int,
        doc_id: str, 
        parser: etree.XMLParser
    ) -> str:
    '''
    Parse an invalid XML in AQUAINT corpus by adding XML headers and then
    going through the file to collect sentences
    Arguments:
        - news_org: the news organization associated with the file (e.g., NYT)
        - time_period: the time period associated with the file (e.g., 200014)
        - doc_id: the document ID to find and extract from
        - parser: an lxml parser to flexibly open the file
    '''
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
    with open(path, 'r') as document:
        doc_as_string = "<XML>" + document.read() + "</XML>"
    documents_tree = etree.fromstring(doc_as_string, parser=parser)
    for child in documents_tree.iterchildren():
        doc = child[0]
        if doc.text.strip() == doc_id:
            for info in child.iterchildren():
                if info.tag == "BODY":
                    article_text = ""
                    for _text in info.iterchildren():
                        # handle headlines and bylines
                        if _text.tag == "HEADLINE" or _text.tag == "DATELINE":
                            text = replace_spaces(_text.text)
                            if re.search(r'\w', text):
                                article_text += (text + ".\n")
                        # handle text
                        elif _text.tag == "TEXT":
                            text = " ".join(
                                [p.text for p in _text.iter()]
                            )
                            text = replace_spaces(text)
                            article_text += (text + ".\n")
                    return article_text


def parse_aquaint2(
        news_org: str, 
        time_period: int,
        doc_id: str
    ) -> str:
    '''
    Parse a valid XML file in AQUAINT2 corpus. Can use regular XML parser here.
    Arguments:
        - news_org: the news organization associated with the file (e.g., NYT)
        - time_period: the time period associated with the file (e.g., 200014)
        - doc_id: the document ID to find and extract from
    '''
    year = str(int(round(time_period/100, 1)))
    file = "".join(
        [news_org.lower(), "_", year, ".xml"]
    )
    path = os.path.join(
        CORPUS_PATHS["AQUAINT2"], 
        news_org.lower(),
        file
    )
    with open(path, 'r') as document:
        doc_as_string = document.read()
    documents_tree = ET.fromstring(doc_as_string)
    doc_nos = documents_tree.findall("DOC")
    for doc_no in doc_nos:
        if doc_no.get('id') != doc_id:
            pass
        else:
            article_text = ""
            for _text in doc_no:
                if _text.tag == 'HEADLINE' or _text.tag == 'DATELINE':
                    article_text += (
                        replace_spaces(_text.text) + ".\n"
                    )
                elif _text.tag == 'TEXT':
                    text = " ".join([p.text for p in _text])
                    text = replace_spaces(text)
                    article_text += text  
            return article_text


def parse_tac(
        news_org: str, 
        doc_id: str,
        doc_number: str,
        parser: etree.XMLParser
    ) -> str:
    '''
    Parse an invalid XML file in TAC shareed task by adding XML headers and then
    going through the file to collect sentences
    Arguments:
        - news_org: the news organization associated with the file (e.g., NYT)
        - doc_id: the full document ID to find and extract from
        - doc_number: the document ID number following the news org
        - parser: an lxml parser to flexibly open the file
    '''
    directory = doc_number.split(".")[0]
    path = os.path.join(
        CORPUS_PATHS["TAC2011"], 
        news_org.lower(),
        directory
    )
    files = os.listdir(path)
    file = [f for f in files if doc_id in f]
    assert len(file) == 1, "Found multiple files found in {} satisfying {}".format(path, doc_id)
    path = os.path.join(path, file[0])
    with open(path, 'r') as document:
        doc_as_string = "<XML>" + document.read() + "</XML>"
    documents_tree = etree.fromstring(doc_as_string, parser=parser)
    doc = next(documents_tree.iterchildren())
    for child in doc.iterchildren():
        if child.tag == 'DOCID':
            assert doc_id in child.text
        elif child.tag == 'BODY':
            article_text = ""
            for _text in child.iterchildren():
                if _text.tag == "HEADLINE":
                    text = replace_spaces(_text.text)
                    if re.search(r'\w', text):
                        article_text += (text + ".\n")
                elif _text.tag == "TEXT":
                    text = " ".join(
                        [p.text for p in _text.iter()]
                    )
                    text = replace_spaces(text)
                    article_text += text
            return article_text


def resolve_path(doc_id: str) -> Path:
    '''
    Like get_xml_document, but only returns path (for debugging)
    Arguments:
        - doc_id: The id of the document in question
    '''
    regex = re.compile(r'([A-Za-z]*|[A-Za-z]*_[A-Za-z]*)_?(\d+\.\d+)')
    parsed_doc = re.match(regex, doc_id)
    news_org, doc = parsed_doc.group(1), parsed_doc.group(2)
    time_period = int(doc.split(".")[0])
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
        return path
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
        return path
    else:
        directory = doc.split(".")[0]
        path = os.path.join(
            CORPUS_PATHS["TAC2011"], 
            news_org.lower(),
            directory
        )
        files = os.listdir(path)
        paths = [os.path.join(path, f) for f in files]
        return paths


def get_xml_document(doc_id: str) -> str:
    '''
    Find the path for the file based on the following heuristics:
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
        return parse_aquaint(news_org, time_period, doc_id, parser)
    elif time_period > 20009999 and time_period <= 20060399:
        return parse_aquaint2(news_org, time_period, doc_id)
    else:
        return parse_tac(news_org, doc_id, doc, parser)


if __name__ == '__main__':
    import sys
    arg = sys.argv[1]
    print(get_xml_document(arg))