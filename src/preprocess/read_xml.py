#!/bin/env python3

"""
This file contains functions to read and process
XML files. Note that the functions are designed to read
different document structures and formats, e.g., some document IDs have
underscores, while others don't.

Documents from training and devtest are found in the ACQUAINT corpus
Documents from training and evaltest are found /corpora/LDC/LDC10E12/12/TAC_2010_KBP_Source_Data
"""

import os
import re

# to include path in typing
Path = str


CORPUS_PATHS = {
    "AQUAINT": "/corpora/LDC/LDC02T31/",
    "AQUAINT2": "/corpora/LDC/LDC08T25/data/",
    "TAC2011": "/corpora/LDC/LDC10E12/12/TAC_2010_KBP_Source_Data/data/2009/nw"
}


def get_aquaint_path(
        news_org: str, 
        time_period: int,
    ) -> str:
    """
    Get AQUAINT file path based on provided info
    Arguments:
        - news_org: the news organization associated with the file (e.g., NYT)
        - time_period: the time period associated with the file (e.g., 200014)
    """
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


def get_aquaint2_path(
        news_org: str, 
        time_period: int,
    ) -> str:
    """
    Get AQUAINT2 file path based on provided info
    Arguments:
        - news_org: the news organization associated with the file (e.g., NYT)
        - time_period: the time period associated with the file (e.g., 200014)
    """
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


def get_tac_path(
        news_org: str, 
        doc_number: str,
    ) -> str:
    """
    Get TAC file path based on provided info
    Arguments:
        - news_org: the news organization associated with the file (e.g., NYT)
        - doc_number: the document ID number following the news org
    """
    directory = doc_number.split(".")[0]
    path = os.path.join(
        CORPUS_PATHS["TAC2011"], 
        news_org.lower(),
        directory
    )
    return path


def resolve_path(doc_id: str) -> (Path, int):
    """
    Like get_xml_document, but only returns path (for debugging)
    Arguments:
        - doc_id: The id of the document in question
    """
    regex = re.compile(r'([A-Za-z]*|[A-Za-z]*_[A-Za-z]*)_?(\d+\.\d+)')
    parsed_doc = re.match(regex, doc_id)
    news_org, doc = parsed_doc.group(1), parsed_doc.group(2)
    time_period = int(doc.split(".")[0])
    if time_period <= 20009999:
        return get_aquaint_path(news_org, time_period), 1
    elif 20009999 < time_period <= 20060399:
        return get_aquaint2_path(news_org, time_period), 2
    else:
        return get_tac_path(news_org, doc), 3


if __name__ == '__main__':
    import sys
    arg = sys.argv[1]
    print(resolve_path(arg))
