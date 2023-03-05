from typing import Tuple, List, TextIO, Union
import re
from lxml import etree
import nltk.data
from nltk.tokenize import word_tokenize
import nltk


# Wrap the nltk.data.load() tokenizer in a class to avoid downloading punkt
class SentenceTokenizer:
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def __call__(self, string: Union[str, List[str]]) -> Union[str, List[List[str]]]:
        if isinstance(string, str):
            return self.tokenizer.tokenize(string.strip())
        elif isinstance(string, list):
            return [self.tokenizer.tokenize(s.strip()) for s in string]
        else:
            raise ValueError("SentenceTokenizer takes strings or lists of strings")


# Set the SentenceTokenizer class as a global callable function
sent_tokenize = SentenceTokenizer()

# copyright strings to avoid including
COPYRIGHT_STRINGS = [
    'COPYRIGHT 1998 BY WORLDSOURCES',
    "NO PORTION OF THE MATERIALS CONTAINED HEREIN MAY BE USED IN ANY MEDIA WITHOUT ATTRIBUTION TO WORLDSOURCES, INC."
]

# headers to be excluding
REMOVE_PATTERN = [
    r'^([A-Z]{2,}|D\.C\.).*\(.*?\)(\s?\-\-|_|:)',
    r'\(Begin optional trim\)',
    r'\(Begin optional trim\)',
    r'^SOURCES: .*$',
    r'\(STORY CAN END HERE . OPTIONAL MATERIAL FOLLOWS.\)',
    r'\(END OPTIONAL TRIM\)',
    r'\(End trim\)',
    r'\(optional trim ends here\)',
    r'For Use By Clients of the New York Times News Service',
    r'Story Filed By Cox Newspapers'
]


def read_by_corpus_type(data_path: str, doc_id: str, category: int, corpus_type: int):
    root = get_root(data_path)
    date = get_date(doc_id)
    headline = ""
    body = []
    if corpus_type == 1:
        headline, body = read_aquaint(root, doc_id)
    elif corpus_type == 2:
        headline, body = read_aquaint2(root, doc_id)
    elif corpus_type == 3:
        headline, body = read_tac(root)
    return category, date, headline, body


def read_aquaint(root: etree.Element, doc_id: str) -> Tuple[str, List[str]]:
    headline = "NONE"
    body = []
    for child in root.findall("DOC"):
        # Compare the <DOCNO> text with doc_id
        if child.find("DOCNO").text.strip() == doc_id:
            # Grab the BODY section
            body_node = child.find("BODY")
            if body_node.find("HEADLINE") is not None:
                headline = body_node.find("HEADLINE").text.strip().replace('\n', ' ')
            if body_node.find("TEXT").find("P") is not None:
                body = extract_p(body_node)
            else:
                body = extract_p_manual(body_node)
            if body == []:
                print("Missing", doc_id)
            # We now find what we need, break so we can move on
            break
    return headline, body


def read_aquaint2(root: etree.Element, doc_id: str) -> Tuple[str, List[str]]:
    headline = "NONE"
    body = []
    for child in root.find("DOCSTREAM").findall("DOC"):
        # Compare the id attributes' text with doc_id
        if child.get("id").strip() == doc_id:
            if child.find("HEADLINE") is not None:
                headline = child.find("HEADLINE").text.strip().replace('\n', ' ')
            if child.find("TEXT").find("P") is not None or child.find("TEXT").find("p") is not None:
                body = extract_p(child)
            else:
                body = extract_p_manual(child)
            if body == []:
                print("Missing", doc_id)
            # We now find what we need, break so we can move on
            break
    return headline, body


def read_tac(root: etree.Element) -> Tuple[str, List[str]]:
    body_node = root.find("DOC").find("BODY")
    headline = "NONE"
    if body_node.find("HEADLINE") is not None:
        headline = body_node.find("HEADLINE").text.strip().replace('\n', ' ')
    body = extract_p(body_node)
    return headline, body


def write_output(output_path: TextIO, category: int, date: str, headline: str, body: List[List[str]]):
    output = open(output_path, "w+")
    output.write("DATE_TIME: " + date + "\n")
    output.write("CATEGORY: " + str(category) + "\n")
    output.write("HEADLINE: " + headline + "\n")
    output.write("\n")
    save_paras = list()
    for paragraph in body:
        save_sents = list()
        for line in paragraph:
            tokenized_sent = word_tokenize(line)
            save_sents.append(tokenized_sent)
            output.write(str(tokenized_sent) + "\n")
        save_paras.append(save_sents)
        output.write("\n")  # extra line between paragraphs
    output.close()
    return category, date, headline, save_paras


def extract_p(root: etree.Element) -> List[List[str]]:
    result = []
    for p_node in root.find("TEXT"):
        s = p_node.text.strip().replace('\n', ' ')
        s = re.sub(r'\s{2,}', ' ', s)
        # replace first sentence if it contains a header
        for pattern in REMOVE_PATTERN:
            if re.search(pattern, s):
                s = re.sub(pattern, '', s)
        if s != '':
            result.append(sent_tokenize(s))
    return result


def extract_p_manual(body_node: etree.Element) -> List[List[str]]:
    result = []
    for s in re.split(r'(?<=\.|_)\s+(?!.*INC.)(?=\w)', body_node.find("TEXT").text):
        s = re.sub('[\n\t\s]+', ' ', s)
        s = re.sub('(^\s+|\s+$)', '', s)
        # replace first sentence if it contains a header
        for pattern in REMOVE_PATTERN:
            if re.search(pattern, s):
                s = re.sub(pattern, '', s)
        if re.search('\S', s) and not any([cs in s for cs in COPYRIGHT_STRINGS]):
            result.append(sent_tokenize(s))
    return result


def get_root(input_file: str) -> etree.Element:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)
    with open(input_file, 'r') as f:
        output = "<root>" + f.read() + "</root>"
    return etree.fromstring(output, parser)


def get_date(doc_id: str) -> str:
    # There are three types of doc_id
    # AQUAINT: APW19980602.1383
    # AQUAINT2: APW_ENG_20041007.0256
    # TAC: AFP_ENG_20061002.0523

    # Remove the APW or APW_ENG prefix
    result = re.sub(r"^[\D_]+", "", doc_id)
    # Remove the article suffix
    result = re.sub(r"\.[\w]+", "", result)
    return result
