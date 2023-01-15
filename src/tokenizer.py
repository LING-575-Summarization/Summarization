import typing
import re
from lxml import etree
import spacy


def read_by_corpus_type(data_path: str, doc_id: str, category: int, corpus_type: int, output_path: str):
    output = open(output_path, "w+")

    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)
    root = get_root(data_path)
    date = get_date(doc_id)
    headline = ""
    body = []
    if corpus_type == 1:
        headline, body = read_aquaint(root, doc_id)
    write_output(output, category, date, headline, body)


def read_aquaint(root: etree.Element, doc_id: str) -> (str, [str]):
    headline = ""
    body = []
    print(doc_id)
    for child in root.findall("DOC"):
        # Compare the <DOCNO> text with doc_id
        if child.find("DOCNO").text.strip() == doc_id:
            # Grab the BODY section
            body = child.find("BODY")
            headline = body.find("HEADLINE").text
            body = [re.sub('\n', ' ', s.strip()) for s in body.find("TEXT").text.split('\t')]
            # We now find what we need, break so we can move on
            break
    return headline, body


def write_output(output: typing.TextIO, category: int, date: str, headline: str, body: [str]):
    output.write("DATE_TIME: " + date)
    output.write("CATEGORY: " + str(category))
    output.write("HEADLINE: " + headline)
    output.write("\n")


def tokenizer(input: str):
    nlp = spacy.load("en_core_web_sm")
    for line in input:
        if line:
            doc = nlp(str)
    return doc


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
