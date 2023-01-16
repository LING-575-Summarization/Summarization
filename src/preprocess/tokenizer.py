import typing
import re
from lxml import etree
import spacy


def read_by_corpus_type(data_path: str, doc_id: str, category: int, corpus_type: int, output_path: str,
                        nlp: spacy.Language):
    output = open(output_path, "w+")

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
    write_output(output, category, date, headline, body, nlp)


def read_aquaint(root: etree.Element, doc_id: str) -> (str, [str]):
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
                for s in body_node.find("TEXT").text.split('\t'):
                    s = s.strip().replace('\n', ' ')
                    if s != '':
                        body.append(s)
            # We now find what we need, break so we can move on
            break
    return headline, body


def read_aquaint2(root: etree.Element, doc_id: str) -> (str, [str]):
    headline = "NONE"
    body = []
    for child in root.find("DOCSTREAM").findall("DOC"):
        # Compare the id attributes' text with doc_id
        if child.get("id").strip() == doc_id:
            if child.find("HEADLINE") is not None:
                headline = child.find("HEADLINE").text.strip().replace('\n', ' ')
            body = extract_p(child)
            # We now find what we need, break so we can move on
            break
    return headline, body


def read_tac(root: etree.Element) -> (str, [str]):
    body_node = root.find("DOC").find("BODY")
    headline = "NONE"
    if body_node.find("HEADLINE") is not None:
        headline = body_node.find("HEADLINE").text.strip().replace('\n', ' ')
    body = extract_p(body_node)
    return headline, body


def write_output(output: typing.TextIO, category: int, date: str, headline: str, body: [str], nlp: spacy.Language):
    output.write("DATE_TIME: " + date + "\n")
    output.write("CATEGORY: " + str(category) + "\n")
    output.write("HEADLINE: " + headline + "\n")
    output.write("\n")
    for line in body:
        output.write(str(tokenizer(line, nlp)) + "\n")
    output.close()


def extract_p(root: etree.Element) -> [str]:
    result = []
    for p_node in root.find("TEXT"):
        s = p_node.text.strip().replace('\n', ' ')
        if s != '':
            result.append(s)
    return result


def tokenizer(paragraph: str, nlp: spacy.Language) -> [str]:
    for line in paragraph:
        if line:
            doc = nlp(paragraph)
    return [token.text for token in doc]


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
