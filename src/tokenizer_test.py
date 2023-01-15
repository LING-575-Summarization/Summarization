import os
import unittest

from lxml import etree

from tokenizer import get_root, read_aquaint, read_aquaint2, read_tac, get_date, tokenizer, write_output, \
    read_by_corpus_type

parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)

master_headline = "\"One-in-100-year flood event\" devastates Western Australia"
master_body = [
    "Test, Test (WIKINEWS) _ Aerial evacuations took place and food was airlifted in yesterday after a "
    "devastating flood Western Australia emergency services minister Stephen Dawson called the 'worst in a "
    "century' isolated communities in the Kimberley.",
    'Flooding began last week after heavy rain from Tropical Cyclone Ellie swelled local rivers, bolstered by '
    'La Niña. Notably, the Fitzroy River broke a 2002 record of 13.95 meters (45.8 ft), reaching a water '
    'level of 15.81 meters (51.9 ft) on Wednesday, according to a Bureau of Meteorology spokesperson.',
    'Authorities estimate it could take months']
temp = "temp.txt"


class TestTokenizer(unittest.TestCase):
    def test_get_root(self):
        root = get_root("snip/tac.sgm")
        headline = root.find("DOC").find("BODY").find("HEADLINE").text.strip().replace('\n', ' ')
        self.assertEqual(headline, master_headline)

    def test_read_aquaint(self):
        root = get_root("snip/aquaint.xml")
        headline, body = read_aquaint(root, "APW19980602.0004")
        self.assertEqual(headline, master_headline)
        self.assertEqual(body, master_body)

    def test_read_aquaint2(self):
        root = get_root("snip/aquaint2.xml")
        headline, body = read_aquaint2(root, "APW_ENG_19980602.0002")
        self.assertEqual(headline, master_headline)
        self.assertEqual(body, master_body)

    def test_read_tac(self):
        root = get_root("snip/tac.sgm")
        headline, body = read_tac(root)
        self.assertEqual(headline, master_headline)
        self.assertEqual(body, master_body)

    def test_get_data(self):
        self.assertEqual(get_date("APW19980602.1383"), "19980602")
        self.assertEqual(get_date("APW_ENG_20041007.0256"), "20041007")
        self.assertEqual(get_date("AFP_ENG_20061002.0523"), "20061002")

    def test_tokenizer(self):
        result = tokenizer("Authorities estimate it could take months")
        self.assertEqual(len(result), 6)
        self.assertEqual(result[2], "it")

    def check_two_txt(self):
        with open(temp) as test, open('snip/gold.txt') as gold:
            for line1, line2 in zip(test, gold):
                self.assertEqual(line1, line2)
        os.remove(temp)

    def test_write_output(self):
        output = open(temp, "w+")
        date = "19980602"
        write_output(output, 1, date, master_headline, master_body)
        self.check_two_txt()

    def test_read_by_corpus_type(self):
        read_by_corpus_type("snip/aquaint.xml", "APW19980602.0004", 1, 1, temp)
        self.check_two_txt()
        read_by_corpus_type("snip/aquaint2.xml", "APW_ENG_19980602.0002", 1, 2, temp)
        self.check_two_txt()
        read_by_corpus_type("snip/tac.sgm", "AFP_ENG_19980602.0149", 1, 3, temp)
        self.check_two_txt()


if __name__ == '__main__':
    unittest.main()
