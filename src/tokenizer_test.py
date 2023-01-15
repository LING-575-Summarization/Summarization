import unittest
from tokenizer import get_root, read_aquaint, read_aquaint2, read_tac
from lxml import etree

parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)


class TestTokenizer(unittest.TestCase):
    def test_read_aquaint(self):
        root = get_root("snip/aquaint.xml")
        headline, body = read_aquaint(root, "APW19980602.0004")
        self.assertEqual(headline, "\"One-in-100-year flood event\" devastates Western Australia")
        self.assertEqual(body[2], "Authorities estimate it could take months")

    def test_read_aquaint2(self):
        root = get_root("snip/aquaint2.xml")
        headline, body = read_aquaint2(root, "APW_ENG_20041001.0002")
        self.assertEqual(headline, "\"One-in-100-year flood event\" devastates Western Australia")
        self.assertEqual(body[2], "Authorities estimate it could take months")

    def test_read_tac(self):
        root = get_root("snip/tac.sgm")
        headline, body = read_tac(root)
        self.assertEqual(headline, "\"One-in-100-year flood event\" devastates Western Australia")
        self.assertEqual(body[2], "Authorities estimate it could take months")


if __name__ == '__main__':
    unittest.main()
