import unittest
from tokenizer import get_root, read_aquaint
from lxml import etree

parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_read_aquaint(self):
        root = get_root("snip/aquaint.xml")
        headline, body = read_aquaint(root, "APW19980602.0004")
        print(headline)
        print(body)
        self.assertEqual(headline, "\"One-in-100-year flood event\" devastates Western Australia")
        self.assertEqual(body[2], "Authorities estimate it could take months")




if __name__ == '__main__':
    unittest.main()
