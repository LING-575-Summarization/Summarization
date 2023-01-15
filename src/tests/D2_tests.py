import os
import re
from xml.dom import minidom

"""
All tests must be run at the top level directory of the repository, at Summarization/
It is assumed you are running this in patas as the data is stored in the patas dropbox on 1/13/23.
It is also assumed that your user root directory ("~") contains the shared dropbox symbolic link.
"""

def dir_exists():
    """
    Checks if all directories exists for processing xml files
    """
    # outer output dir
    assert os.path.exists("outputs/output_dir")

    # three subdirectories
    assert os.path.exists("outputs/training")
    assert os.path.exists("outputs/devtest")
    assert os.path.exists("outputs/evaltest")

   # three sub-sub directories 
    assert os.path.exists("outputs/training/output_dir")
    assert os.path.exists("outputs/devtest/output_dir")
    assert os.path.exists("outputs/evaltest/output_dir")



def script_exists():
    """
    scripts to process xml files exist
    """
    assert os.path.exists("src/proc_docset.sh")
    assert os.path.exists("src/proc_docset.py")


def docset_a_stuff_exists(data, path):
    """
    Checks that directories and files exist with right names, concerning docsetA specifically
    """
    for dir_name, file_name_list in data.items():
        sub_path = path + "/output_dir/" + dir_name
        assert os.path.exists(sub_path)
        for file_name in file_name_list:
            assert os.path.exists(sub_path + "/" + file_name)


def process_xml(path):
    assert os.path.exists(path)
    processed_xml = dict()
    xml_file = minidom.parse(path)
    docset_a_files = xml_file.getElementsByTagName("docsetA")
    for docset_a in docset_a_files:
        docset_a_id = docset_a.attributes['id'].value
        doc_a_doc_list = docset_a.getElementsByTagName("doc")
        doc_a_doc_id_list = list()
        for doc_a_doc in doc_a_doc_list:
            doc_a_doc_id = doc_a_doc.attributes['id'].value
            doc_a_doc_id_list.append(doc_a_doc_id)
        processed_xml[docset_a_id] = doc_a_doc_id_list
    return processed_xml


def check_correct_format(string: str):
    '''Checks if the document is in the correct format or not'''
    check_regex = re.compile(
        r'((^[A-Z]*: .*$)*)((\n\n.*)*)', 
        flags=re.MULTILINE
    )
    if check_regex:
        return True
    else:
        return False


def run_all_tests():
    '''
    Run all the tests listed in this file
    '''
    script_exists()
    print("all scripts exist.")
    dir_exists()
    print("all of training, devtest, evaltest, output_dir directories exist.")

    home = os.path.expanduser("~")
    training_path = os.path.join(
        home, "dropbox/22-23/575x/Data/Documents/training/2009/UpdateSumm09_test_topics.xml"
    )
    devtest_path = os.path.join(
        home, "dropbox/22-23/575x/Data/Documents/devtest/GuidedSumm10_test_topics.xml"
    )
    evaltest_path = os.path.join(
        home, "dropbox/22-23/575x/Data/Documents/evaltest/GuidedSumm11_test_topics.xml"
    )

    training_processed = process_xml(training_path)
    devtest_processed = process_xml(devtest_path)
    evaltest_processed = process_xml(evaltest_path)
    print("all paths to original data exists and data is now pre-processed for checking.")

    docset_a_stuff_exists(training_processed, "outputs/training")
    docset_a_stuff_exists(devtest_processed, "outputs/devtest")
    docset_a_stuff_exists(evaltest_processed, "outputs/evaltest")
    print("all docSetA subdirectories and all files in docSetA subdirectories exist.")

    print("### All tests passed! :) ###")


if __name__ == '__main__':
    run_all_tests()
