import csv
import pandas as pd
import xml.etree.ElementTree as ET
import re
from html.parser import HTMLParser

#list of global variables for script
data = pd.DataFrame()
ngrams = dict()

def extract_ngrams(message, n):
    return

def extract_user_text(user):
    filename = 'Resources/data/' + user + '.xml'
    xml = ET.parse(filename)
    root = xml.getroot();
    href = re.compile('<a*.?</a>')

    p = HTMLParser()
    for documents in root:
        for document in documents:
            print (document.text)

    return

def build_target_data():
    target = open('Resources/data/truth.txt', 'r')
    target = target.readlines()

    with open ('Resources/data.csv', 'w') as data:
        datawriter = csv.writer(data)
        datawriter.writerow(['user', 'sex', 'age'])
        for users in target:
            user = users.split(':::')
            datawriter.writerow(user)


def main():
    build_target_data()

    global data
    data = pd.read_csv('Resources/data.csv')
    data = pd.DataFrame(data)

    #count dictionaries of n-grams for every age/sex class
    row = 0
    for index, row in data.iterrows():
        user = row['user']
        text_corpus = extract_user_text(user)
        break



if __name__ == "__main__":
    main()