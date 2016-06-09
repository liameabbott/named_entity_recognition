#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:35:22 2016

@author: liabbott
"""

import os
import pandas as pd
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize


def clean_text(text):
    """Clean the input text string by stripping out unwanted characters and
       making certain string replacements to assist in the classification of
       named entities found in the text."""

    symbols_to_remove = [u"\u201a",
                         u"\u201c",
                         u"\u201d",
                         u"\u0160",
                         u"\u0152",
                         u"\u2013",
                         "0",
                         "1",
                         "2",
                         "3",
                         "4",
                         "5",
                         "6",
                         "7",
                         "8",
                         "9"]
    for symbol in symbols_to_remove:
        text = text.replace(symbol, "")

    word_replace = []
    with open('nerStringsToReplace.txt', 'rb') as f:
        for line in f.readlines():
            words = line.split(',')
            word_replace.append((words[0], words[1].rstrip()))

    for word in word_replace:
        text = text.replace(word[0], word[1])

    return text


def classify_text(text):
    """Using the 3-class Stanford Named Entity Recognition model, classify each
       word in the input text as a PERSON, LOCATION, ORGANIZATION, or O (for
       other)."""

    directory = "C:/Users/liabbott/Documents/Projects/CBP OIT/stanford_ner/"
    mod = "classifiers/english.all.3class.distsim.crf.ser.gz"
    tag = "stanford-ner.jar"
    path_to_model = os.path.normpath(directory + mod)
    path_to_tagger = os.path.normpath(directory + tag)
    st = StanfordNERTagger(path_to_model, path_to_tagger, encoding='utf-8')

    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)

    return classified_text


def reclassify_words(df):
    """Taking a dataframe of words and their respective classifications,
       reclassify certain words according to mappings defined in directory
       file."""

    types_to_change = []
    with open('nerTypesToChange.txt', 'rb') as f:
        for line in f.readlines():
            if line == '\n':
                continue
            words = line.split(',')
            types_to_change.append((words[0], words[1].rstrip()))

    for word_type in types_to_change:
        df.ix[df['word'] == word_type[0], 'type'] = word_type[1]

    return df


def agg_count_words(df):
    """Taking a dataframe of words and their respective classifications,
       aggregate the dataframe to one unique word per row and add a column
       counting the number of occurences of each word. Sort the returned
       dataframe by word counts within each classification type."""

    cnts = df.groupby('word').size()
    types = df.groupby('word').type.first()

    df = pd.DataFrame({'document': df['document'][0],
                       'word': list(cnts.index),
                       'count': list(cnts),
                       'type': list(types)})

    df['type'] = pd.Categorical(df['type'],
                                ['O', 'PERSON', 'ORGANIZATION', 'LOCATION'])
    df = df.sort_values(['type', 'count'], ascending=False)

    return df


def main():
    """Define main function of the script."""

    # read in names of documents to load
    document_filename = 'DOC_FILENAMES.txt'
    print "Reading document filenames from '{0}'...".format(document_filename)
    with open(document_filename, 'rb') as f:
        filenames = [line.strip() for line in f.readlines()]
    print "Document filenames read."

    # load each document
    df_total = pd.DataFrame(columns=['document', 'word', 'count', 'type'])
    data_path = os.path.join(os.getcwd(), 'data')
    print "Cleaning and classifying text from documents ", \
          "in '../data/' directory..."
    for name in filenames:

        # load document text
        with open(os.path.join(data_path, name + '.txt'), 'rb') as f:
            text = f.read().decode('utf8')

        # clean the text string by stripping out unwanted characters and
        # making certain string replacements to assist in the classification
        # of entities
        text = clean_text(text)

        # split the text into words and classify each word
        classified_text = classify_text(text)
        n_words = len(classified_text)

        # load words and their classification into a dataframe
        df = pd.DataFrame(index=range(n_words), columns=['word', 'type'])
        for i in range(n_words):
            word_type = classified_text[i]
            df.ix[i, 'document'] = name
            df.ix[i, 'word'] = word_type[0]
            df.ix[i, 'type'] = word_type[1]

        # change the classification of certain words
        df = reclassify_words(df)

        # aggregate the dataframe to one unique word per row, counting
        # the number of occurences of each word.
        df = agg_count_words(df)

        # add document dataframe to master dataframe
        phrase = "Adding '{0}' document dataframe to master dataframe..."
        print phrase.format(name)
        df_total = pd.concat([df_total, df], ignore_index=True)

    # filter out words that occur only once across corpus
    counts = df_total.groupby('word').count()
    words_to_keep = counts[counts['count'] > 1].index

    # filter to include only words that occur more than once
    df_total = df_total[df_total['word'].isin(words_to_keep)]
    df_total = df_total.sort_values(by=['document', 'type', 'count', 'word'],
                                    ascending=[True, False, False, True])

    # filter out type 'O' (other) rows
    df_total = df_total[df_total['type'] != 'O']

    # write results to csv file
    output_path = os.path.join(os.getcwd(), 'output')
    df_total.to_csv(os.path.join(output_path, 'nerDocsWordsCountsTypes.csv'),
                    encoding='utf-8',
                    index=False,
                    columns=['document', 'type', 'word', 'count'])
    print "Document names, words, types, and counts saved to ", \
          "'../output/nerDocsWordsCountsTypes.csv' (omitted words ", \
          "occuring only once across corpus)."


if __name__ == '__main__':
    main()
