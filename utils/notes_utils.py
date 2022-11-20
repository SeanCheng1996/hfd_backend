"""
    This file contains code for the procedures used to clean and analyze the notes column in the dataset
"""
#Import Statements
from __future__ import print_function
# library to clean data
import re
# Natural Language Tool Kit
import nltk
nltk.download('stopwords')
# to remove stopword
from nltk.corpus import stopwords
# for Stemming propose
from nltk.stem.porter import PorterStemmer
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
# import difflib
import jellyfish
import pandas as pd
import numpy as np
import itertools
import nltk
from string import punctuation as PUNCTUATIONS
import sklearn
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
# from ggplot import *
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import notes_utils
import pandas as pd
nlp = spacy.load("en_core_web_sm")


def do_nlp_process(data):
    """
      This is a helper function which cleans and processes the data
      => Removal of Stopwords (not including negation words)
      => Removal of punctuations and numbers
      => Generating Tokens from text data
      Input:
        data = Text -  within the notes column
      Output:
        [data] = a deep copy of [data] with column in [columns] processed
      """

    tokens = []
    text_ = []
    text_lemma = []
    nouns = []
    namedEnt = []
    DependencyParDict = {}
    NEGATIONS = ['not', 'neither', 'never', 'no', 'nobody', 'none', 'nor', 'nothing', 'nowhere',
                 'without', 'hardly', 'lack', 'barely', 'rarely', 'seldom']  # exclude negations from stopwords

    for doc in nlp.pipe(data, batch_size=1000):
        # String Text in Lemma
        text_lemma.append(
            " ".join([tk.lemma_ if tk.lemma_ != '-PRON-' else tk.text.lower() for tk in doc])
        )
        # String text with more process
        text_.append(" ".join([tk.lemma_ for tk in doc
            if ((tk.lemma_ not in ENGLISH_STOP_WORDS) or (tk.lemma_ in NEGATIONS))
            and tk.lemma_ not in PUNCTUATIONS
            and tk.lemma_ != '-PRON-'
            and tk.dep_ not in ['punct']
            and tk.pos_ not in ['PUNCT','SPACE','PROPN']
            and not(tk.dep_=='pobj' and tk.pos_=='NOUN')
            and (tk.pos_!='NUM' or (tk.pos_ == 'NUM' and tk.head.lemma_ in ['day','week','month','score','star']))
         ]))
        # List of tokens
        tokens.append([tk.lemma_ for tk in doc
            if ((tk.lemma_ not in ENGLISH_STOP_WORDS) or (tk.lemma_ in NEGATIONS))
            and tk.lemma_ not in PUNCTUATIONS
            and tk.lemma_ != '-PRON-'
            and tk.dep_ not in ['punct']
            and tk.pos_ not in ['PUNCT','SPACE','PROPN']
            and not(tk.dep_=='pobj' and tk.pos_=='NOUN')
            and (tk.pos_!='NUM' or (tk.pos_ == 'NUM' and tk.head.lemma_ in ['day','week','month','score','star']))
         ])

        ### Noun Entity Recognize, Named Entity Recognition
        nouns.extend(list(doc.noun_chunks))
        namedEnt.extend(list(doc.ents))
    return (text_lemma, tokens, text_, nouns, namedEnt)

def get_namedEntity_list(data):
    """
    Function to extract names entities from text data  persons, date, time, money, percent, companies, locations, organizations
    Input:
        data = Text -  within the notes column
    Output: List of named entities extracted
    """

    PERSON = [ner.lemma_ for ner in list(set(data)) if ner.label_  in ['PERSON','ORG']] # NamedEntity Persons
    result_list = [ner.lemma_ for ner in list(set(data))
                    if len(ner.lemma_.split())>1 #and len(ner.text.split())<=2
                     and ner.label_ not in ['PERSON','TIMEI','ORG','CARDINAL','GPE','PRODUCT']
                     ] # contains digits for Date, Time, Percent, Money...
    return result_list


def get_Nouns_list(data):
    """
        Function to extract nouns  without stopwords (except negation), digits, PERSON names
        Input:
            data = Text -  within the notes column
        Output: List of nouns extracted
        """
    NEGATIONS = ['not', 'neither', 'never', 'no', 'nobody', 'none', 'nor', 'nothing', 'nowhere',
                 'without', 'hardly', 'lack', 'barely', 'rarely', 'seldom']  # exclude negations from stopword
    PERSON = [ner.lemma_ for ner in list(set(data)) if ner.label_  in ['PERSON','ORG']] # NamedEntity Persons
    result_list = [" ".join([i for i in ner.lemma_.split() if (i not in ENGLISH_STOP_WORDS) or (i in NEGATIONS)])

                  for ner in list(set(data))
                  if len(ner.lemma_.split())>1 #and len(ner.text.split())<=2 # only bi_grams
                  and ('-PRON-' not in ner.lemma_.split())
                  and all([(i not in PUNCTUATIONS and i.isalpha() and len(i)>1) for i in ner.lemma_.split()]) # letter only
                  ]
    result_list = [i for i in list(set(result_list)) if len(i.split()) > 1 and i not in PERSON]
    return result_list

def get_collocation(data):
    """
    Function to extract top collocations
    Input:
        data = Text -  within the notes column
    Output: List of top collocations extracted
    """
    N_TOP_COL = 100 # number of top collocations
    words = []
    for t in data:
        words += t
    bgm    = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(words)
    scored = finder.score_ngrams( bgm.likelihood_ratio  )
    finder.apply_freq_filter(min_freq = 1)
    colls = finder.nbest(bgm.likelihood_ratio, N_TOP_COL)
    collocation = [" ".join(col) for col in colls]
    # tri_gm   = nltk.collocations.BigramAssocMeasures()
    # finder_tri = nltk.collocations.TrigramCollocationFinder.from_words(words)
    # scored_tri = finder_tri.score_ngrams(tri_gm.raw_freq)
    # print(scored_tri)
    return collocation

def create_vocabulary(data, n_grams):
    """
    function to create a custom vocabulary for the topic model which includes n_grams
    INPUT - text data , n-grams within the text data
    Output - Vocabulary corpus for topic model
    """

    # Create custom Vocabulary
    vectorizer = CountVectorizer(stop_words=None, lowercase=False, max_df=0.9, min_df=0.005)
    X = vectorizer.fit_transform(data)
    vocabulary_ = vectorizer.get_feature_names()

    # extend Vocal with n_grams
    vocabulary_.extend(n_grams)
    vocabulary_processed = [x for x in vocabulary_ if (len(x.split())>1 or x.isalpha())] # more than one letter
    return vocabulary_processed

def get_topic_words(modelName, data, vocabulary, no_top_words=5, no_topics=20, apply_transform=True):
    """
    Function to Generate the topic model
    Input: modelName, data, custom vocabulary genrated , maximum number of words and topic
    """
    if modelName == 'NMF':
        vectorizer = TfidfVectorizer(stop_words=['got'], lowercase=False, vocabulary=vocabulary)
        fitdata = vectorizer.fit_transform(data)
        model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(fitdata)
    elif modelName == 'LDA':
        vectorizer = CountVectorizer(stop_words=['got'], lowercase=False, vocabulary=vocabulary)
        fitdata = vectorizer.fit_transform(data)
        model = LatentDirichletAllocation(n_components=no_topics, max_iter=10, learning_method='online',
                                          learning_offset=50., random_state=0).fit(fitdata)
    return (model)


def explore_topic(lda_model, topic_number, topn, output=True):
    """
     helper function to extract top n terms in topic model
    Input :  ldamodel, a topic number and top n vocabs of interest
    Output :  formatted list of the topn terms
    """
    terms = []
    for term, frequency in lda_model.show_topic(topic_number, topn=topn):
        terms += [term]
        if output:
            print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))

    return terms


def show_topics(vectorizer=vectorizer, lda_model=model, n_words=20):
    """
        function to extract top n terms in topic model
        Input :  ldamodel, a topic number and top n vocabs of interest
        Output :  formatted list of the topn terms
    """
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


