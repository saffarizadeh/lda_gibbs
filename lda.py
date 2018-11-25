#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:36:38 2018

@author: Kambiz Saffarizadeh
"""
import psycopg2
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from scipy.special import gammaln, psi
import random
from collections import defaultdict
import matplotlib.pyplot as plt

n_docs = 5000
n_topics = 20

conn_str = "host={} dbname={} user={} password={}".format("localhost", "___", "___", "___")
conn = psycopg2.connect(conn_str)

sql_query = "SELECT concat(title, '. ', body) AS Review FROM public.app_reviewflat ORDER BY id ASC LIMIT " + str(n_docs)
data = pd.read_sql(sql_query, con=conn)
docs = data['review']


stop = stopwords.words('english') + list(string.punctuation) + ["would", "app", "''", '``']

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower()) if t not in stop]

lemmatizer =LemmaTokenizer()

# find corpus term frequency
frequency = defaultdict(int)
for doc in docs:
    for word in lemmatizer(doc):
        frequency[word] += 1

# build dictionary
vocab = list({word for doc in docs for word in lemmatizer(doc) if frequency[word] > 2})
vocab.sort()

dfs = defaultdict(int)
for word in vocab:
    for doc in docs:
        if word in doc:
            dfs[word] += 1
    dfs[word] = dfs[word]/n_docs

vocab = [word for word, df in dfs.items() if df<0.5]

# Replace all words with their corresponding word_ids
corpus = {}
for doc_id, doc in enumerate(docs):
    words_list = lemmatizer(doc)
    corpus[doc_id] = [vocab.index(word) for word in words_list if word in vocab]

class LDA(object):
    def __init__(self, corpus, vocab, n_topics):
        self.vocab = vocab
        # NUMBER OF TOPICS
        self.n_topics = n_topics
        # Corpus: {DocID: [WordID1, WordID1, WordID2, ...]}
        self.documents = corpus
        
        # NUMBER OF DOCUMENTS
        self.n_docs = len(self.documents)
        # NUMBER OF WORDS IN VOCABULARY
        self.n_words = len(self.vocab)
        
        # alpha = a small value like 0.1
        self.alpha = np.ones(self.n_topics)*np.random.gamma(0.1, 1)
        # beta = a small value like 0.1
        self.beta = np.random.gamma(0.1, 1)
        # SPACE FOR THETA MATRIX WITH 0s
        self.theta = np.zeros((self.n_docs, self.n_topics))
        self.theta_estimates = []
        # SPACE FOR PHI MATRIX WITH 0s
        self.phi = np.zeros((self.n_topics, self.n_words))
        self.phi_estimates = []


        # Initialize the topic assignment dictionary.
        # MAKE SPACE FOR TOPIC-ASSIGNMENT MATRICES WITH 0s
        # {INDEX OF DOC: [TOPIC ASSIGNMENT FOR EACH WORD]}
        # key-value pairs of form (m,i):z
        self.topic_assignments = {}
        for doc_id, doc in self.documents.items():
            self.topic_assignments[doc_id] = [0]*len(doc)
        # NUMBER OF TOPICS ASSIGNED TO A WORD
        # The (i,j) entry of self.nzw is the number of times term j is assigned to topic i.
        # nzw
        self.topic_likes_word = np.zeros((self.n_topics, self.n_words))
        # NUMBER OF TOPICS ASSIGNED IN A DOCUMENT
        # The (i,j) entry of self.nmz is the number of words in document i assigned to topic j.
        # nmz
        self.doc_likes_topic = np.zeros((self.n_docs, self.n_topics))
        # ASSIGNMENT COUNT FOR EACH TOPIC
        # nz
        self.topic_count = np.zeros(self.n_topics)
        # ASSIGNMENT COUNT FOR EACH DOCUMENT = LENGTH OF DOCUMENT
        self.doc_length = np.zeros(self.n_docs)
        
        # RANDOMLY ASSIGN TOPIC TO EACH WORD
        for doc_id, doc in self.documents.items():
            for i, word_id in enumerate(doc):
                # RANDOM TOPIC ASSIGNMENT
                random_topic = random.randint(0, self.n_topics-1)
                # RANDOMLY ASSIGN TOPIC TO EACH WORD
                self.topic_assignments[doc_id][i] = random_topic
                self.topic_likes_word[random_topic, word_id] += 1
                self.doc_likes_topic[doc_id, random_topic] += 1
                self.topic_count[random_topic] += 1
                self.doc_length[doc_id] += 1

    # DROW TOPIC SAMPLE FROM FULL-CONDITIONAL DISTRIBUTION
    def assignTopics(self, doc_id, word_id, word_position_in_doc):
        # TOPIC ASSIGNMENT OF WORDS FOR EACH DOCUMENT
        topic_id = self.topic_assignments[doc_id][word_position_in_doc]
        self.topic_likes_word[topic_id, word_id] -= 1
        self.doc_likes_topic[doc_id, topic_id] -= 1
        self.topic_count[topic_id] -= 1
        self.doc_length[doc_id] -= 1
        
        probFullCond = self.conditional(doc_id, word_id)
        # NOTE: 'prFullCond' is MULTINOMIAL DISTRIBUTION WITH THE LENGTH EQUAL NUMBER OF TOPICS
        # RANDOM SAMPLING FROM FULL-CONDITIONAL DISTRIBUTION
        # argmax returns the new topc_id that was randomly drawn from the distribution
        new_z = np.random.multinomial(1, probFullCond).argmax()
        self.topic_assignments[doc_id][word_position_in_doc] = new_z
        self.topic_likes_word[new_z, word_id] += 1
        self.doc_likes_topic[doc_id, new_z] += 1
        self.topic_count[new_z] += 1
        self.doc_length[doc_id] += 1

    # FULL-CONDITIONAL DISTRIBUTION
    def conditional(self, doc_id, word_id):
        dist_l = (self.doc_likes_topic[doc_id, :] + self.alpha) / (self.doc_length[doc_id] + np.sum(self.alpha))
        dist_r = (self.topic_likes_word[:,word_id] + self.beta) / (self.topic_count + self.beta * self.n_words)
        # FULL-CONDITIONAL DISTRIBUTION
        full_conditional_dist = dist_l * dist_r
        # TO OBTAIN PROBABILITY
        full_conditional_dist /= np.sum(full_conditional_dist)
        return full_conditional_dist
        
    # FIND (JOINT) LOG-LIKELIHOOD VALUE
    def LogLikelihood(self):
        ll = 0
        # log p(w|z,\beta)
        for topic_id in range(self.n_topics):
            ll += gammaln(self.n_words * self.beta)
            ll -= self.n_words * gammaln(self.beta)
            ll += np.sum(gammaln(self.topic_likes_word[topic_id] + self.beta))
            ll -= gammaln(np.sum(self.topic_likes_word[topic_id] + self.beta))
        # log p(z|\alpha)
        for doc_id, doc in self.documents.items():
            ll += gammaln(np.sum(self.alpha))
            ll -= np.sum(gammaln(self.alpha))
            ll += np.sum(gammaln(self.doc_likes_topic[doc_id] + self.alpha))
            ll -= gammaln(np.sum(self.doc_likes_topic[doc_id] + self.alpha))
        return ll
    
    def update_alpha(self):
        # ADJUST ALPHA AND BETA BY USING MINKA'S FIXED-POINT ITERATION
        numerator = 0
        denominator = 0
        for doc_id in range(self.n_docs):
            numerator += psi(self.doc_likes_topic[doc_id] + self.alpha) - psi(self.alpha)
            denominator += psi(np.sum(self.doc_likes_topic[doc_id] + self.alpha)) - psi(np.sum(self.alpha))
        # UPDATE ALPHA
        self.alpha *= numerator / denominator

    def update_beta(self):
        numerator = 0
        denominator = 0
        for topic_id in range(self.n_topics):
            numerator += np.sum(psi(self.topic_likes_word[topic_id] + self.beta) - psi(self.beta))
            denominator += psi(np.sum(self.topic_likes_word[topic_id] + self.beta)) - psi(self.n_words * self.beta)
        # UPDATE BETA
        self.beta = (self.beta * numerator) / (self.n_words * denominator)

    def get_theta(self):
        # SPACE FOR THETA
        th = np.zeros((self.n_docs, self.n_topics))
        for doc_id in range(self.n_docs):
            for topic_id in range(self.n_topics):
                th[doc_id][topic_id] = (self.doc_likes_topic[doc_id][topic_id] + self.alpha[topic_id]) / (self.doc_length[doc_id] + np.sum(self.alpha))
        return th
    
    def get_phi(self):
        # SPACE FOR PHI
        ph = np.zeros((self.n_topics, self.n_words))
        for topic_id in range(self.n_topics):
            for word_id in range(self.n_words):
                ph[topic_id][word_id] = (self.topic_likes_word[topic_id][word_id] + self.beta) / (self.topic_count[topic_id] + self.beta * self.n_words)
        return ph
    
    def topterms(self, n_terms=10):
        vec = np.atleast_2d(np.arange(0, self.n_words))
        topics = []
        for k in range(self.n_topics):
            probs = np.atleast_2d(self.phi[k,:])
            mat = np.append(probs,vec,0)
            sind = np.array([mat[:,i] for i in np.argsort(mat[0])]).T
            topics.append([self.vocab[int(sind[1,self.n_words - 1 - i])] for i in range(n_terms)])
        return topics
    
    # GIBBS SAMPLER KERNEL
    def run(self, nsamples, burnin):
        # PRINT TRAINING DATA INFORMATION
        print("# of DOCS:", self.n_docs)
        print("# of TOPICS:", self.n_topics)
        print("# of words:", self.n_words)
        # COLLAPSED GIBBS SAMPLING
        print("INITIAL STATE")
        # FIND (JOINT) LOG-LIKELIHOOD
        print("\tLikelihood:", self.LogLikelihood())
        print("\tAlpha:", end="")
        for topic_id in range(self.n_topics):
            print(" %.5f" % self.alpha[topic_id], end="")
        print("\n\tBeta: %.5f" % self.beta)
        samples = 0
        for s in range(nsamples):
            for doc_id, doc in self.documents.items():
                for position_in_doc, word_id in enumerate(doc):
                    # DROW TOPIC SAMPLE FROM FULL-CONDITIONAL DISTRIBUTION
                    self.assignTopics(doc_id, word_id, position_in_doc)
            # UPDATE ALPHA AND BETA VALUES
            self.update_alpha()
            self.update_beta()
            lik = self.LogLikelihood()
            print("SAMPLE #" + str(s))
            print("\tLikelihood:", lik)
            print("\tAlpha:", end="")
            for topic_id in range(self.n_topics):
                print(" %.5f" % self.alpha[topic_id], end="")
            print("\n\tBeta: %.5f" % self.beta)
            # FIND PHI AND THETA AFTER BURN-IN POINT
            if s > burnin:
                theta = self.get_theta()
                self.theta_estimates.append(theta)
                self.theta += theta
                phi = self.get_phi()
                self.phi_estimates.append(phi)
                self.phi += phi
                samples += 1
        # AVERAGING GIBBS SAMPLES OF THETA
        self.theta /= samples
        # AVERAGING GIBBS SAMPLES OF PHI
        self.phi /= samples
        return lik

lda = LDA(corpus, vocab, n_topics)
lda.run(200, 100)
lda.topterms()
docs_topics = np.array(lda.theta_estimates)


doc_id = 3
topic_prob = docs_topics[:, doc_id, :].mean(axis=0)
topic_std = docs_topics[:, doc_id, :].std(axis=0)
colors = plt.cm.get_cmap('hsv', n_topics)
colors = [colors(topic) for topic in range(n_topics)]
plt.axhline(y=1)
plt.bar(range(n_topics), topic_prob, yerr=topic_std, align='center', alpha=0.5, color=colors)
plt.xticks(range(n_topics), range(n_topics))
plt.ylabel('Topic Probability')
plt.title('Document #'+str(doc_id))
plt.show()
