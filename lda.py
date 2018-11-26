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

np.random.seed(12345)

# n_docs is used to retrieve the proper number of docs from the database
n_docs = 5000
n_topics = 20

# Database address and credentials
conn_str = "host={} dbname={} user={} password={}".format("localhost", "itunes", "postgres", "linux116")
# Creating the postgresql connection object to use in pandas
conn = psycopg2.connect(conn_str)
# SQL query to retrieve n_docs number of reviews and concatenate the title and body for each review
sql_query = "SELECT concat(title, '. ', body) AS Review FROM public.app_reviewflat ORDER BY id ASC LIMIT " + str(n_docs)
# Hit the database and get the data
data = pd.read_sql(sql_query, con=conn)
# Choose only the reviews' text to be used in topic modeling
docs = data['review']

# Create a list of stop words to be removed from the corpus
stop = stopwords.words('english') + list(string.punctuation) + ["would", "app", "''", '``']

# A lemmatizer class that receives a document.
# The lemmatizer makes the document lowercase, tokenize the words and reduce them to their lemmas
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower()) if t not in stop]

# Create an object from the lemmazier
lemmatizer =LemmaTokenizer()

# Find corpus term frequency
frequency = defaultdict(int)
for doc in docs:
    for word in lemmatizer(doc):
        frequency[word] += 1

# Build dictionary
vocab = list({word for doc in docs for word in lemmatizer(doc) if frequency[word] > 2})
vocab.sort()

# Find document term frequency
dfs = defaultdict(int)
for word in vocab:
    for doc in docs:
        if word in doc:
            dfs[word] += 1
    dfs[word] = dfs[word]/n_docs

# Remove the words that appeared in more than 50% of the documents
vocab = [word for word, df in dfs.items() if df<0.5]

# Replace all words with their corresponding word_ids
corpus = {}
for doc_id, doc in enumerate(docs):
    words_list = lemmatizer(doc)
    corpus[doc_id] = [vocab.index(word) for word in words_list if word in vocab]

# An object of LDA class can be initialized by passing corpus, vocabulary, and number of topics.
# It can be run using run() method, passing number of samples and number of burn-in
class LDA(object):
    def __init__(self, corpus, vocab, n_topics):
        self.vocab = vocab
        # Nunber of topics
        self.n_topics = n_topics
        # Corpus: {doc_id: [word_1, word_1, word2, ...]}
        self.documents = corpus
        
        # Number of documents
        self.n_docs = len(self.documents)
        # Number of words in vocabulary
        self.n_words = len(self.vocab)
        
        # alpha is the hyperparameter for theta
        # It is used for smoothing the probability
        # alpha = a small value like 0.1
        self.alpha = np.ones(self.n_topics)*np.random.gamma(0.1, 1)
        # beta is the hyperparameter for phi
        # It is used for smoothing the probability
        # beta = a small value like 0.1
        self.beta = np.random.gamma(0.1, 1)
        # Create an empty matrix for theta
        self.theta = np.zeros((self.n_docs, self.n_topics))
        # Create an empty list to keep all theta estimates
        self.theta_estimates = []
        # Create an empty matrix for phi
        self.phi = np.zeros((self.n_topics, self.n_words))
        # Create an empty list to keep all phi estimates
        self.phi_estimates = []
        # Initialize the topic assignment dictionary
        # {doc_id: [topic_id assignment for each word in the doc]}
        # Create an empty matrix for topic-assignment
        self.topic_assignments = {}
        for doc_id, doc in self.documents.items():
            self.topic_assignments[doc_id] = [0]*len(doc)
        # Number of topics assigned to a word
        # The (i,j) entry of self.topic_likes_word is the number of times word j is assigned to topic i
        self.topic_likes_word = np.zeros((self.n_topics, self.n_words))
        # Number of topics assigned in a document
        # The (i,j) entry of self.doc_likes_topic is the number of words in document i assigned to topic j.
        self.doc_likes_topic = np.zeros((self.n_docs, self.n_topics))
        # Assignement count for each topic
        self.topic_count = np.zeros(self.n_topics)
        # Assignment count for each document = Length of each document
        self.doc_length = np.zeros(self.n_docs)
        # Randomly assign a topic a each word of each document
        for doc_id, doc in self.documents.items():
            for i, word_id in enumerate(doc):
                # Draw a random topic
                random_topic = random.randint(0, self.n_topics-1)
                # Assign the randomly drawn topic to ith word of the current document (doc_id)
                self.topic_assignments[doc_id][i] = random_topic
                # Update count variables according to the newly assigned topic to the word
                self.update_count_variables(doc_id, random_topic, word_id, 1)
    
    # Update the count variables using the value of change.
    # change = 1 can be used to increase the counts by 1.
    # change = -1 can be used to decrease the counts by 1.
    def update_count_variables(self, doc_id, topic_id, word_id, change):
        self.topic_likes_word[topic_id, word_id] += change
        self.doc_likes_topic[doc_id, topic_id] += change
        self.topic_count[topic_id] += change
        self.doc_length[doc_id] += change

    # Draw new random topic (for a specific word in a specific document) from full conditional distribution
    def assign_topics(self, doc_id, word_id, word_position_in_doc):
        # Find the current topic assignment of the word
        topic_id = self.topic_assignments[doc_id][word_position_in_doc]
        # Remove the current topic assigment by decreasing the count variables
        self.update_count_variables(doc_id, topic_id, word_id, -1)
        # Find the full conditional distribution
        full_conditional_dist = self.conditional(doc_id, word_id)
        # NOTE: 'full_conditional_dist' is a multinomial distribution with the length equal to the number of topics
        # Draw a new topic from the full conditional distribution
        # argmax returns the new topc_id that was randomly drawn from the distribution
        new_z = np.random.multinomial(1, full_conditional_dist).argmax()
        # Assign the randomly drawn topic to the word in word_position_in_doc of the current document (doc_id)
        self.topic_assignments[doc_id][word_position_in_doc] = new_z
        # Update count variables according to the newly assigned topic to the word
        self.update_count_variables(doc_id, new_z, word_id, 1)
        

    # Full conditional distribution
    def conditional(self, doc_id, word_id):
        # For more clarity, we calculate two parts of the distribution, separetly
        # Note that the ':' means 'all'
        part_1 = (self.topic_likes_word[:,word_id] + self.beta) / (self.topic_count + self.beta * self.n_words)
        part_2 = (self.doc_likes_topic[doc_id, :] + self.alpha) / (self.doc_length[doc_id] + np.sum(self.alpha))
        # Full conditional distribution
        full_conditional_dist = part_1 * part_2
        # To obtain probability
        full_conditional_dist /= np.sum(full_conditional_dist)
        return full_conditional_dist
        
    # Calculate Log-Likelihood value
    # Note that this Log-Likelihood is not directly used in our estimation process
    def log_likelihood(self):
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
        # Adjust alpha using Minka's fixed-point iteration (Minka 2000, latest revision 2012)
        numerator = 0
        denominator = 0
        for doc_id in range(self.n_docs):
            numerator += psi(self.doc_likes_topic[doc_id] + self.alpha) - psi(self.alpha)
            denominator += psi(np.sum(self.doc_likes_topic[doc_id] + self.alpha)) - psi(np.sum(self.alpha))
        # Update alpha
        self.alpha *= numerator / denominator

    def update_beta(self):
        # Adjust beta using Minka's fixed-point iteration (Minka 2000, latest revision 2012)
        numerator = 0
        denominator = 0
        for topic_id in range(self.n_topics):
            numerator += np.sum(psi(self.topic_likes_word[topic_id] + self.beta) - psi(self.beta))
            denominator += psi(np.sum(self.topic_likes_word[topic_id] + self.beta)) - psi(self.n_words * self.beta)
        # Update beta
        self.beta = (self.beta * numerator) / (self.n_words * denominator)

    def get_theta(self):
        # Create an empty matrix for theta
        theta = np.zeros((self.n_docs, self.n_topics))
        # Calculate theta for each pair of document-topic
        for doc_id in range(self.n_docs):
            for topic_id in range(self.n_topics):
                # For a better understanding, you can compare this with the formula for full conditional distribution
                theta[doc_id][topic_id] = (self.doc_likes_topic[doc_id][topic_id] + self.alpha[topic_id]) / (self.doc_length[doc_id] + np.sum(self.alpha))
        return theta
    
    def get_phi(self):
        # Create an empty matrix for phi
        phi = np.zeros((self.n_topics, self.n_words))
        for topic_id in range(self.n_topics):
            for word_id in range(self.n_words):
                # For a better understanding, you can compare this with the formula for full conditional distribution
                phi[topic_id][word_id] = (self.topic_likes_word[topic_id][word_id] + self.beta) / (self.topic_count[topic_id] + self.beta * self.n_words)
        return phi
    
    def topterms(self, n_terms=10):
        vec = np.atleast_2d(np.arange(0, self.n_words))
        topics = []
        for k in range(self.n_topics):
            probs = np.atleast_2d(self.phi[k,:])
            mat = np.append(probs,vec,0)
            sind = np.array([mat[:,i] for i in np.argsort(mat[0])]).T
            topics.append([self.vocab[int(sind[1,self.n_words - 1 - i])] for i in range(n_terms)])
        return topics
    
    # Print training data information
    def print_train_data_info(self):
        print("# of DOCS:", self.n_docs)
        print("# of TOPICS:", self.n_topics)
        print("# of words:", self.n_words)

    # Print information about the current state
    def print_current_state(self):
        print("\tLikelihood:", self.log_likelihood())
        print("\tAlpha:", end="")
        for topic_id in range(self.n_topics):
            print(" %.5f" % self.alpha[topic_id], end="")
        print("\n\tBeta: %.5f" % self.beta)
    
    # Gibbs sampler
    def run(self, nsamples, burnin):
        self.print_train_data_info()

        # Collapsed Gibbs Sampling
        print("INITIAL STATE")
        # Show the initial log likelihood based on the pure random topic assignment
        self.print_current_state()
        
        # Start the sampling process
        samples_after_burnin = 0
        for sample in range(nsamples):
            for doc_id, doc in self.documents.items():
                for position_in_doc, word_id in enumerate(doc):
                    # Draw new random topic from full conditional distribution
                    self.assign_topics(doc_id, word_id, position_in_doc)
            # Update alpha and beta values
            self.update_alpha()
            self.update_beta()
            print("SAMPLE #" + str(sample))
            self.print_current_state()
            # Find theta and phi after burn-in
            if sample > burnin:
                theta = self.get_theta()
                # Keep the theta estimate to find theta distribution
                self.theta_estimates.append(theta)
                self.theta += theta
                phi = self.get_phi()
                # Keep the phi estimate to find phi distribution
                self.phi_estimates.append(phi)
                self.phi += phi
                samples_after_burnin += 1
        # Find point estimates (mean) for theta
        self.theta /= samples_after_burnin
        # Find point estimates (mean) for phi
        self.phi /= samples_after_burnin
        return self.log_likelihood()

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
x = np.array(range(n_topics)) + 1
plt.bar(x, topic_prob, yerr=topic_std, align='center', alpha=0.5, color=colors)
plt.xticks(x, x)
plt.ylabel('Topic Probability')
plt.xlabel('Topic ID')
plt.title('Document #'+str(doc_id))
plt.savefig('filename.png', dpi=300)
plt.show()

topterms = lda.topterms()
topic_topwords = {}
for topic_id in range(n_topics):
    topic_topwords[topic_id] = topterms[topic_id]
topic_topwords_df = pd.DataFrame(topic_topwords)
topic_topwords_df.to_csv("topic_topwords.csv", sep='\t', encoding='utf-8')
