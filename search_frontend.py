from flask import Flask, request, jsonify
from inverted_index_gcp import *
import pandas as pd
import pickle
import json
import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import hashlib
from contextlib import closing
import numpy as np
from collections import Counter, defaultdict
import math

# downloading stopwords
nltk.download('stopwords')

title_index = pd.read_pickle('./title_postings/index_title.pkl')
body_index = pd.read_pickle('./body_postings/index_body.pkl')
anchor_index = pd.read_pickle('./anchor_postings/index_anchor.pkl')

title_index_wo_stem = pd.read_pickle('./title_posting_without_stem/index_title.pkl')
body_index_wo_stem = pd.read_pickle('./body_posting_without_stem/index_body.pkl')
anchor_index_wo_stem = pd.read_pickle('./anchor_posting_without_stem/index_anchor.pkl')

page_view = pd.read_pickle('./pageviews-202108-user.pkl')
page_rank = pd.read_pickle('./pagerank.pkl')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


def read_posting_list(inverted, w, file_name):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    locs = [(file_name + loc[0], loc[1]) for loc in locs] # fixing posting_locs so we know from what folder to read the bin
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list


english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became","how","do","you","when","why"]
all_stopwords = english_stopwords.union(corpus_stopwords)


def rearrange_words(words_txt):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    words = [word.group() for word in RE_WORD.finditer(words_txt.lower()) if word.group() not in all_stopwords]
    return words


def stemmer(txt):
    tokens = rearrange_words(txt)
    st = PorterStemmer()
    tokens = [st.stem(i) for i in tokens]
    return tokens

def binary_rank(query, part, stemm=False):
    '''
    binary ranking documents according to number of unique query words
    appearing in title or anchor of document

    '''
    if stemm:
        query_list = stemmer(query)
        if part == 'title':
            idx = title_index
            path = f'./title_postings/'
        elif part == 'anchor':
            idx = anchor_index
            path = f'./anchor_postings/'
    else:
        if part == 'title':
            idx = title_index_wo_stem
            path = f'./title_posting_without_stem/'
        elif part == 'anchor':
            idx = anchor_index_wo_stem
            path = f'./anchor_posting_without_stem/'
        query_list = rearrange_words(query)

    docs = {}
    # Tokenize to words
    q_words = np.unique(query_list)  # Distinct words
    for tok in q_words:  # For every word
        try:
            pos_lst = read_posting_list(idx, tok, path)
            # for each doc id counts the number of tokens from the query that appear
            for doc_id, tf in pos_lst:
                if doc_id in docs:
                    docs[doc_id] += 1
                else:
                    docs[doc_id] = 1

        except:
            continue

    ret = []
    sorted_lst = sorted(docs.items(), key=lambda doc_count: doc_count[1], reverse=True)
    for doc_id, cnt in sorted_lst:
        title = title_index.title[doc_id]  # if id in anchor doesn't exist anymore, then we can't retrieve its title
        if title is not None:
            ret.append((doc_id, cnt, title))
    return ret


def srh_body(query, stemm=False):
    '''
    running tfidf and cosine similarity in order to find top 100 documents
    for given query
    '''
    if stemm:
        idx = body_index
        path = './body_postings/'
        query_list = stemmer(query)
    else:
        idx = body_index_wo_stem
        path = './body_posting_without_stem/'
        query_list = rearrange_words(query)

    norm_tf = {}

    num_tok = Counter(query_list)
    q_words = np.unique(query_list) # Distinct words
    ln_q = len(query_list)
    cos_sim = {}
    for tok in q_words:
        try:
            pos_lst = read_posting_list(idx, tok, path)
            for doc_id, tf in pos_lst:
                # normalizing term frequency -> fij/|# terms in document j|
                norm_tf[(doc_id, tf)] = tf / idx.DL[doc_id]

        except:
            continue

        for doc_id, tf in pos_lst:
            if doc_id not in cos_sim:
                cos_sim[doc_id] = ((num_tok[tok] / ln_q) * (norm_tf[(doc_id, tf)] * idx.idf_dict[tok])) / (idx.DL[doc_id] * ln_q)  # inner product of query and document
            else:
                cos_sim[doc_id] += ((num_tok[tok] / ln_q) * (norm_tf[(doc_id, tf)] * idx.idf_dict[tok])) / (idx.DL[doc_id] * ln_q)  # dividing by length of document and query

    lst_result = []
    for doc_id, cos_val in cos_sim.items():
        title = title_index.title[doc_id]
        # if id in body doesn't exist anymore, then we can't retrieve its title
        if title is not None:
            lst_result.append((doc_id, cos_val, title))
    lst_res = sorted(lst_result, key=lambda x: x[1], reverse=True)
    return lst_res[:100]


def page_ranking(wiki_ids):
    '''
    returning pagerank values for given wiki ids
    '''
    res = []
    for wiki in wiki_ids:
        if wiki in page_rank:
            res.append(page_rank[wiki])
        else:
            res.append(0)
    return res


def page_viewing(wiki_ids):
    '''
    returning pageviews for given wiki ids
    '''
    res = []
    for wiki in wiki_ids:
        if wiki in page_view:
            res.append(page_view[wiki])
        else:
            res.append(0)
    return res


def bm25_score(query, inverted, N, path, stemm=False):
    score = 0.0

    if stemm:
        idx = body_index
        tokens = stemmer(query)
    else:
        idx = body_index_wo_stem
        tokens = rearrange_words(query)

    tmp = defaultdict(int)
    BM25_score = []

    if inverted == idx:
        size_corpus = idx.corpus_len
        sum_sc = 2028630613
        avg_doc_len = sum_sc / size_corpus
    else:
        avg_doc_len = sum(inverted.DL.values()) / inverted.corpus_len

    for tok in tokens:
        post_lst = read_posting_list(inverted, tok, path)
        for doc_id, freq in post_lst:
            numerator = idx.idf_dict[tok] * freq * (1.5 + 1)
            denominator = freq + 1.5 * (1 - 0.75 + 0.75 * inverted.DL[doc_id] / avg_doc_len)
            tmp[doc_id] += (numerator / denominator)

    sort_tmp = sorted(tmp.items(), key=lambda sc: sc[1], reverse=True)
    topN = sort_tmp[:N]

    for doc in topN:
        BM25_score.append((doc[0], float(doc[1]), title_index.title[doc[0]]))

    return BM25_score


def srh(query, stemm=False):
    '''
    Searches in body, anchor and title; and weight averages all results
    together to keep only the top 100 results.
    Then runs pagerank and pageview on these wiki_ids and reorganizes
    results in a secondary ranking before retrieval concludes.
    Parameters
    __________
    query: str.
    query to search.
    '''
    if stemm:
        tokens = stemmer(query)
    else:
        tokens = rearrange_words(query)

    if len(tokens) == 1 or len(tokens) == 2:
        # running search methods, picking only first 100
        body = srh_body(query, stemm)
        top100_title = binary_rank(query, 'title', stemm)[:100]
        tit_mean = np.mean([i[1] for i in top100_title])
        above_tit = [i for i in top100_title if i[1] >= tit_mean]

        # weighting according to index in retrieval
        title_weighted_1 = list(map(lambda x: (x[0], (100 - above_tit.index(x)) * 0.85), above_tit))
        c = Counter(dict(title_weighted_1))

        body_weighted_1 = list(map(lambda x: (x[0], (100 - body.index(x)) * 0.1), body))
        c.update(dict(body_weighted_1))
        scores_weighted = sorted(c.most_common(), key=lambda x: x[1], reverse=True)
        docs = list(dict(scores_weighted).keys())  # running pagerank on documents and keeping only top 80
        pager = list(zip(docs, np.array(page_ranking(docs))))
        rerank1 = sorted(pager, key=lambda x: x[1], reverse=True)
        rank_weighted_1 = list(map(lambda x: (x[0], (100 - rerank1.index(x)) * 0.05), rerank1))
        c.update(dict(rank_weighted_1))
        scores_weighted_1 = sorted(c.most_common(), key=lambda x: x[1], reverse=True)[:80]
        docs1 = list(dict(scores_weighted_1).keys())  # running pageview on documents and keeping only top 60
        pagev = list(zip(docs1, np.array(page_viewing(docs1))))
        rerank2 = sorted(pagev, key=lambda x: x[1], reverse=True)
        return [(i, title_index.title[i]) for i, _ in rerank2]
    else:
        body = srh_body(query, stemm)  # running search methods, picking only first 100
        top100_title = binary_rank(query, 'title', stemm)[:100]
        tit_mean = np.mean([i[1] for i in top100_title])
        above_tit = [i for i in top100_title if i[1] >= tit_mean]

        if stemm:
            idx = title_index
            path = './title_postings/'
        else:
            idx = title_index_wo_stem
            path = './title_posting_without_stem/'

        bm25_body = bm25_score(query, idx, 100, path, stemm)

        title_weighted_1 = list(map(lambda x: (x[0], (100 - above_tit.index(x)) * 0.05), above_tit))  # weighting according to index in retrieval
        c = Counter(dict(title_weighted_1))

        # bm25_body = bm25_score(query, idx, 100, path, stemm)
        body_weighted_2 = list(map(lambda x: (x[0], (100 - bm25_body.index(x)) * 0.2), bm25_body))
        c.update(dict(body_weighted_2))

        if stemm:
            idx = body_index
            path = './body_postings/'
        else:
            idx = body_index_wo_stem
            path = './body_posting_without_stem/'

        bm25_body = bm25_score(query, idx, 100, path, stemm)

        # bm25_body = bm25_score(query, idx, 100, path, stemm)
        body_weighted_2 = list(map(lambda x: (x[0], (100 - bm25_body.index(x)) * 0.3), bm25_body))
        c.update(dict(body_weighted_2))


        body_weighted_1 = list(map(lambda x: (x[0], (100 - body.index(x)) * 0.45), body))
        c.update(dict(body_weighted_1))
        scores_weighted = sorted(c.most_common(), key=lambda x: x[1], reverse=True)[:100]
        docs = list(dict(scores_weighted).keys())  # running pagerank on documents and keeping only top 80
        pager = list(zip(docs, np.array(page_ranking(docs))))
        rerank1 = sorted(pager, key=lambda x: x[1], reverse=True)[:80]
        docs1 = list(dict(rerank1).keys())  # running pageview on documents and keeping only top 60
        pagev = list(zip(docs1, np.array(page_viewing(docs1))))
        rerank2 = sorted(pagev, key=lambda x: x[1], reverse=True)[:60]

        return [(i, title_index.title[i]) for i, _ in rerank2]


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = srh(query, False)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = srh_body(query, False)
    res = [(tup[0], tup[2]) for tup in res]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = binary_rank(query, 'title', False)
    res = [(tup[0], tup[2]) for tup in res]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = binary_rank(query, 'anchor', False)
    res = [(tup[0], tup[2]) for tup in res]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = page_ranking(wiki_ids)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = page_viewing(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
