# Import all the libraries:
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

# downloading stopwords:
nltk.download('stopwords')

# Read all the pickle files that contain the inverted indexes and that was located on our bucket:
# Read all the indexes with the stemming to get the words stemmed:
title_index = pd.read_pickle('./title_postings/index_title.pkl')
body_index = pd.read_pickle('./body_postings/index_body.pkl')
anchor_index = pd.read_pickle('./anchor_postings/index_anchor.pkl')
# Read all the indexes without the stemming to get the words original:
title_index_wo_stem = pd.read_pickle('./title_posting_without_stem/index_title.pkl')
body_index_wo_stem = pd.read_pickle('./body_posting_without_stem/index_body.pkl')
anchor_index_wo_stem = pd.read_pickle('./anchor_posting_without_stem/index_anchor.pkl')
# Read the pageview and the pagerank from our bucket:
page_view = pd.read_pickle('./pageviews-202108-user.pkl')
page_rank = pd.read_pickle('./pagerank.pkl')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Read the bins that contain the posting lists and the indexes itself:
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

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

# Gather all the english stopwords together with no necessary words:
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became","how","do","you","when","why"]
all_stopwords = english_stopwords.union(corpus_stopwords)

# Tokenize all the sentences in the text:
def rearrange_words(words_txt):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    words = [word.group() for word in RE_WORD.finditer(words_txt.lower()) if word.group() not in all_stopwords]
    return words

# Tokenize all the sentences in the text with porter stemmer:
def stemmer(txt):
    tokens = rearrange_words(txt)
    st = PorterStemmer()
    tokens = [st.stem(i) for i in tokens]
    return tokens

# Binary ranking using both the titles and the anchors of articles:
def binary_rank(query, part, stemm=False):
    '''
    binary ranking documents according to number of unique query words
    appearing in title or anchor of document
    '''
    # Check if we want to use stemming:
    if stemm:
        # Use stemming at the tokenize on the query:
        query_list = stemmer(query)
        # Use on the title both index with stemming and path to the location stemmed in our bucket:
        if part == 'title':
            idx = title_index
            path = f'./title_postings/'
        # Use on the anchor both index with stemming and path to the location stemmed in our bucket:
        elif part == 'anchor':
            idx = anchor_index
            path = f'./anchor_postings/'
    # Check if we don't want to use stemming:
    else:
        # Use on the title both index without stemming and path to the location not stemmed in our bucket:
        if part == 'title':
            idx = title_index_wo_stem
            path = f'./title_posting_without_stem/'
        # Use on the anchor both index without stemming and path to the location not stemmed in our bucket:
        elif part == 'anchor':
            idx = anchor_index_wo_stem
            path = f'./anchor_posting_without_stem/'
        # Tokenize the query:
        query_list = rearrange_words(query)
    docs = {}
    q_words = np.unique(query_list)  # Distinct words
    for tok in q_words:  # For every word
        try:
            # Read the posting list of specific word that is located on our bucket and
            # with required index to read from him:
            pos_lst = read_posting_list(idx, tok, path)
            # For each doc id counts the number of tokens from the query that appear
            # in his text:
            for doc_id, tf in pos_lst:
                if doc_id in docs:
                    docs[doc_id] += 1
                else:
                    docs[doc_id] = 1

        except:
            continue

    ret = []
    # Sort all the docs by their value from top to down:
    sorted_lst = sorted(docs.items(), key=lambda doc_count: doc_count[1], reverse=True)
    # Go over all the sorted list:
    for doc_id, cnt in sorted_lst:
        # If doc id doesn't exist in our corpus anymore, then we can't retrieve its title:
        title = title_index.title[doc_id]
        # If it's exist in the corpus, put (doc id, count, title) in the list:
        if title is not None:
            ret.append((doc_id, cnt, title))
    return ret

# Cosine similarity using tf-idf on the body of articles:
def srh_body(query, stemm=False):
    '''
    running tfidf and cosine similarity in order to find top 100 documents
    for given query
    '''
    # Check if we want to use stemming:
    if stemm:
        # Use on the body both index with stemming and path to the location stemmed in our bucket:
        idx = body_index
        path = './body_postings/'
        # Use stemming at the tokenize on the query:
        query_list = stemmer(query)
    # Check if we don't want to use stemming:
    else:
        # Use on the body both index without stemming and path to the location stemmed in our bucket:
        idx = body_index_wo_stem
        path = './body_posting_without_stem/'
        # Tokenize the query:
        query_list = rearrange_words(query)

    norm_tf = {}
    num_tok = Counter(query_list) # Dictionary of the words in query (keys) and there time appearance (value)
    q_words = np.unique(query_list) # Distinct words
    ln_q = len(query_list) # Length of the query
    cos_sim = {}
    # For each word in the query:
    for tok in q_words:
        try:
            # Read the posting list of specific word that is located on our bucket and
            # with required index to read from him:
            pos_lst = read_posting_list(idx, tok, path)
            # For each doc id and tf:
            for doc_id, tf in pos_lst:
                # Normalizing term frequency -> fij/|# terms in document j|:
                norm_tf[(doc_id, tf)] = tf / idx.DL[doc_id]

        except:
            continue
        # For each doc id and tf:
        for doc_id, tf in pos_lst:
            # If we didn't already calculated it:
            if doc_id not in cos_sim:
                # We establish for each id the next calculation: ((tf in the query)*(idf on all of the doc)*(tf on all of the doc))/((size of query)*(size of the doc id)):
                cos_sim[doc_id] = ((num_tok[tok] / ln_q) * (norm_tf[(doc_id, tf)] * idx.idf_dict[tok])) / (idx.DL[doc_id] * ln_q)  # inner product of query and document
            else:
                # We add to the same doc id the additional tokens that didn't add before:
                cos_sim[doc_id] += ((num_tok[tok] / ln_q) * (norm_tf[(doc_id, tf)] * idx.idf_dict[tok])) / (idx.DL[doc_id] * ln_q)  # dividing by length of document and query

    lst_result = []
    # Go over all the docs and take there ids and cossine values:
    for doc_id, cos_val in cos_sim.items():
        # Take the title from doc id:
        title = title_index.title[doc_id]
        # if id in body doesn't exist anymore, then we can't retrieve its title
        if title is not None:
            lst_result.append((doc_id, cos_val, title))
    # Sort them by their cossine value and return the first 100:
    lst_res = sorted(lst_result, key=lambda x: x[1], reverse=True)
    return lst_res[:100]


def page_ranking(wiki_ids):
    '''
    returning pagerank values for given wiki ids
    '''
    res = []
    # Go over wiki ids:
    for wiki in wiki_ids:
        # If it exists in the page rank:
        if wiki in page_rank:
            # Return the pagerank grade:
            res.append(page_rank[wiki])
        else:
            # Return 0:
            res.append(0)
    return res


def page_viewing(wiki_ids):
    '''
    returning pageviews for given wiki ids
    '''
    res = []
    # Go over wiki ids:
    for wiki in wiki_ids:
        # If it exists in the page view:
        if wiki in page_view:
            # Return the pageview grade:
            res.append(page_view[wiki])
        else:
            # Return 0:
            res.append(0)
    return res


def bm25_score(query, inverted, N, path, stemm=False):

    # Check if we want to use stemming:
    if stemm:
        # Use on the body both index with stemming and path to the location stemmed in our bucket:
        idx = body_index
        tokens = stemmer(query)
    # Check if we don't want to use stemming:
    else:
        # Use on the body both index without stemming and path to the location not stemmed in our bucket:
        idx = body_index_wo_stem
        tokens = rearrange_words(query)

    tmp = defaultdict(int)
    BM25_score = []

    # If the index is body:
    if inverted == idx:
        # Calculate the corpus size:
        size_corpus = idx.corpus_len
        sum_sc = 2028630613
        # Avg of length of the docs:
        avg_doc_len = sum_sc / size_corpus
    # If the index is not body:
    else:
        # Calculate for all docs: (all the doc length / num of docs):
        avg_doc_len = sum(inverted.DL.values()) / inverted.corpus_len

    # For each token:
    for tok in tokens:
        # Read his posting list according to the index:
        post_lst = read_posting_list(inverted, tok, path)
        # For each doc id and frequency:
        for doc_id, freq in post_lst:
            # Calculate: (idf of the token)*c(w,d)*(k+1):
            numerator = idx.idf_dict[tok] * freq * (1.5 + 1)
            # Calculate: c(w,d)+k(1-b+b*(|d|/avdl))
            denominator = freq + 1.5 * (1 - 0.75 + 0.75 * inverted.DL[doc_id] / avg_doc_len)
            # Do sum over all the words that is the conjuction between the docs and the query:
            tmp[doc_id] += (numerator / denominator)

    # Sort the lst by their values:
    sort_tmp = sorted(tmp.items(), key=lambda sc: sc[1], reverse=True)
    # Return the N docs that is the best:
    topN = sort_tmp[:N]

    #Go over each doc id and return (doc id, bm25 val,title):
    for doc in topN:
        BM25_score.append((doc[0], float(doc[1]), title_index.title[doc[0]]))

    return BM25_score


def srh(query, stemm=False):
    '''
    Searches in body and title(both cossine similarity for short queries and bm25 for long queries);
    Sum all together with weights on all results to keep only the top 100 results.
    Then runs pagerank and pageview on these wiki_ids and reorganizes
    results for a secondary ranking before retrieval concludes.
    Parameters
    __________
    query: str.
    query to search.
    stemming: boolean.
    use stemming or not.
    '''

    # Check if we want to use stemming:
    if stemm:
        # Stem with tokenize the query:
        tokens = stemmer(query)
    # If we don't want to use stemming:
    else:
        # Tokenize the query:
        tokens = rearrange_words(query)

    # If the query length is 1 or 2:
    if len(tokens) == 1 or len(tokens) == 2:
        # Run search method on the body that we created to pick only first 100 on the body:
        body = srh_body(query, stemm)
        # Run search method on the title that we created to get only first 100 on the title:
        top100_title = binary_rank(query, 'title', stemm)[:100]
        # Average of all vals in the title:
        tit_mean = np.mean([i[1] for i in top100_title])
        # Take only those that are higher than the average at the title:
        above_tit = [i for i in top100_title if i[1] >= tit_mean]

        # weighting the title according to index (the place each doc was ranked in search retrieval)
        # we weight the title from the biggest weight to the smallest weight:
        title_weighted_1 = list(map(lambda x: (x[0], (100 - above_tit.index(x)) * 0.85), above_tit))
        # Put it into the dictionary that we created:
        c = Counter(dict(title_weighted_1))

        # weighting the body according to index (the place each doc was ranked in search retrieval)
        # we weight the body from the biggest weight to the smallest weight:
        body_weighted_1 = list(map(lambda x: (x[0], (100 - body.index(x)) * 0.1), body))
        # Update the dictionary for each doc:
        c.update(dict(body_weighted_1))

        # Sort all the dictionary by the value:
        scores_weighted = sorted(c.most_common(), key=lambda x: x[1], reverse=True)
        # Take the keys (doc id):
        docs = list(dict(scores_weighted).keys())
        # Run pagerank on documents:
        pager = list(zip(docs, np.array(page_ranking(docs))))
        # Sort all the docs by their pagerank
        rerank1 = sorted(pager, key=lambda x: x[1], reverse=True)
        # weighting the pagerank according to index (the place each doc was ranked)
        # we weight the pagerank from the biggest weight to the smallest weight:
        rank_weighted_1 = list(map(lambda x: (x[0], (100 - rerank1.index(x)) * 0.05), rerank1))
        # Update the dictionary for each doc:
        c.update(dict(rank_weighted_1))
        # Sort the dictionary again according to pagerank and take only the 80 docs that are highest:
        scores_weighted_1 = sorted(c.most_common(), key=lambda x: x[1], reverse=True)[:80]
        # Take the keys (doc id):
        docs1 = list(dict(scores_weighted_1).keys())
        # run pageview on documents:
        pagev = list(zip(docs1, np.array(page_viewing(docs1))))
        # Sort them according the pageview:
        rerank2 = sorted(pagev, key=lambda x: x[1], reverse=True)
        # Return the docs after the reranking:
        return [(i, title_index.title[i]) for i, _ in rerank2]

    # If the query length more than 3 (include):
    else:
        # Run search method on the body that we created to pick only first 100 on the body:
        body = srh_body(query, stemm)
        # Run search method on the title that we created to get only first 100 on the title:
        top100_title = binary_rank(query, 'title', stemm)[:100]
        # Average of all vals in the title:
        tit_mean = np.mean([i[1] for i in top100_title])
        # Take only those that are higher than the average at the title:
        above_tit = [i for i in top100_title if i[1] >= tit_mean]

        # If we want to use stemming:
        if stemm:
            # Use on the title both index with stemming and path to the location stemmed in our bucket:
            idx = title_index
            path = './title_postings/'
        # If  we don't want to use stemming:
        else:
            # Use on the title both index without stemming and path to the location not stemmed in our bucket:
            idx = title_index_wo_stem
            path = './title_posting_without_stem/'

        # weighting the title according to index (the place each doc was ranked in search retrieval)
        # we weight the title from the biggest weight to the smallest weight:
        title_weighted_1 = list(map(lambda x: (x[0], (100 - above_tit.index(x)) * 0.05), above_tit))
        # Update the dictionary for each doc his val:
        c = Counter(dict(title_weighted_1))

        # Calculate BM25 on the title and return the top 100 docs:
        bm25_title = bm25_score(query, idx, 100, path, stemm)
        # weighting the title according to index (the place each doc was ranked in bm25)
        # we weight the title from the biggest weight to the smallest weight:
        title_weighted_2 = list(map(lambda x: (x[0], (100 - bm25_title.index(x)) * 0.2), bm25_title))
        # Update the dictionary for each doc his val:
        c.update(dict(title_weighted_2))

        # If we want to use stemming:
        if stemm:
            # Use on the body both index with stemming and path to the location stemmed in our bucket:
            idx = body_index
            path = './body_postings/'
        # If  we don't want to use stemming:
        else:
            # Use on the body both index without stemming and path to the location not stemmed in our bucket:
            idx = body_index_wo_stem
            path = './body_posting_without_stem/'

        # Calculate BM25 on the title and return the top 100 docs:
        bm25_body = bm25_score(query, idx, 100, path, stemm)

        # weighting the body according to index (the place each doc was ranked in bm25)
        # we weight the title from the biggest weight to the smallest weight:
        body_weighted_2 = list(map(lambda x: (x[0], (100 - bm25_body.index(x)) * 0.3), bm25_body))
        # Update the dictionary for each doc his val:
        c.update(dict(body_weighted_2))

        # weighting the body according to index (the place each doc was ranked in search retrieval)
        # we weight the body from the biggest weight to the smallest weight:
        body_weighted_1 = list(map(lambda x: (x[0], (100 - body.index(x)) * 0.45), body))
        # Update the dictionary for each doc his val:
        c.update(dict(body_weighted_1))
        # Sort the weights according to the values in the dictionary and return the 100 docs:
        scores_weighted = sorted(c.most_common(), key=lambda x: x[1], reverse=True)[:100]
        # Take the keys (doc id):
        docs = list(dict(scores_weighted).keys())
        # Run pagerank on documents: and keeping only top 80
        pager = list(zip(docs, np.array(page_ranking(docs))))
        #Sort the doc id according to pagerank and keep only top 80:
        rerank1 = sorted(pager, key=lambda x: x[1], reverse=True)[:80]
        # Take the keys (doc id):
        docs1 = list(dict(rerank1).keys())
        # Run pageview on documents:
        pagev = list(zip(docs1, np.array(page_viewing(docs1))))
        # Sort the doc id by their pageview and keep only top 60:
        rerank2 = sorted(pagev, key=lambda x: x[1], reverse=True)[:60]
        # Return the doc id and the title of the relevant doc:
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
    # Call function named srh without stemming:
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
    # Call the function called srh_body without stemming:
    res = srh_body(query, False)
    # Return the doc id and title as required:
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
    # Call the function called binary rank without stemming with title index:
    res = binary_rank(query, 'title', False)
    # Return the doc id and title as required:
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
    # Call the function called binary rank without stemming with anchor index:
    res = binary_rank(query, 'anchor', False)
    # Return the doc id and title as required:
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
    # Call the function called page ranking on all the docs in the corpus:
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
    # Call the function called page viewing on all the docs in the corpus:
    res = page_viewing(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
