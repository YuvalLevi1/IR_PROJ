# IR_PROJ
Informaiom Retrieval Project Repository

This repo contains:
1. Creating_the_index.ipynb - The code to create the indexes used in this project and saving them in our bucket.
2. inverted_index_gcp.py - Inverted index class that helps to build the indexes and MultiFileReader and MultiFileWriter classes for writing and reading the data from the index.
3. search_frontend.py - contains all of the main functions of our retrievel engine. this file also contains the helper functions for the main functions like binary ranking for title and anchor and cosine similarity for the body using tf-idf and BM25 score function.
4. index files.txt - All index files in our bucket.

## The main functions of our retrieval engine

### Search Body
--------------------------------
Returns up to a 100 search results for the query using TF-IDF and Cosine similarity of the body of articles.

### Search Title
--------------------------------
Returns ALL search results that contain a query word
in the title of articles using binary ranking. For example, a document 
with a title that matches two distinct query words will be ranked before a 
document with a title that matches only one distinct query word, 
regardless of the number of times the term appeared in the title.

### Search anchor
--------------------------------
Returns ALL search results that contain a query word
in the anchor text of articles, ordered in descending order of the
number of query words that appear in anchor text linking to the page.For example, 
a document with a anchor text that matches two distinct query words will 
be ranked before a document with anchor text that matches only one 
distinct query word, regardless of the number of times the term appeared 
in the anchor text

### Search
--------------------------------
Searches in body and title(both cossine similarity for short queries and bm25 for long queries);
Sum all together with weights on all results to keep only the top 100 results.
Then runs pagerank and pageview on these wiki_ids and reorganizes
results for a secondary ranking before retrieval concludes.

### Get PageRank
--------------------------------
Returns PageRank values for a list of provided wiki article IDs.

### Get PageView
--------------------------------
Returns the number of page views that each of the provide wiki articles
had in August 2021
