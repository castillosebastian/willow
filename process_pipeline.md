# extract
- 1. identify news articles:
  - 1. by url with *cosine similarity*
    - 1. confirmation by content *
- e_metrics:
  - datetime e, 
  - urls
  - narco_articles / source size
# transform
- 1. Consolidate document by news_itemp adding
  - NER 
  - summary
  - topic
  - embedings (clean text first)
    - news_text_embeddings
    - NER_concaten_embeddings
# save_load 
- mongodb