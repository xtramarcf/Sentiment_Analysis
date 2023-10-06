from eventregistry import *


# Method for getting articles with the content and the sentiment.
# Requires the two arguments company, for the name of the company
# and the archive_search.
# allowUseOfArchive=False will only search for data from the last 31 days.
# max_items will limit the number of found articles.
# min_sentiment and max_sentiment allows to get articles within a 'sentiment-range'.
# This is very useful for evaluating and testing the model predictions.
def get_articles(company, archive_search, min_sentiment, max_sentiment, max_items=100):
    er = EventRegistry(apiKey="6e9198fc-7f8b-42a7-90b0-171338dcb0cb", allowUseOfArchive=archive_search)

    # get the URIs for the company and the category
    # Scrapes only english articles, because the models are trained with english datasets.
    microsoft_uri = er.getConceptUri(company)
    business_uri = er.getCategoryUri("news business")
    q = QueryArticlesIter(
        conceptUri=QueryItems.OR(microsoft_uri),
        categoryUri=business_uri,
        lang="eng",
        minSentiment=min_sentiment,
        maxSentiment=max_sentiment)
    articles = []
    for article in q.execQuery(er, sortBy="date", maxItems=max_items):
        articles.append([article['title'] + article['body'], article['sentiment']])

    return articles
