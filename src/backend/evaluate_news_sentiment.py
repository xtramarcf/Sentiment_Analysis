import backend.news_api as news_api
import keras
import paths as paths
from backend.processing import preprocess_articles


# Aggregates and returns the api sentiment.
def __get_aggregated_sentiment(articles):
    aggregated_sentiment = 0.0
    for article in articles:
        if article[1] is not None:
            aggregated_sentiment = aggregated_sentiment + article[1]
    return aggregated_sentiment / len(articles)


# Gets the aggregated sentiment of all scraped articles with the news model.
def __get_articles_sentiment_news(scraped_articles):
    model = keras.models.load_model(paths.model_path_news)
    preprocessed_articles = preprocess_articles(scraped_articles, paths.tokenizer_path_news)
    articles_sentiment = 0.000
    for preprocessed_article in preprocessed_articles:
        article_sentiment = 0.000
        prediction = model.predict(preprocessed_article[0])
        for value in prediction:
            article_sentiment = article_sentiment + value

        article_sentiment = article_sentiment / len(preprocessed_article[0])
        articles_sentiment = articles_sentiment + article_sentiment

    articles_sentiment = articles_sentiment / len(preprocessed_articles)

    return articles_sentiment


# Expects a list of articles as an argument.
# Evaluates the aggregated sentiment of all scraped articles with the twitter model.
# The prediction is represented in an array with 3 values. The first value
# is the probability for negative, the second for neutral and the third for positive sentiment.
def __get_articles_sentiment_twitter(scraped_articles):
    model = keras.models.load_model(paths.model_path_twitter)
    preprocessed_articles = preprocess_articles(scraped_articles, paths.tokenizer_path_twitter)
    articles_sentiment = [0.0, 0.0, 0.0]
    for preprocessed_article in preprocessed_articles:
        article_sentiment = [0.0, 0.0, 0.0]
        prediction = model.predict(preprocessed_article[0])
        for value in prediction:
            article_sentiment[0] = article_sentiment[0] + value[0]
            article_sentiment[1] = article_sentiment[1] + value[1]
            article_sentiment[2] = article_sentiment[2] + value[2]

        article_sentiment[0] = article_sentiment[0] / len(prediction)
        article_sentiment[1] = article_sentiment[1] / len(prediction)
        article_sentiment[2] = article_sentiment[2] / len(prediction)

        articles_sentiment[0] = articles_sentiment[0] + article_sentiment[0]
        articles_sentiment[1] = articles_sentiment[1] + article_sentiment[1]
        articles_sentiment[2] = articles_sentiment[2] + article_sentiment[2]
    articles_sentiment[0] = articles_sentiment[0] / len(preprocessed_articles)
    articles_sentiment[1] = articles_sentiment[1] / len(preprocessed_articles)
    articles_sentiment[2] = articles_sentiment[2] / len(preprocessed_articles)

    return articles_sentiment


# Evaluates the numerous api sentiment to a positive, neutral or negative state.
def __evaluate_api_sentiment(api_sentiment):
    if -0.05 < api_sentiment < 0.05:
        return "neutral"
    elif api_sentiment <= -0.05:
        return "negative"
    else:
        return "positive"


# Evaluates the numerous news sentiment to a positive, neutral or negative state.
def __evaluate_news_sentiment(news_sentiment):
    if -0.005 < news_sentiment < 0.005:
        return "neutral"
    elif news_sentiment <= -0.005:
        return "negative"
    else:
        return "positive"


# Evaluates the numerous twitter sentiment to a positive, neutral or negative state.
def __evaluate_twitter_sentiment(twitter_sentiment):
    if twitter_sentiment[1] > twitter_sentiment[0] and twitter_sentiment[1] > twitter_sentiment[2]:
        return "neutral"
    elif twitter_sentiment[0] > twitter_sentiment[1] and twitter_sentiment[0] > twitter_sentiment[2]:
        return "negative"
    else:
        return "positive"


# Gets and evaluates all the sentiment states and returns them with the length of the found articles.
# If there are no articles found, 1 will be returned.
def get_all_sentiment(company, archive_search, min_sentiment=-1.0, max_sentiment=1.0, max_items=100):
    articles = news_api.get_articles(company, archive_search, min_sentiment, max_sentiment, max_items)
    if len(articles) != 0:
        api_sentiment = __evaluate_api_sentiment(__get_aggregated_sentiment(articles))
        # news_sentiment = __evaluate_news_sentiment(__get_articles_sentiment_news(articles))
        twitter_sentiment = __evaluate_twitter_sentiment(__get_articles_sentiment_twitter(articles))
        # return len(articles), api_sentiment, news_sentiment, twitter_sentiment
        return len(articles), api_sentiment, None, twitter_sentiment
    else:
        return 1
