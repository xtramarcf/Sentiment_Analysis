import paths as fn
import keras
import h5py
import random


# Gets the sentiment in the three classifications neutral, positive and negative with the textblob library for testing.
def __get_sentiment_classification(prediction):
    if prediction == [1, 0, 0]:
        return "negative"
    elif prediction == [0, 1, 0]:
        return "neutral"
    else:
        return "positive"


# Loads all the preprocessed testdata out of the csv file and returns the single columns.
def __load_processed_test_data(test_file):
    with h5py.File(test_file, 'r') as file:
        test_padded = file['test_padded'][:]
        test_sentences = file['test_sentences'][:]

    return test_padded, test_sentences


# Gets a random the range for news and twitter datasets.
def __get_random_test_data(number_of_sets, test_padded_news, test_padded_twitter):
    news_val_1, news_val_2 = __generate_random_range(number_of_sets, test_padded_news)
    twitter_val_1, twitter_val_2 = __generate_random_range(number_of_sets, test_padded_twitter)
    return news_val_1, news_val_2, twitter_val_1, twitter_val_2


# Generates and returns a random range of datasets out of data
def __generate_random_range(number_of_sets, data):
    random_number_news = random.randint(0, len(data) - 1)
    if random_number_news + number_of_sets > len(data):
        return random_number_news - number_of_sets, random_number_news
    else:
        return random_number_news, random_number_news + number_of_sets


# Method prints out a variety of combinations for manuel testing.
# A variable number of sentences from the news and the twitter data will be predicted with both models.
def model_test():
    # Loads the data out of the test data files.
    (test_padded_news,
     test_sentences_news) = __load_processed_test_data(fn.h5_processed_test_data_news)
    (test_padded_twitter,
     test_sentences_twitter) = __load_processed_test_data(fn.h5_processed_test_data_twitter)

    # Loading the saved model.
    model_news = keras.models.load_model("../data/news/model")
    model_twitter = keras.models.load_model("../data/twitter/model")

    # Sets the amount of datasets, which will be printed.
    number_of_test_sets = 5

    # Gets a random range of datasets out of the news and twitter data.
    news_val_1, news_val_2, twitter_val_1, twitter_val_2 = __get_random_test_data(number_of_test_sets, test_padded_news,
                                                                                  test_padded_twitter)
    # Prints out the formatted sentences and the prediction of both models.
    print("Dataset with news sentence:")
    print("Prediction sentence: ")
    print(test_sentences_news[news_val_1:news_val_2])
    print("News model: ")
    print(model_news.predict(test_padded_news[news_val_1:news_val_2]))
    print("Twitter model: ")
    print(model_twitter.predict(test_padded_news[news_val_1:news_val_2]))

    print("Dataset with twitter sentence:")
    print("Prediction sentence: ")
    print(test_sentences_twitter[twitter_val_1: twitter_val_2])
    print("News model: ")
    print(model_news.predict(test_padded_twitter[twitter_val_1: twitter_val_2]))
    print("Twitter model: ")
    print(model_twitter.predict(test_padded_twitter[twitter_val_1: twitter_val_2]))


model_test()
