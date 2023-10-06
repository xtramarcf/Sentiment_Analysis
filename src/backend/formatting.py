import re
import string
import sklearn.utils
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize


# Removing all urls.
def __remove_url(text):
    text = str(text)
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


# Removing all punctuation.
def __remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


# Removing all line breaks.
def __remove_line_breaks(text):
    return text.replace('\n', ' ')


# Removing all numbers.
def __remove_numbers(text):
    text_without_numbers = re.sub(r'\d+', '', text)
    return text_without_numbers


# Removes stopwords like 'and, in, on, into' and so on,
# because they do not represent information about the sentiment.
def __remove_stopwords(text):
    stop = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


# Lemmatizate text for converting words to their basic form.
def __lemmatizate_text(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text


# Applies all formatting methods on the news dataframe.
def __preformatting_news_dataframe(df):
    df["Headline"] = df.Headline.map(__remove_url)
    df["Headline"] = df.Headline.map(__remove_punctuation)
    df["Headline"] = df.Headline.map(__remove_line_breaks)
    df["Headline"] = df.Headline.map(__remove_numbers)
    df["Headline"] = df.Headline.map(__remove_stopwords)
    df["Headline"] = df.Headline.map(__lemmatizate_text)
    df["Title"] = df.Title.map(__remove_url)
    df["Title"] = df.Title.map(__remove_punctuation)
    df["Title"] = df.Title.map(__remove_line_breaks)
    df["Title"] = df.Title.map(__remove_numbers)
    df["Title"] = df.Title.map(__remove_stopwords)
    df["Title"] = df.Title.map(__lemmatizate_text)


# Applies all formatting methods on the twitter dataframe.
def __preformatting_twitter_dataframe(df):
    df["clean_text"] = df.clean_text.map(__remove_url)
    df["clean_text"] = df.clean_text.map(__remove_punctuation)
    df["clean_text"] = df.clean_text.map(__remove_line_breaks)
    df["clean_text"] = df.clean_text.map(__remove_numbers)
    df["clean_text"] = df.clean_text.map(__remove_stopwords)
    df["clean_text"] = df.clean_text.map(__lemmatizate_text)


# Processing the dataframe for equalizing the number of datasets with the categories
# negative, neutral and positive.
def __equalize_sentiment_dataset_ratio(sentiment_label, df):
    negative = df[df[sentiment_label] < 0]
    neutral = df[df[sentiment_label] == 0]
    positive = df[df[sentiment_label] > 0]

    min_samples = min(len(negative), len(neutral), len(positive))

    negative_balanced = sklearn.utils.resample(negative, replace=False, n_samples=min_samples, random_state=42)
    neutral_balanced = sklearn.utils.resample(neutral, replace=False, n_samples=min_samples, random_state=42)
    positive_balanced = sklearn.utils.resample(positive, replace=False, n_samples=min_samples, random_state=42)

    balanced_df = pd.concat([negative_balanced, neutral_balanced, positive_balanced])
    return balanced_df


# Extracts all the relevant information out of the news train data for the sentiment analysis.
# If the flag balanced is set, the method returns a dataframe, where the ratio between
# the positive, neutral and negative sentiments are equal. If the flag is not set,
# the method returns all formatted datasets.
def formatting_train_data_news(balanced, train_data_file):
    raw_train_df = pd.read_csv(train_data_file)
    __preformatting_news_dataframe(raw_train_df)

    if balanced:
        df_balanced_headline = __equalize_sentiment_dataset_ratio('SentimentHeadline', raw_train_df)
        df_balanced_title = __equalize_sentiment_dataset_ratio('SentimentTitle', raw_train_df)
        headline_data_balanced = [df_balanced_headline['Headline'], df_balanced_headline['SentimentHeadline']]
        title_data_balanced = [df_balanced_title['Title'], df_balanced_title['SentimentTitle']]
        return [pd.concat([headline_data_balanced[0], title_data_balanced[0]], axis=0),
                pd.concat([headline_data_balanced[1], title_data_balanced[1]], axis=0)]
    else:
        headline_data = [raw_train_df['Headline'], raw_train_df['SentimentHeadline']]
        title_data = [raw_train_df['Title'], raw_train_df['SentimentTitle']]
        return [pd.concat([headline_data[0], title_data[0]], axis=0),
                pd.concat([headline_data[1], title_data[1]], axis=0)]


# Extracts all the relevant information out of the news test data for testing the trained model.
def formatting_test_data_news(test_data_file):
    raw_test_df = pd.read_csv(test_data_file)
    __preformatting_news_dataframe(raw_test_df)
    overall_data = [pd.concat([raw_test_df['Headline'], raw_test_df['Title']], axis=0)]
    return overall_data


# Extracts all the relevant information out of the twitter train data for the sentiment analysis.
def formatting_train_data_twitter(train_data_file):
    raw_train_df = pd.read_csv(train_data_file)
    __preformatting_twitter_dataframe(raw_train_df)
    data = [raw_train_df['clean_text'], raw_train_df['category']]
    return data


# Extracts all the relevant information out of the twitter test data for the sentiment analysis.
def formatting_test_data_twitter(test_data_file):
    raw_test_df = pd.read_csv(test_data_file)
    __preformatting_twitter_dataframe(raw_test_df)
    return raw_test_df['clean_text']


# Formats a single dataset and returns it.
def formatting_single_dataset(data_set):
    data_set = __remove_url(data_set)
    data_set = __remove_punctuation(data_set)
    data_set = __remove_line_breaks(data_set)
    data_set = __remove_numbers(data_set)
    data_set = __remove_stopwords(data_set)
    data_set = __lemmatizate_text(data_set)
    return data_set


# Splits a text in its single sentences.
def text_to_sentences(text):
    sentences = sent_tokenize(text)
    return sentences
