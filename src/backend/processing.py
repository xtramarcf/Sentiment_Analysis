import h5py
import csv
import pickle
import numpy as np
import paths as paths
from backend.formatting import (formatting_train_data_news,
                                formatting_test_data_news,
                                formatting_single_dataset,
                                formatting_test_data_twitter,
                                formatting_train_data_twitter,
                                text_to_sentences)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter


# Counting all different words
def __counter_words(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


# Splits the data into two sets. One set for the training and another set for the validation.
# 80% of the overall datasets will be for training and 20% for the validation.
def __split_data_into_train_and_val(data):
    val_size = int(data[0].size * 0.2)
    data_train = [data[0][val_size:], data[1][val_size:]]
    data_val = [data[0][:val_size], data[1][:val_size]]
    return data_train, data_val


# Splits the train and validation dataframe into text and sentiment (label).
def __split_text_and_labels(data):
    data_train, data_val = __split_data_into_train_and_val(data)
    train_sentences = np.array(data_train[0])
    train_labels = np.array(data_train[1])
    val_sentences = np.array(data_val[0])
    val_labels = np.array(data_val[1])
    return train_sentences, train_labels, val_sentences, val_labels


# Vectorize a text corpus by turning the text into a sequence of integers.
# Each word has a unique index.
# Lowest index 1 is the word with most usage.
# Sentences are the sentences with real words while the sequence has indices for each word.
# Saving the tokenizer for later usage. The tokenizer is also needed for processing productive and testdata.
def __tokenizing_data_and_saving_tokenizer(data, train_sentences, val_sentences, tokenizer_file, save_tokenizer):
    counter = __counter_words(data[0])
    num_unique_words = len(counter)
    tokenizer = Tokenizer(num_words=num_unique_words)
    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)

    if save_tokenizer:
        with open(tokenizer_file, 'wb') as tf:
            pickle.dump(tokenizer, tf)
    return train_sequences, val_sequences, num_unique_words


# Padding the sequences to the same length.
# The length is the max number of words in a sequence.
def __pad_data_to_defined_length(max_length, sequences):
    sequence_padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    return sequence_padded


# Preprocess the data by splitting into text and label, tokenization and padding.
def __preprocessing_data(data, tokenizer_file):
    train_sentences, train_labels, val_sentences, val_labels = __split_text_and_labels(data)
    train_sequences, val_sequences, num_unique_words = __tokenizing_data_and_saving_tokenizer(
        data,
        train_sentences,
        val_sentences,
        tokenizer_file,
        True)
    train_padded = __pad_data_to_defined_length(max_padding_length, train_sequences)
    val_padded = __pad_data_to_defined_length(max_padding_length, val_sequences)
    return num_unique_words, max_padding_length, train_padded, val_padded, train_labels, val_labels


# Saves the training data in h5 format.
# This allows the model a faster usage, because the preprocessed data can be loaded out of the h5 file.
def __saves_training_data_in_h5(columns, preprocessed_data, train_data_file):
    with h5py.File(train_data_file, 'w') as hdf5_file:
        for i, column_data in enumerate(preprocessed_data):
            dataset_name = columns[i]
            hdf5_file.create_dataset(dataset_name, data=np.array(column_data))


# Saves the parameter in a separate csv file.
def __saves_parameter_in_csv(preprocessed_data, train_data_parameter_file):
    with open(train_data_parameter_file, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['num_unique_words', 'max_length'])
        csv_writer.writerow([preprocessed_data[0], preprocessed_data[1]])


# Processes the data and saves all the data needed for fitting the model into two files.
def process_and_save_train_data(data, processed_data_file, processed_data_parameter_file, tokenizer_file):
    preprocessed_data = __preprocessing_data(data, tokenizer_file)
    columns = ['num_unique_words', 'max_length', 'train_padded', 'val_padded', 'train_labels', 'val_labels']

    __saves_training_data_in_h5(columns, preprocessed_data, processed_data_file)
    __saves_parameter_in_csv(preprocessed_data, processed_data_parameter_file)


# Process and save test data in another h5 file.
# For Testing the padded and the real sentence is needed.
# It is important to compare the prediction with the real sentence for evaluation.
def process_and_save_test_data(test_data, test_data_file, tokenizer_file):
    with open(tokenizer_file, 'rb') as tf:
        tokenizer = pickle.load(tf)
    if len(test_data) == 1:
        test_sentences = np.array(test_data[0])
    else:
        test_data_ = [test_data]
        test_sentences = np.array(test_data_[0])

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = __pad_data_to_defined_length(max_padding_length, test_sequences)

    with h5py.File(test_data_file, 'w') as hf:
        hf.create_dataset('test_padded', data=test_padded)
        hf.create_dataset('test_sentences', data=test_sentences)


# Expects a list of articles and the path of a tokenizer_file as arguments.
# Processes the articles to a padded list of sentences.
# Returns an object with all the padded data and the original sentences.
def preprocess_articles(articles, tokenizer_file):
    processed_articles = []

    with open(tokenizer_file, 'rb') as tf:
        tokenizer = pickle.load(tf)

    for article in articles:
        sentences = np.array(text_to_sentences(article[0]))
        formatted_sentences = []
        for sentence in sentences:
            formatted_sentences.append(formatting_single_dataset(sentence))
        article_sentences = np.array(formatted_sentences)
        article_sequences = tokenizer.texts_to_sequences(article_sentences)
        article_padded = __pad_data_to_defined_length(max_padding_length, article_sequences)
        processed_articles.append([article_padded, sentences])

    return processed_articles


# Method tests a given sequence by decoding the sequence into the original sentence.
def __decoding_tokenization(data, train_sentences, sequence):
    counter = __counter_words(data[0])
    num_unique_words = len(counter)
    tokenizer = Tokenizer(num_words=num_unique_words)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])
    return " ".join([reverse_word_index.get(idx, "=") for idx in sequence])


# Encode a sentence into a sequence and decodes it. Compares the initial value with the result.
# If they are equal, the test is passed.
def encode_decode_tokenization_test(data):
    train_sentences, train_labels, val_sentences, val_labels = __split_text_and_labels(data)
    train_sequences, val_sequences, num_unique_words = __tokenizing_data_and_saving_tokenizer(
        data,
        train_sentences,
        val_sentences,
        None,
        False)
    decoded_text = __decoding_tokenization(data, train_sentences, train_sequences[5])
    print(train_sentences[5])
    print(train_sequences[5])
    print(decoded_text)
    if train_sentences[5] == decoded_text:
        return True


# Sets the maximum length for the sentence padding.
max_padding_length = 30


# Preprocesses the train and test of news and twitter data.
def process_news_data():
    ##############################################################################################
    # Preprocessing News data
    # Formatting the train and test data.
    data_formatted_news = formatting_train_data_news(False, paths.raw_train_data_news)
    data_formatted_test_news = formatting_test_data_news(paths.raw_test_data_news)
    # Tests if the tokenization works fine.
    if encode_decode_tokenization_test(data_formatted_news):
        # Processes and saves the train data in files.
        process_and_save_train_data(data_formatted_news,
                                    paths.h5_processed_train_data_news,
                                    paths.csv_processed_train_data_parameter_news,
                                    paths.tokenizer_path_news)
        # Processes and saves the test data in files.
        process_and_save_test_data(data_formatted_test_news,
                                   paths.h5_processed_test_data_news,
                                   paths.tokenizer_path_news)
    else:
        print("Error: Tokenization failed.")


def process_twitter_data():
    ###############################################################################################
    # Preprocessing Twitter data
    # Formatting the train and test data.
    data_formatted_twitter = formatting_train_data_twitter(paths.raw_train_data_twitter)
    data_formatted_test_twitter = formatting_test_data_twitter(paths.raw_test_data_twitter)
    # Tests if the tokenization works fine.
    if encode_decode_tokenization_test(data_formatted_twitter):
        # Processes and saves the train data in files.
        process_and_save_train_data(data_formatted_twitter,
                                    paths.h5_processed_train_data_twitter,
                                    paths.csv_processed_train_data_parameter_twitter,
                                    paths.tokenizer_path_twitter)
        # Processes and saves the test data in files.
        process_and_save_test_data(data_formatted_test_twitter,
                                   paths.h5_processed_test_data_twitter,
                                   paths.tokenizer_path_twitter)
    else:
        print("Error: Tokenization failed.")
