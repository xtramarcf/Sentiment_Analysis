import numpy as np
from keras import layers
from keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow import keras
import h5py
import csv
import paths as paths
import matplotlib.pyplot as plt
import os
import json


# Plots the training history for the twitter training and saves the plot in the model directory.
def plot_and_save_training_history_news(history, model_directory):
    err = history.history['mean_squared_error']
    val_err = history.history['val_mean_squared_error']
    epochs = range(1, len(err) + 1)

    # Creates a diagram for the mean absolute error
    plt.subplot(1, 2, 2)
    plt.plot(epochs, err, label='Training Error')
    plt.plot(epochs, val_err, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('MeanSquaredError')
    plt.legend()
    plt.xticks(epochs[::2])

    # Saves the diagram.
    plt.savefig(os.path.join(model_directory, 'training_history.jpg'))


# Plots the training history for the twitter training and saves the plot in the model directory.
def plot_and_save_training_history_twitter(history, model_directory):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    epochs = range(1, len(loss) + 1)

    # Creates a diagram for the loss.
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(epochs[::2])

    # Creates a diagram for the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(epochs[::2])

    # Saves the diagram.
    plt.savefig(os.path.join(model_directory, 'training_history.jpg'))


# Loads the processed data out of the h5 and csv files and returns the required data for the model.
def load_processed_train_data(data_file_h5, parameter_file_csv):
    with h5py.File(data_file_h5, 'r') as file:
        train_padded = file['train_padded'][:]
        val_padded = file['val_padded'][:]
        train_labels = file['train_labels'][:]
        val_labels = file['val_labels'][:]

    with open(parameter_file_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            data.append(row)

        num_unique_words = data[2][0]
        max_length = data[2][1]

    return num_unique_words, max_length, train_padded, val_padded, train_labels, val_labels


# Gets the training configuration and saves it in json files in the same directory,
# where the model is saved.
# Converts float32 values to float64 (which is serializable with JSON) for the optimizer and the loss.
def save_training_configuration(model, optim, loss, model_path):
    model_config = model.get_config()
    with open(model_path + "/model_config.json", "w") as model_file:
        json.dump(model_config, model_file, indent=4)

    optim_config = optim.get_config()

    optim_config = {key: float(value) if isinstance(value, np.float32) else value for key, value in
                    optim_config.items()}
    with open(model_path + "/optim_config.json", "w") as optim_file:
        json.dump(optim_config, optim_file, indent=4)

    loss_config = loss.get_config()
    loss_config = {key: float(value) if isinstance(value, np.float32) else value for key, value in loss_config.items()}
    with open(model_path + "/loss_config.json", "w") as loss_file:
        json.dump(loss_config, loss_file, indent=4)


# Formatting the labels for the twitter data to a form which the model can process.
# [1,0,0] = negative    [0,1,0] = neutral    [0,0,1] = positive
def formatting_labels(labels):
    formatted_labels = []
    for label in labels:
        if label == -1:
            formatted_labels.append([1, 0, 0])
        elif label == 0:
            formatted_labels.append([0, 1, 0])
        else:
            formatted_labels.append([0, 0, 1])
    return np.array(formatted_labels)


# Schedules the learning rate after 4 epochs for a more precise optimization.
def learning_rate_scheduler(epoch, lr):
    if epoch < 4:
        return lr
    else:
        return lr * 0.9


# Training the NLP model.
def train_model_news():
    model_path = paths.model_path_news

    # Loads the processed data out of the data and the parameter file.
    (num_unique_words,
     max_length,
     train_padded,
     val_padded,
     train_labels,
     val_labels) = (load_processed_train_data(
        paths.h5_processed_train_data_news,
        paths.csv_processed_train_data_parameter_news))

    # Casts the two parameters from string to int.
    num_unique_words = int(num_unique_words)
    max_length = int(max_length)

    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)

    # Stops the training if the val_loss does not continue improving over the time of 5 epochs.
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=2,
                                   restore_best_weights=True)

    # Creating the model and adding different layers with appropriate hyperparameters.
    model = keras.models.Sequential()
    model.add(layers.Embedding(num_unique_words, 100, input_length=max_length))
    model.add(layers.LSTM(128, dropout=0.8))
    model.add(layers.Dense(32))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='tanh'))

    optim = keras.optimizers.Adam(learning_rate=0.0001)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError()]


    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    # Callbacks are used for additional functions like early stopping or reducing the
    # learning rate during the training process.
    callbacks = [lr_scheduler, early_stopping]

    # Using validation dataset for optimizing. Training the model with the data.
    # Saves the training history for a plot.
    training_history = model.fit(train_padded, train_labels, epochs=30,
                                 validation_data=(val_padded, val_labels),
                                 callbacks=callbacks, verbose=2)
    # This line saves the model. Then it is possible to load the trained model for making predictions.
    model.save(model_path)

    # Uses the history for creating a plot for the training and saves it into the model folder.
    plot_and_save_training_history_news(training_history, model_path)

    # Saves the training configuration in json files for analytical reasons and comparing the different trainings.
    save_training_configuration(model, optim, loss, model_path)

    print("")
    print("Training the model with news data finished!")
    print("")


# Training the twitter NLP model.
def train_model_twitter():
    model_path = paths.model_path_twitter

    # Loads the processed data out of the data and the parameter file.
    (num_unique_words,
     max_length,
     train_padded,
     val_padded,
     train_labels,
     val_labels) = (load_processed_train_data(
        paths.h5_processed_train_data_twitter,
        paths.csv_processed_train_data_parameter_twitter))

    # Casts the two parameters from string to int.
    num_unique_words = int(num_unique_words)
    max_length = int(max_length)

    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)

    # Stops the training if the val_loss does not continue improving over the time of 5 epochs.
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=2,
                                   restore_best_weights=True)

    # Creating the model and adding different layers with appropriate hyperparameters.
    model = keras.models.Sequential()
    model.add(layers.Embedding(num_unique_words, 100, input_length=max_length))
    model.add(layers.LSTM(128, dropout=0.8))
    model.add(layers.Dense(32))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(3, activation='softmax'))


    train_labels = formatting_labels(train_labels)
    val_labels = formatting_labels(val_labels)

    optim = keras.optimizers.Adam(learning_rate=0.0001)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = ['categorical_accuracy']

    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    # Callbacks are used for additional functions like early stopping or reducing the
    # learning rate during the training process.
    callbacks = [lr_scheduler, early_stopping]

    # Using validation dataset for optimizing. Training the model with the data.
    # Saves the training history for a plot.
    training_history = model.fit(train_padded, train_labels, epochs=30,
                                 validation_data=(val_padded, val_labels),
                                 callbacks=callbacks, verbose=2)

    # This line saves the model. Then it is possible to load the trained model for making predictions.
    model.save(model_path)

    model.get_config()
    optim.get_config()
    loss.get_config()

    # Uses the history for creating a plot for the training and saves it into the model folder.
    plot_and_save_training_history_twitter(training_history, model_path)

    # Saves the training configuration in json files for analytical reasons and comparing the different trainings.
    save_training_configuration(model, optim, loss, model_path)

    print("")
    print("Training the model with twitter data finished!")
    print("")


# train_model_twitter()
train_model_news()
