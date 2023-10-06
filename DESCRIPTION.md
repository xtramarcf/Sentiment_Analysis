# Sentiment Analysis - Documentation
The development project has two main parts: the data directory and the src directory. The data directory holds all the stored data, while the src directory contains the Python code.

## 1. Data
The "data" directory has two subdirectories for data storage. One is for news data and the model trained with it, while the other is for Twitter data and the associated model. Both folders contain raw and processed data for both training and testing. Additionally, the tokenizer is saved within these directories.
In the "model" directory, you'll find the saved model along with all its associated data for usage. The model itself, as well as the configuration for loss and optimizer, are saved in .json files. Furthermore, the training history of the model was plotted and saved in this directory.

## 2. Src
The "src" directory consists of the backend and frontend package. You'll also find app.py, which serves as the starting point for our application, and paths.py, where the data directory paths for the application are stored.
### Frontend
The Frontend is represented by a single script running the application and invoking the methods for processing the input data to outputdata.
### Backend
Evaluate News Sentiment - Aggregates and interprets the sentiment of news articles.\n
Formatting - Formatting and cleaning the rawdata as an upstreamed process of the processing itself.
Manual Testing -Supports the developer for manual tests.
Model - Represents the Model configuration and methods for training the model.
News API - Configures an API for requesting news and their sentiment.
Processing - Processes the formatted data in a format the model understands.

This description should be viewed as an overview of the development project. More detailed information about the methods and processes can be found within the commented code.
