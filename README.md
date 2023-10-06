# Sentiment_Analysis
This repository is solely for the development project. It does not serve as the application intended for production use, but this README will describe, how the application can be installed for the productive environment. 
First, we'll explain how to install the application. Then we'll provide instructions on setting up the development project.

## 1. Running the Application in Production / Installation
The marketing department exclusively uses Windows operating systems, so the application will be provided for installation as a .exe file.
Please do the following steps for installing the Sentiment_Analysis application:

1. Download the News_Sentiment_Analysis.zip with the following link:
   https://www.dropbox.com/scl/fi/wf0zza9d20z7mlxikzv6r/News_Sentiment_Analysis.zip?rlkey=wlr6tdookh2adxwnye9jx3e5y&dl=0

2. Unpack the .zip.

3. Start the application: ..\News_Sentiment_Analysis\app\app.bat

Starting the application by the app.bat will automatically minimize the cli. Of course it is also possible to execute the app.exe.
The application with all its dependencies and the python runtime was packaged together by pyinstaller. That means the application does not require a python installation.


## 2. Setting up the Development Project
The repository contains all scripts of the development project. 
But the project also requires some data files, which represents the raw data, the formatted data, the trained models and so on.
The data files needs to be added manually.

Please do the following steps:

1. Create a local folder for the Sentiment_Analysis and pull the project to that folder.

2. Download data.zip from the following link: 
   https://www.dropbox.com/scl/fi/x9605m6a7lee7l88zfu8p/data.zip?rlkey=g8tdzd8w4yxoan8vyrrw8tkl4&dl=0

3. You have to unpack the downloaded .zip and copy the data folder from the .zip file to the project directory.
   The src-folder of the project and the data-folder need to be in the same directory.
   
Now you can open the project in an IDE of your choice. With adding the data files, all the required paths are now available.
