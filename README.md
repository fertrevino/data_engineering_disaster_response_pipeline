# Disaster Response Pipeline Project

## Introduction
This repository contains the code for training and using a model to classify social media messages that might be relevant for a disaster response scenario.

The project is divided into three parts: data, training and application.

* The data part constructs a SQLite data base by processing the training data stored in CSV files.
* The training part builds an ETL data pipeline, trains the model and stores it into a .pkl file.
* The application part loads the .pkl model, displays information about the training data set and allows writing a message to classify it. This application is built as a Flask web app.

The dataset used for this project has been obtained from the [Udacity](https://eu.udacity.com/) Data Science Nanodegree workspace.

## Installation
You can download this repository with the following command:
```Shell
git@github.com:fertrevino/data_engineering_disaster_response_pipeline.git
```
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier  saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Motivation
This project is part of the Udacity Nanodegree program.

## File(s) description
This repository contains three main folders: data, models and app, which correspond to the sections mentioned in the introduction above.

## Flask web app overview
The Flask app displays informatino about the training data set. There are three graphs shown.

### Distribution of message genres
This plot shows the genres of the messages used during training. The genre can be seen in the CSV files in the column 'genre'.
![Alt text](resources/distribution_message_genres.png?raw=true "Distribution of message genres")

### Distribution of related and unrelated messages
This plot shows the amount of messages that are related and unrealted ro a disaster response scenario. This information can be seen in the CSV files in the column 'related_messages'.
![Alt text](resources/related_unrelated_messages.png?raw=true "Distribution of related and unrelated messages")

### Distribution of message categories
This plot shows the distribution of all the of message categories. As it can be seen, the two most common categories are 'aid_related' and 'weather_related'. This information can be seen in the database in the encoded columns generated from the CSV files.
![Alt text](resources/categories_messages.png?raw=true "Distribution of message categories")
