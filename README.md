# Disaster Response

# Project description
This project is part of my Udacity Data Science nanodegree coursework. 

The aim is to build a model that classifying messages sent during disasters into 36 pre-defined categories. The classification outcome helps route these messages to the appropriate disaster relief agencies. The project has three steps:
1. Performing an Extract, Transform, Load (ETL) pipeline: extracting data from source, clean the data and save them in a SQLite DB
2. Machine Learning Pipeline: a multi-label classification problem as any message can be in more than one category.
3. The outputs of the classification can be viewed via a web application. The app takes a user's message as an input.

The data is provided by Figure Eight.

# Dependencies 
- Python 3.7+
- Libraries: NumPy, SciPy, Pandas, sklearn, NLTK, SQLalchemy, Pickle, Flask, Plotly, Json, SqLite3

# Execution instructions
1. Run the following commands in the project's directory to set up the database, train model and save the model.

2. Run ETL pipeline to clean data and store the processed data in the database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db

3. To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl

4. Run the following command in the app's directory to run your web app. python run.py

5. Go to http://0.0.0.0:3001/

# File description
app/templates/*: templates/html files for web app

data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

models/train_classifier.py: ML pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

run.py: This file can be used to launch the Flask web app used to classify disaster messages

The two jupyter notebooks are part of the course and provide an overview of the model build

# Licensing, authors
https://www.udacity.com/

Figure8 now Appen - https://appen.com/


