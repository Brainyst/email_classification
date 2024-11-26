# Spam Email Classification

This project implements a **spam email classification model** using **Naive Bayes**. The model is trained on a dataset of emails to classify whether a message is spam or ham (not spam). The project uses basic text preprocessing techniques to clean and prepare the data, and then applies a machine learning model to make predictions.

## Project Overview

The goal of this project is to classify emails into two categories:
- Spam: Unwanted or unsolicited emails.
- Ham: Legitimate, non-spam emails.

The code applies the following steps:

1. Data Preprocessing: Clean and transform text data (removes special characters, converts to lowercase, removes stopwords, and applies stemming/lemmatization).
2. Feature Extraction: Convert text messages into a numeric format using CountVectorizer.
3. Model Training: Train a Naive Bayes model on the cleaned and transformed data.
4. Prediction: Use the trained model to predict whether new messages are spam or ham.

## Requirements

- Python 3.x
- pandas for data handling
- nltk for text preprocessing
- scikit-learn for machine learning and model evaluation
- matplotlib and seaborn for visualizations 

You can install the required libraries using pip:

> pip install pandas nltk scikit-learn matplotlib seaborn

## Files

1. email.csv: The dataset of emails that includes two columns:

    - Message: The email text.
    - Category: The label indicating whether the email is "Spam" or "Ham".

2. Spam_Email_Classification.ipynb: The Python notebook containing the code to preprocess the text, train the model, and test it on new data.

## Steps to Run  

1. **Download the Dataset** <br>
First, download the dataset (email.csv) from your source and place it in the same directory as the code file.

2. **Preprocess the Data**<br>
The dataset will be preprocessed by:
- Removing any non-alphabetic characters from the emails.
- Converting all text to lowercase for uniformity.
- Removing stopwords (common words like "the," "is," etc., that do not add much meaning).
- Stemming the words (reducing them to their base form).

3. **Convert Text to Features**<br>
We use **CountVectorizer** to convert the cleaned text data into numerical features that can be used for model training. This creates a matrix of word counts for each message.

4. **Train the Model**<br>
The **Multinomial Naive Bayes** classifier is used to train the model on the features extracted from the emails.

5. **Evaluate the Model**<br>
The model is evaluated on the test set, and the accuracy of the predictions is printed.

6. **Test with a New Message**<br>
You can also test the model with a new email message by adding it to the code, and it will predict whether it's spam or ham.

## Conclusion
This project demonstrates how to classify emails as spam or ham using machine learning. It shows how text data can be cleaned, transformed into features, and used to train a model that can classify new emails.
