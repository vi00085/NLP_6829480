# LSTM with TF-IDF Vectorization - Flask Application

This repository contains a Flask web application that deploys a Long Short-Term Memory (LSTM) model with Term Frequency-Inverse Document Frequency (TF-IDF) vectorization. The project involves comparing multiple vectorization techniques and machine learning models, implementing hyperparameter tuning, and finally deploying the optimized model on a cloud server using Flask.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Experiments](#experiments)
  - [Experiment 1: Vectorization Comparison](#experiment-1-vectorization-comparison)
  - [Experiment 2: Model Architecture Comparison](#experiment-2-model-architecture-comparison)
  - [Experiment 3: Optimizer Comparison](#experiment-3-optimizer-comparison)
  - [Experiment 4: Hyperparameter Tuning](#experiment-4-hyperparameter-tuning)
- [Files Included](#files-included)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Results](#results)

## Project Overview

The project is structured to answer a series of questions in natural language processing (NLP) by experimenting with vectorization techniques, different recurrent neural network architectures, optimizers, and hyperparameter tuning. The final model, which combines LSTM and TF-IDF with hyperparameter tuning via Random Search, is deployed as an API using Flask.

## Dataset

The dataset used in this project is text-based and focuses on a biomedical or genetic research domain. Common words include terms like "gene," "associated," and "protein." The dataset has been preprocessed and includes a split into training and test sets.

- **Word Cloud Visualization**: A word cloud was generated to understand the most frequent words and gain insights into the dataset.
- **Text Length Histogram**: Text lengths were visualized using a histogram to understand the distribution of sentence lengths.

## Experiments

### Experiment 1: Vectorization Comparison

**Goal**: Compare the performance of different text vectorization techniques (TF-IDF, Word2Vec, and GloVe) using an Artificial Neural Network (ANN).

**Results**: 
- **TF-IDF** performed the best in this comparison.

### Experiment 2: Model Architecture Comparison

**Goal**: Compare the performance of **RNN**, **LSTM**, and **GRU** models using TF-IDF for vectorization.

**Results**: 
- **LSTM** outperformed RNN and GRU.

### Experiment 3: Optimizer Comparison

**Goal**: Compare two optimizers, **Adam** and **SGD**, for training the LSTM model with TF-IDF vectorization.

**Results**: 
- **Adam Optimizer** showed better performance in terms of model accuracy and loss compared to **SGD**.

### Experiment 4: Hyperparameter Tuning

**Goal**: Optimize the LSTM model using TF-IDF through two hyperparameter tuning methods:
- **Bayesian Optimization**
- **Random Search**

**Results**: 
- **Random Search** performed better, and the resulting model was deployed in the Flask application.

## Files Included

- **`NLP.ipynb`**: Initial experiments and model development.
- **`Sandeep_6829480_NLP.ipynb`**: Refined experiments, including hyperparameter tuning and optimizers.
- **`Sandeep_6829480_NLP.pdf`**: PDF version of the notebook for easy reference.
- **`app.py`**: Flask web application for serving the LSTM model via API.
- **`locustfile.py`**: Performance testing script for load testing with Locust.
- **`logparse.py`**: Script for log parsing and analysis.
- **`requirement.txt`**: List of required Python packages.
- **`test_api.py`**: Script to test API functionality.

## Installation and Setup

1. Clone the repository:
   git clone https://github.com/vio0085/NLP_6829480.git

2. Navigate to the project directory:
     cd NLP_6829480

3. Install the required dependencies:
     pip install -r requirements.txt

4. Run the Flask application:
     python app.py

## Running the Application

- Start the Flask server, which will expose endpoints for interacting with the LSTM model.
- The API allows users to input text data for classification using the LSTM model with optimized hyperparameters.

## API Endpoints

- POST /predict: Predicts the class of the input text using the trained LSTM model.
- Input: JSON object containing the text to be classified.
- Output: Predicted class (binary classification).

## Results

- Best Model: LSTM with TF-IDF, optimized using Random Search and trained with Adam Optimizer.
- This model was successfully deployed via Flask, allowing for easy interaction with the trained model using RESTful API endpoints.
