# Real / Fake Job Posting Prediction

[![N|Solid](https://www.kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

## About the Project 
The Kaggle provides a base for this project, due to the fact that it allows users to find and publish datasets, explore and build models in a web-based data-science environment. 


Τhe file job_postings.csv contains 18K job descriptions. The “description” column includes the text from job ads, while the “fraudulent” column tells us if one ad is real or a scam. The goal of this project is the guess of information in the second column using a neural network.


*This project prepared for my [M.Sc.](https://ddcdm.ceid.upatras.gr/en/641-2/) course "Big Data Management and Mining Methods".*

## Build with 
- Apache Spark 
- PySpark
- Spark NLP 

## Tasks 
The project includes three parts: 

###### *PART A*
- Preprocessing of the [job postings](job_postings.csv) 
- Removing the stop words from dataset
- Run a word embeddings technique

###### *PART B*
- Run the pretrained BERT transformer

###### *PART C*
- Train a neural network (Multilayer perceptron, ClassifierDL)
- Evaluation the performance (Accuracy, F1 score, Precision & Recall)

## License
Copyright © Gerasimos Katevas 2022