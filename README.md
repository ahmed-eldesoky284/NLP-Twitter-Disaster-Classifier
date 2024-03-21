# Natural Language Processing (NLP) and Machine Learning for Disaster Tweet Classification

## Overview
This project utilizes Natural Language Processing (NLP) techniques and machine learning algorithms to classify tweets into those related to natural disasters and those that are not. The aim is to demonstrate how NLP and machine learning can be applied to real-world problems, particularly in enhancing disaster response mechanisms through social media analysis.

## Dataset
The dataset used for this project is sourced from Kaggle and consists of tweets related to natural disasters. You can find the dataset [here](https://www.kaggle.com/competitions/nlp-getting-started).

## Project Goals
- Preprocess and analyze tweet text using NLP techniques.
- Train various machine learning models to classify tweets as disaster-related or not.
- Evaluate model performance to identify the most accurate one.
- Gain practical experience in data analysis, modeling, and interpretation within the context of NLP and machine learning.

## Steps
1. **Data Acquisition:** Download the dataset from Kaggle and understand its structure and content.
2. **Data Preprocessing:** Clean the tweet text, normalize it, and tokenize for further analysis.
3. **Feature Extraction:** Utilize TF-IDF or word embeddings to convert text data into a numerical format suitable for machine learning.
4. **Model Training and Selection:** Train different machine learning models including Naive Bayes, Logistic Regression, Support Vector Machines, and neural networks.
5. **Model Evaluation:** Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
6. **Interpretation and Application:** Select the best-performing model and discuss its potential impact on disaster response strategies.
7. **Documentation and Presentation:** Document the entire project process, methodologies, model choices, and evaluation outcomes.

## Usage
1 - **Clone the Repository:**
   ```
   git clone https://github.com/ahmed-eldesoky284/NLP-Twitter-Disaster-Classifier.git
   cd NLP-Twitter-Disaster-Classifier
```

2 - **Install Dependencies:**
  Ensure you have the necessary dependencies installed. You can install them using:
  ```
  pip install pandas nltk scikit-learn tensorflow
  ```
3 - Run the Jupyter Notebook:
  Open the provided Jupyter Notebook 
  (NLP Twitter Disaster Classifier.ipynb) using Jupyter Notebook or JupyterLab and execute each cell sequentially. Make sure to have the CSV 
  file (train.csv) in the same directory.

4 - Interpret the Results:
   After running the notebook, observe the output to understand the data distribution, model training process, and evaluation results.


## File Structure
  `NLP Twitter Disaster Classifier.ipynb`: Jupyter Notebook containing the code for data exploration, preprocessing, model building,
  and evaluation.
  `train.csv`: CSV file containing the bank marketing dataset.
  README.md: This file providing instructions and information about the project.
  
## Contributors
Ahmed Eldesoky
## Licens

This project is licensed under the MIT License.

## Acknowledgements
- Kaggle for providing the dataset.
- NLTK and scikit-learn for providing essential NLP and machine learning tools.
