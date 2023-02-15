# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## NLP Coding Session
# MAGIC 
# MAGIC In this notebook, we will use classic ML models for training an intent prediction model. We will use the [banking77](https://huggingface.co/datasets/banking77) dataset from Hugging Face, along with the following:
# MAGIC 
# MAGIC * Scikit-Learn
# MAGIC   * FunctionTransformer
# MAGIC   * TfIdfVectorizer
# MAGIC * NLTK

# COMMAND ----------

# DBTITLE 1,Downloading our Dataset
!pip install datasets

from datasets import load_dataset

dataset = load_dataset("banking77")

# COMMAND ----------

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

train_df

# COMMAND ----------

dataset["train"].info.features["label"].names

# COMMAND ----------

# DBTITLE 1,Preprocessing our Data
# We will do some basic preprocessing:
# 1. Converting text to lowercase
# 2. Removing stopwords
# 3. Performing lemmatization

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text_series: pd.Series) -> pd.Series:

  # Lower case and remove stopwords
  text_series = text_series.str.lower()
  output = []

  for item in text_series.values:
    word_tokens = tokenizer.tokenize(item)
    filtered_tokens = []

    for token in word_tokens:
      if token not in stop_words:
        filtered_tokens.append(token)

    text = " ".join(filtered_tokens)
    text = lemmatizer.lemmatize(text)
    output.append(text)

  text_series = output
  return text_series

# COMMAND ----------

test_sentence = pd.DataFrame({"text": ["The book, on the table it is!"]})
preprocess(test_sentence["text"])

# COMMAND ----------

train_df

# COMMAND ----------

# DBTITLE 1,Creating a Function Transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

preprocessor = FunctionTransformer(preprocess)
preprocessor.fit_transform(train_df["text"])

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow

mlflow.autolog(False)

clf = LogisticRegression()
vectorizer = TfidfVectorizer()

pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("vectorizer", vectorizer),
  ("clf", clf)
])

with mlflow.start_run(run_name = "sklearn_pipeline_1", nested = True) as run:

  pipeline.fit(train_df["text"], train_df["label"])
  pred = pipeline.predict(test_df["text"])
  prob = pipeline.predict_proba(test_df["text"])

  metrics = {
    "accuracy_test": accuracy_score(test_df['label'], pred),
    "f1_test": f1_score(test_df['label'], pred, average = 'weighted'),
    "roc_test": roc_auc_score(
      y_true = test_df['label'].values,
      y_score = prob,
      average = 'weighted',
      multi_class = 'ovr'
    )
  }

  mlflow.log_metrics(metrics)
