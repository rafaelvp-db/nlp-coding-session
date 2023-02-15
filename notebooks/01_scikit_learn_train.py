# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## NLP Coding Session: Part 1
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

def preprocess(df: pd.DataFrame) -> pd.Series:

  # Lower case and remove stopwords
  text_series = df["text"].str.lower()
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

  return output

# COMMAND ----------

test_sentence = pd.DataFrame({"text": ["The book, on the table it is!"]})
preprocess(test_sentence)

# COMMAND ----------

# DBTITLE 1,Creating a Function Transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

preprocessor = FunctionTransformer(preprocess)
preprocessor.fit_transform(train_df)

# COMMAND ----------

# DBTITLE 1,Training our Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow

clf = LogisticRegression()
vectorizer = TfidfVectorizer()

pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("vectorizer", vectorizer),
  ("clf", clf)
])

with mlflow.start_run(run_name = "sklearn_pipeline_1", nested = True) as run:

  pipeline.fit(train_df.loc[:, ["text"]], train_df["label"])
  pred = pipeline.predict(test_df.loc[:, ["text"]])
  prob = pipeline.predict_proba(test_df.loc[:, ["text"]])

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
  #mlflow.sklearn.log_model(pipeline, "sklearn_pipeline")

# COMMAND ----------

# DBTITLE 1,Visualizing our Metrics
metrics = mlflow.search_runs(
  filter_string = "status = 'FINISHED'",
  order_by = ["metrics.f1_score_test DESC"])

metrics

# COMMAND ----------

# DBTITLE 1,Registering our Model
target_run_id = metrics.loc[0, "run_id"]
target_run_id

# COMMAND ----------

mlflow.register_model(
  f"runs:/{target_run_id}/model",
  "sklearn_chatbot"
)

# COMMAND ----------

# DBTITLE 1,Non-NLTK Version
pipeline_no_nltk = Pipeline([
  ("vectorizer", vectorizer),
  ("clf", clf)
])

# COMMAND ----------

with mlflow.start_run(run_name = "sklearn_pipeline_no_nltk", nested = True) as run:

  pipeline_no_nltk.fit(train_df["text"], train_df["label"])
  pred = pipeline_no_nltk.predict(test_df["text"])
  prob = pipeline_no_nltk.predict_proba(test_df["text"])

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

# COMMAND ----------

metrics = mlflow.search_runs(
  filter_string = "attributes.run_name = 'sklearn_pipeline_no_nltk' and status = 'FINISHED'",
  order_by = ["metrics.f1_score_test DESC"])

metrics

# COMMAND ----------

target_run_id = metrics.loc[0, "run_id"]
target_run_id

# COMMAND ----------

mlflow.register_model(
  f"runs:/{target_run_id}/model",
  "sklearn_chatbot_no_nltk"
)

# COMMAND ----------


