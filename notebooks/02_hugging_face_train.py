# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## NLP Coding Session: Part 2
# MAGIC 
# MAGIC In this notebook, we will use classic ML models for training an intent prediction model. We will use the [banking77](https://huggingface.co/datasets/banking77) dataset from Hugging Face, along with the following:
# MAGIC 
# MAGIC * Hugging Face

# COMMAND ----------

# DBTITLE 1,Hugging Face Example
import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

tokenized_segments = tokenizer(["I didn't like this movie"], return_tensors="pt", padding=True, truncation=True)
tokenized_segments_input_ids, tokenized_segments_attention_mask = tokenized_segments.input_ids, tokenized_segments.attention_mask
model_predictions = F.softmax(model(input_ids=tokenized_segments_input_ids, attention_mask=tokenized_segments_attention_mask)['logits'], dim=1)

print("Positive probability: "+str(model_predictions[0][1].item()*100)+"%")
print("Negative probability: "+str(model_predictions[0][0].item()*100)+"%")

# COMMAND ----------

# DBTITLE 1,Downloading our Dataset
!pip install datasets

from datasets import load_dataset

dataset = load_dataset("banking77")

# COMMAND ----------

# DBTITLE 1,Training DistilBERT
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, PretrainedConfig

num_labels = len(dataset["train"].info.features["label"].names)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = num_labels)
model

# COMMAND ----------

from transformers import DataCollatorWithPadding

def preprocess_function(examples):
  return tokenizer(examples["text"], truncation=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# COMMAND ----------

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# COMMAND ----------

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='/tmp/results',
    learning_rate=2e-3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# COMMAND ----------

encoded = tokenizer("lost my credit card", return_tensors = "pt")
pred = model(**encoded)
pred

# COMMAND ----------

np.argmax(pred.logits.detach().numpy())
