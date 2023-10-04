from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from loguru import logger
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
import numpy as np
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaForSequenceClassification

from sklearn.metrics import confusion_matrix
import os
import sys
from typing import List
from torch.utils.data import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from tqdm import tqdm
import numpy as np
import json
from google.colab import drive
import fire
import torch
from datasets import load_dataset
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
import torch
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np



# define the compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    # calculate macro-averaged precision, recall, and F1 score
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # auc = roc_auc_score(labels, preds, multi_class='ovr')

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        # 'auc': auc
    }

# Defining a Dataset object to put our data in
class LlamaDataset(Dataset):
    """
    Special dataset class built on top of the torch Dataset class
    useful to have memory efficient dataloading tokenization batching and trainning.

    Huggingface can use these types of dataset as inputs and run all trainning/prediction on them.
    """
    def __init__(self, input_data, sentiment_targets, tokenizer, max_len):
        """
        Basic generator function for the class.
        -----------------
        input_data : array
            Numpy array of string  input text to use for downstream task
        sentiment_targets :
            Numpy array of integers indexed in  the pytorch style of [0,C-1] with C being the total number of classes
            In our example this means the target sentiments should range from 0 to 2.
        tokenizer  : Huggingface tokenizer
            The huggingface tokenizer to use
        max_len :
            The truncation length of the tokenizer
        -------------------

        Returns :

            Tokenized text with inputs, attentions and labels, ready for the Training script.
        """
        self.input_data = input_data
        self.sentiment_targets = sentiment_targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Function required by torch huggingface to batch efficiently
        """
        return len(self.input_data)

    def __getitem__(self, item):
        text = str(self.input_data[item])
        target = self.sentiment_targets[item]
        # only difference with the previuous tokenization step is the encode-plus for special tokens
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation = True
        )
        return {
         # 'text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(target, dtype=torch.long)
        }

df['label']= df['label'].astype(int)
# to make the task comparable to the Bayesian classifier above we will also drop the neutral class

df = df[df['label']!=2]
df = df#.sample(frac=0.10)
train, test = train_test_split(df, test_size=0.3, random_state=123)
test, val = train_test_split(test, test_size=0.5, random_state=123

BASE_MODEL = "decapoda-research/llama-2-7b-hf"

model = LlamaForSequenceClassification.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left

LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 5e-5
TRAIN_STEPS = 3 # the actual parameter that determines how long you train for
#                 today, we are only running this model for a very small number of trainning steps
#                 this is just to give you an idea of how to run these models.
OUTPUT_DIR = "experiments"

torch.cuda.empty_cache()
torch.cuda.is_available()

model_llama = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="SequenceClassification",
)
model_llama = get_peft_model(model_llama, config)
model_llama.print_trainable_parameters()


training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard",
    remove_unused_columns=False
)

# Creating our train-val-test datasets
MAX_LEN = 128
train_ds = LlamaTutorialDataset(
    input_data=train['sentence'].to_numpy(),
        sentiment_targets=train['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
val_ds = LlamaTutorialDataset(
    input_data=val['sentence'].to_numpy(),
        sentiment_targets=val['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

test_ds = LlamaTutorialDataset(
    input_data=test['sentence'].to_numpy(),
        sentiment_targets=test['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

trainer = transformers.Trainer(
    model=model_llama,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=training_arguments,
)

# get parameter efficient fine tuning representation of our model
model_llama.config.use_cache = False
old_state_dict = model_llama.state_dict
model_llama.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model_llama, type(model_llama))

# compiling the model ahead of running it
model_llama = torch.compile(model_llama)

trainer.train()

# If you want to evaluate the trainer run the code below
predictions = trainer.predict(test_ds