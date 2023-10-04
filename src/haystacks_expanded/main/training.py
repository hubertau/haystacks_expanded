from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from loguru import logger
from transformers import TrainingArguments, Trainer, EvalPrediction, GenerationConfig, LlamaTokenizer, LlamaForSequenceClassification, EarlyStoppingCallback
import numpy as np
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

import transformers
from typing import List
from torch.utils.data import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
import json
import fire
from pathlib import Path
import torch
from datasets import load_dataset
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

from ..utils import get_save_path

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

def combine_original_and_aug(original_file, aug_file, outfile = None):

    # set output file dir an object name
    if not outfile:
        logger.info(f'No outfile path specified: saving to same folder as input data')
        outfile = Path(original_file).parent.resolve() / f"{Path(original_file).stem}_ORIG+AUG.csv"
    else:
        # Test
        try:
            outfile = get_save_path(outfile)
        except ValueError as e:
            print(e)
    logger.info(f'Output to be saved to: {outfile}')

    with open(aug_file, 'r') as f:
        aug = json.load(f)

    df = pd.read_csv(original_file)

        # Step 1: Set the 'original' column in the DataFrame to 1
    df['original'] = 1

    # Step 2: Convert JSON data to DataFrame format
    hashes = []
    sentences = []
    for k, v in aug.items():
        for sentence in v:
            hashes.append(k)
            sentences.append(sentence)

    json_df = pd.DataFrame({
        'hash': hashes,
        'sentence': sentences,
        'label': [1] * len(hashes)
    })

    # Step 3: Set the 'original' column in the JSON DataFrame to 0
    json_df['original'] = 0

    # Step 4: Append the JSON DataFrame to the original DataFrame
    final_df = pd.concat([df, json_df], ignore_index=True)

    logger.info(f'Total Number of datapoints: {len(final_df)}')
    logger.info(f'+ve label: {len(final_df[final_df["label"]==1])}')
    logger.info(f'-ve label: {len(final_df[final_df["label"]==0])}')

    final_df.to_csv(outfile, index=False)

def make_tdt_split(combined_orig_aug, BASE_MODEL, outfile = None, MAX_LEN = 128):

    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"

    if not outfile:
        logger.info(f'No outfile path specified: saving to same folder as input data')
        outfile = Path(combine_original_and_aug).parent.resolve() / f"{Path(combine_original_and_aug).stem}_split"
    else:
        # Test
        try:
            outfile = get_save_path(outfile)
        except ValueError as e:
            print(e)
    logger.info(f'Output to be saved to: {outfile}')

    df = pd.read_csv(combined_orig_aug)

    # make sure labels are int
    df['label'] = df['label'].astype(int)

    # Shuffle the unique hashes
    unique_hashes = df['hash'].unique()
    np.random.shuffle(unique_hashes)

    # Calculate target number of rows for each split
    total_rows = len(df)
    train_target_rows = int(0.7 * total_rows)
    test_target_rows = int(0.2 * total_rows)

    train_hashes = []
    test_hashes = []
    dev_hashes = []

    cumulative_train_rows = 0
    cumulative_test_rows = 0

    for h in unique_hashes:
        h_rows = len(df[df['hash'] == h])
        if cumulative_train_rows + h_rows <= train_target_rows:
            train_hashes.append(h)
            cumulative_train_rows += h_rows
        elif cumulative_test_rows + h_rows <= test_target_rows:
            test_hashes.append(h)
            cumulative_test_rows += h_rows
        else:
            dev_hashes.append(h)

    # Split DataFrame based on these hashes
    train_df = df[df['hash'].isin(train_hashes)]
    test_df = df[df['hash'].isin(test_hashes)]
    dev_df = df[df['hash'].isin(dev_hashes)]

    logger.info(f'Train set size: {len(train_df)}')
    logger.info(f'Dev set size: {len(dev_df)}')
    logger.info(f'Test set size: {len(test_df)}')

    # Creating our train-val-test datasets
    train_ds = LlamaDataset(
        input_data=train_df['sentence'].to_numpy(),
            sentiment_targets=train_df['label'].to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )
    dev_ds = LlamaDataset(
        input_data=dev_df['sentence'].to_numpy(),
            sentiment_targets=dev_df['label'].to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )

    test_ds = LlamaDataset(
        input_data=test_df['sentence'].to_numpy(),
            sentiment_targets=test_df['label'].to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )

    data = DatasetDict({
        'train': train_ds,
        'dev': dev_ds,
        'test': test_ds
    })

    data.save_to_disk(outfile)

def train(dataset_dict, OUTPUT_DIR, BASE_MODEL = None):

    data = DatasetDict.load_from_disk(dataset_dict)

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
    tokenizer.padding_side = "left"

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
    TRAIN_STEPS = 3000

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

    training_arguments = TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
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
        remove_unused_columns=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        early_stopping_patience=3,
    )

    trainer = transformers.Trainer(
        model=model_llama,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        args=training_arguments,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 10)]
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
    trainer.save_model()

    # If you want to evaluate the trainer run the code below
    # predictions = trainer.predict(data['test'])

if __name__=='__main__':
    pass