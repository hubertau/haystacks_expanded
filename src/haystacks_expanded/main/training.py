import json
import os
import warnings
from pathlib import Path
from typing import List

import evaluate
import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformers
from datasets import Dataset as HF_Dataset
from datasets import DatasetDict, load_dataset
from loguru import logger
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  prepare_model_for_kbit_training)
from pylab import rcParams
from sentence_transformers import SentenceTransformer
# from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertForSequenceClassification, BertTokenizerFast,
                          EarlyStoppingCallback, BitsAndBytesConfig, EvalPrediction, GenerationConfig, LlamaForSequenceClassification,
                          LlamaTokenizer, Trainer, TrainingArguments)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Filter out the specific warning
warnings.filterwarnings("ignore", message="Some weights of the model were not initialized from the model checkpoint")

from ..utils import get_save_path

# # Defining a Dataset object to put our data in
# class LlamaDataset(HF_Dataset):
#     """
#     Special dataset class built on top of the torch Dataset class
#     useful to have memory efficient dataloading tokenization batching and trainning.

#     Huggingface can use these types of dataset as inputs and run all trainning/prediction on them.
#     """
#     def __init__(self, input_data, targets, tokenizer, max_len):
#         """
#         Basic generator function for the class.
#         -----------------
#         input_data : array
#             Numpy array of string  input text to use for downstream task
#         targets :
#             Numpy array of integers indexed in  the pytorch style of [0,C-1] with C being the total number of classes
#             In our example this means the target sentiments should range from 0 to 2.
#         tokenizer  : Huggingface tokenizer
#             The huggingface tokenizer to use
#         max_len :
#             The truncation length of the tokenizer
#         -------------------

#         Returns :

#             Tokenized text with inputs, attentions and labels, ready for the Training script.
#         """
#         self.input_data = input_data
#         self.targets = targets
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         """
#         Function required by torch huggingface to batch efficiently
#         """
#         return len(self.input_data)

#     def __getitem__(self, item):
#         text = str(self.input_data[item])
#         target = self.targets[item]
#         # only difference with the previuous tokenization step is the encode-plus for special tokens
#         encoding = self.tokenizer.encode_plus(
#           text,
#           add_special_tokens=True,
#           max_length=self.max_len,
#           return_token_type_ids=False,
#           padding='max_length',
#           return_attention_mask=True,
#           return_tensors='pt',
#           truncation = True
#         )
#         return {
#          # 'text': text,
#           'input_ids': encoding['input_ids'].flatten(),
#           'attention_mask': encoding['attention_mask'].flatten(),
#           'labels': torch.tensor(target, dtype=torch.long)
#         }

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

def make_tdt_split(combined_orig_aug, BASE_MODEL, model_type = 'LLM', outfile = None, MAX_LEN = 128):
    logger.debug(f'max length is set to {MAX_LEN}')

    if model_type == 'LLM':
        tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.padding_side = "left"
    elif model_type == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

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

    # Convert DataFrames directly to HuggingFace's dataset format
    train_ds = HF_Dataset.from_pandas(train_df[['sentence', 'label']])
    dev_ds = HF_Dataset.from_pandas(dev_df[['sentence', 'label']])
    test_ds = HF_Dataset.from_pandas(test_df[['sentence', 'label']])

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')

    train_ds = train_ds.map(tokenize_function, batched=True)
    dev_ds = dev_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    train_ds.set_format(type='torch', columns=['input_ids', 'label', 'attention_mask'])
    train_ds = train_ds.rename_column('label', 'labels')
    dev_ds.set_format(type='torch', columns=['input_ids', 'label', 'attention_mask'])
    dev_ds = dev_ds.rename_column('label', 'labels')
    test_ds.set_format(type='torch', columns=['input_ids', 'label', 'attention_mask'])
    test_ds = test_ds.rename_column('label', 'labels')

    data = DatasetDict({
        'train': train_ds,
        'dev': dev_ds,
        'test': test_ds
    })

    data.save_to_disk(outfile)

def train_model(dataset_dict, OUTPUT_DIR, BASE_MODEL = None, batch_size=16, resume = True, num_train_epochs=20, esp = None, savelim=10):

    logger.info(f'base model is {BASE_MODEL}')

    data = DatasetDict.load_from_disk(dataset_dict)

    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    model = LlamaForSequenceClassification.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
        ),
        device_map="auto",
    )
    model.config.pad_token_id = 0

    LORA_R = 4
    LORA_ALPHA = 16
    LORA_DROPOUT= 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]

    MICRO_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = batch_size // MICRO_BATCH_SIZE
    LEARNING_RATE = 5e-5
    TRAIN_STEPS = 3000

    torch.cuda.empty_cache()
    assert torch.cuda.is_available()

    # model_llama = prepare_model_for_kbit_training(model)
    # config = LoraConfig(
    #     r=LORA_R,
    #     lora_alpha=LORA_ALPHA,
    #     target_modules=LORA_TARGET_MODULES,
    #     lora_dropout=LORA_DROPOUT,
    #     bias="none",
    #     task_type="SequenceClassification",
    # )
    # model_llama = get_peft_model(model_llama, config)
    # # get parameter efficient fine tuning representation of our model
    # model_llama.config.use_cache = False
    # old_state_dict = model_llama.state_dict
    # model_llama.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model_llama, type(model_llama))
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        # modules_to_save=["classifier"],
        task_type="SEQ_CLS"
    )
    model = prepare_model_for_kbit_training(model)
    model_llama = get_peft_model(model, config)

    # set model id2label
    model_llama.config.id2label = {
        0: "Not Checkworthy",
        1: "Checkworthy"
    }

    training_arguments = TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        # max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        disable_tqdm=True,
        eval_steps=50,
        save_steps=50,
        output_dir=OUTPUT_DIR,
        save_total_limit=savelim,
        load_best_model_at_end=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        label_names=['labels']
    )

    # model_llama = torch.compile(model_llama)

    callbacks = [EarlyStoppingCallback(early_stopping_patience = esp)]
    if esp == 0 or esp is None:
        callbacks = None
        logger.info(f'No callbacks are set')

    trainer = Trainer(
        model=model_llama,
        tokenizer=tokenizer,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        args=training_arguments,
        compute_metrics=compute_metrics,
        callbacks = callbacks,
    )

    with torch.autocast("cuda"):
        trainer.train(resume_from_checkpoint = resume)
    logger.info(f'Best checkpoint: {trainer.state.best_model_checkpoint}')
    trainer.save_model()

    # If you want to evaluate the trainer run the code below
    # predictions = trainer.predict(data['test'])

def train_bert_model(dataset_dict, OUTPUT_DIR, BASE_MODEL = None, batch_size=16, resume = True, num_train_epochs=20, esp = None, savelim=10):

    logger.info(f'Running BERT model training')
    logger.info(f'base model is {BASE_MODEL}')

    data = DatasetDict.load_from_disk(dataset_dict)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        # torch_dtype=torch.float16,
        device_map="cuda"
    )

    # change num_labels to 2
    logger.info('Converting num labels to 2')
    config = model.config
    config.num_labels = 2
    model.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
    model.num_labels = 2
    model.config = config

    MICRO_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = batch_size // MICRO_BATCH_SIZE
    LEARNING_RATE = 5e-5
    TRAIN_STEPS = 3000

    torch.cuda.empty_cache()
    assert torch.cuda.is_available()

    training_arguments = TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        # max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        # fp16=True,
        disable_tqdm=True,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        output_dir=OUTPUT_DIR,
        save_total_limit=savelim,
        load_best_model_at_end=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # set model id2label
    model.config.id2label = {
        0: "Not Checkworthy",
        1: "Checkworthy"
    }
    logger.info(f'id2label: {model.config.id2label}')

    callbacks = [EarlyStoppingCallback(early_stopping_patience = esp)]
    if esp == 0 or esp is None:
        callbacks = None
        logger.info(f'No callbacks are set')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        args=training_arguments,
        compute_metrics=compute_metrics,
        callbacks = callbacks,
    )

    with torch.autocast("cuda"):
        trainer.train(resume_from_checkpoint = resume)

    logger.info(f'Best checkpoint: {trainer.state.best_model_checkpoint}')
    trainer.save_model()

if __name__=='__main__':
    pass