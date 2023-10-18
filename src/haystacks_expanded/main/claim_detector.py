from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path
import gc

class ClaimDetector:

    def __init__(self, model_name, tok_name = None, modeltype='LLM', short_name = None, device='cpu', **kwargs):
        if tok_name is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if modeltype == 'LLM':
            model_name = Path(model_name)
            logger.debug(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map = device,
                load_in_8bit=True,
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_use_double_quant=False,
                ),local_files_only=True, **kwargs)
            self.model.config.pad_token_id = 0
            self.tokenizer.pad_token_id = 0
            if os.path.isfile(f"{model_name}/score.original_module.pt"):
                score_weights = torch.load(f"{model_name}/score.original_module.pt", map_location='cuda')
                self.model.score.load_state_dict(score_weights)
                logger.info(f'Score weights loaded')
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map = device, **kwargs)
        self.device = device

        # Check if label2id and id2label are set, if not, set default values
        if not hasattr(self.model.config, "label2id") or not self.model.config.label2id:
            logger.info(f'No label2id found. Setting default...')
            self.model.config.label2id = {
                "Not Checkworthy": 0,
                "Checkworthy": 1
            }
        if short_name is not None:
            self.model.config.label2id = {f"{short_name}-{k}":v for k, v in self.model.config.label2id.items()}

        self.model.config.id2label = {v: k for k, v in self.model.config.label2id.items()}

        self.model.config.temperature=0

        self.id2label = self.model.config.id2label

        logger.debug(self.model.config.label2id)

    def __call__(self, dataset, batch_size=32):
        # Tokenize the sentences
        def tokenize_function(examples):
            return self.tokenizer(examples["sentence"], padding="max_length", truncation=True, return_tensors='pt', max_length=128)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        # Convert the tokenized dataset to PyTorch tensors
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        # tokenized_sentences = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
        # input_ids = torch.tensor(tokenized_sentences['input_ids'])
        # attention_mask = torch.tensor(tokenized_sentences['attention_mask'])
        # dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)
        if isinstance(self.device, dict):
            to_dev = 'cuda'
        else:
            to_dev = self.device

        results = []
        for batch in dataloader:
            logger.debug(batch)
            inputs = {key: val.to(to_dev) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = logits.softmax(dim=1)

            for prob in probs:
                results.append([{'label': self.id2label[idx], 'score': score.item()} for idx, score in enumerate(prob)])

            del batch
            gc.collect()
            torch.cuda.empty_cache()
        return results

if __name__ == "__main__":

    test_sentences = [
        'I like cheese',
        'Covid-19 is a virus and not a bacterial disease'
    ]

    detector = ClaimDetector(model_name="Nithiwat/mdeberta-v3-base_claimbuster")
    print(detector(test_sentences))
