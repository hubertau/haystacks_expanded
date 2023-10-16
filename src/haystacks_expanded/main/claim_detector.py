from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class ClaimDetector:

    def __init__(self, model_name, tok_name = None, short_name = None, device='cpu', **kwargs):
        if tok_name is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs).to(device)
        self.device = device

        # Check if label2id and id2label are set, if not, set default values
        if not hasattr(self.model.config, "label2id") or not self.model.config.label2id:
            logger.info(f'No id2label found. Setting default...')
            self.model.config.label2id = {
                "Not Checkworthy": 0,
                "Checkworthy": 1
            }
        if short_name is not None:
            self.model.config.label2id = {f"{short_name}-{k}":v for k, v in self.model.config.label2id.items()}

        self.model.config.id2label = {v: k for k, v in self.model.config.label2id.items()}

        self.model.config.temperature=0

        self.id2label = self.model.config.id2label

    def __call__(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = logits.softmax(dim=1)

        results = []
        for prob in probs:
            results.append([{'label': self.id2label[idx], 'score': score.item()} for idx, score in enumerate(prob)])

        return results

if __name__ == "__main__":

    test_sentences = [
        'I like cheese',
        'Covid-19 is a virus and not a bacterial disease'
    ]

    detector = ClaimDetector(model_name="Nithiwat/mdeberta-v3-base_claimbuster")
    print(detector(test_sentences))
