'''
This script is responsible for the processing of claim detection. It should be modular for the rest of the pipeline.
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from loguru import logger


class ClaimDetector:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def from_transformers(cls, name = "sschellhammer/SciTweets_SciBert"):

        pipe = pipeline(task='text-classification', model = name, top_k=None)

        return ClaimDetector(pipe)

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)


if __name__ == "__main__":

    test_sentences = [
        'I like cheese',
        'Covid-19 is a virus and not a baterial disease'
    ]

    test_pipeline = ClaimDetector.from_transformers()

    print(test_pipeline(test_sentences))