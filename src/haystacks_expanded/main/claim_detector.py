'''
This script is responsible for the processing of claim detection. It should be modular for the rest of the pipeline.
'''

from loguru import logger
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)


class ClaimDetector:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def from_transformers(cls, model = "Nithiwat/mdeberta-v3-base_claimbuster", device='cpu'):
        #Nithiwat/mdeberta-v3-base_claimbuster
        #sschellhammer/SciTweets_SciBert

        pipe = pipeline(
            task='text-classification',
            model = model,
            top_k=None,
            device=device
        )

        return ClaimDetector(pipe)

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    def detect(self, video_feature_directory):
        pass

if __name__ == "__main__":

    test_sentences = [
        'I like cheese',
        'Covid-19 is a virus and not a bacterial disease'
    ]

    test_pipeline = ClaimDetector.from_transformers()

    print(test_pipeline(test_sentences))