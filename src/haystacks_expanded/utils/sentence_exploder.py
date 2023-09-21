'''Simple script to take a csv with paragraphs in a column and explode them
'''

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path

def sent_explode_func(infile):
    infile = Path(infile)
    df = pd.read_csv(infile, names = [
        'debate id', 'motion speaker id', 'motion speaker name', 'motion speaker party', 'debate title', 'motion text', 'speaker id', 'speaker name', 'speaker party', 'vote', 'utterance'])

    df['sentence'] = df['utterance'].apply(sent_tokenize)

    df_exploded = df.explode('sentence')
    df_exploded = df_exploded.drop(columns='utterance')

    df_exploded.to_csv(infile.parent / (infile.stem + "_exploded" + infile.suffix))

if __name__ == '__main__':
    sent_explode_func()