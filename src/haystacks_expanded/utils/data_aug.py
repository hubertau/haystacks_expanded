import openai
import pandas as pd
import glob
import textwrap
from pathlib import Path
import os
from loguru import logger
import numpy as np
import json
import configparser
import hashlib
from collections import defaultdict

def generate_sentence_hash(sentence):
    md5 = hashlib.md5()
    md5.update(sentence.encode('utf-8'))
    return md5.hexdigest()

def consolidate_annots(file_dir, glob_pattern, checkworthy_min, seed = 1):
    '''Function to consolidate annotations into one file.

    Ensure that your file has the following:
    * a column named 'Annotations' with null entries for not check-worthy and 1 for those that are.
    * a column named 'sentence' with the original sentence.
    '''

    np.random.seed(seed)
    assert os.path.isdir(file_dir)
    assert 0 < checkworthy_min < 1

    full_pattern = os.path.join(file_dir, glob_pattern)
    logger.info(f'Glob pattern is: {full_pattern}')
    files = glob.glob(full_pattern)


    positive = []
    negative = []
    for file in files:
        logger.info(f'Processing {file}')
        if Path(file).suffix == '.xlsx':
            file_df = pd.read_excel(file)
        else:
            file_df = pd.read_csv(file)

        annotation_name = f'{"Annotations" if "Annotations" in file_df.columns else "Annotation"}'
        # print(file_df.columns)
        # print(file_df[file_df['Annotations']==1])
        positive.extend(file_df[file_df[annotation_name]==1]['sentence'].to_list())

        # Step 1: Split the DataFrame into two groups
        group_1_df = file_df[(file_df[annotation_name]!=1) & (file_df['Check-worthy Factual'] > checkworthy_min)]
        group_2_df = file_df[file_df['Check-worthy Factual'] <= checkworthy_min]

        # Step 2: Determine the size of the smaller group
        min_group_size = min(len(group_1_df), len(group_2_df))

        # Step 3: Sample an equal number of rows from each group
        sampled_group_1 = group_1_df.sample(n=min_group_size, random_state=42)
        sampled_group_2 = group_2_df.sample(n=min_group_size, random_state=42)

        # Step 4: Combine the sampled DataFrames if needed
        sampled_df = pd.concat([sampled_group_1, sampled_group_2])
        sampled_sentences = sampled_df['sentence'].to_list()

        negative.extend(sampled_sentences)

    # Create two separate DataFrames for positive and negative sentences
    positive_df = pd.DataFrame({'sentence': positive, 'label': 1})
    negative_df = pd.DataFrame({'sentence': negative, 'label': 0})

    # Concatenate the two DataFrames into a single DataFrame
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Filter rows that are not strings
    combined_df = combined_df[combined_df['sentence'].apply(lambda x: isinstance(x, str))]

    # Generate hashes as unique ids
    combined_df['hash'] = combined_df['sentence'].apply(generate_sentence_hash)

    # Create a mask to identify rows with unique hash values
    unique_hash_mask = ~combined_df.duplicated(subset='hash', keep='first')

    # Filter the DataFrame to keep only rows with unique hash values
    unique_df = combined_df[unique_hash_mask]

    rows_dropped = len(combined_df) - len(unique_hash_mask)

    # Finally drop sentences too long
    unique_df = unique_df[unique_df['sentence'].apply(len) < 500]

    savepath = Path(file_dir) / 'CONSOLIDATED.csv'
    unique_df.to_csv(savepath, index=False)
    logger.info(f'Duplicate rows dropped: {rows_dropped}')
    logger.info(f'Positive instances: {len(positive)}')
    logger.info(f'Negative instances: {len(negative)}')
    logger.info(f'Total length: {len(unique_df)}')
    logger.info(f'Saved to {savepath}')


def tts(total_train_file):
    '''Train test split, making sure no leakage on basis of data augmentation'''
    pass

from pathlib import Path

def get_save_path(path: str) -> Path:
    """Determine the correct save path for the output."""
    # Convert the input to a Path object
    p = Path(path)

    # If it's just a filename (no directory parts)
    if p.parent == Path('.'):
        return p.cwd() / p

    # If the relative folder exists
    if p.parent.exists():
        return p.resolve()

    # Raise an error if the directory doesn't exist
    raise ValueError(f"Directory {p.parent} does not exist.")


def api_augment(
        data_to_augment,
        api_config_file,
        outfile = None,
        batch_size = 10,
        augment_size = 10,
        up_to = None,
        overwrite = False,
        skip_existing = True,
    ):

    logger.info('Received Parameters:\n')
    logger.info(f'Data file: {data_to_augment}')
    logger.info(f'API Config File: {api_config_file}')
    logger.info(f'Specified outfile: {outfile}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Augment size: {augment_size}')
    logger.info(f'(for debug only) number of batches to process up to: {up_to}')
    logger.info(f'Overwrite: {overwrite}')
    logger.info(f'Skip Existing: {skip_existing}')

    # Read in data
    df = pd.read_csv(data_to_augment)

    # set output file dir an object name
    if not outfile:
        logger.info(f'No outfile path specified: saving to same folder as input data')
        outfile = Path(data_to_augment).parent.resolve() / f"{Path(data_to_augment).stem}_aug.json"
    else:
        # Test
        try:
            outfile = get_save_path(outfile)
        except ValueError as e:
            print(e)
    logger.info(f'Output to be saved to: {outfile}')

    # if outfile exists, read in
    if os.path.isfile(outfile):
        with open(outfile, 'r') as f:
            out_dict = json.load(f)
        out_dict = defaultdict(list, out_dict)
    else:
        out_dict = defaultdict(list)

    # define batch generator
    def batched_key_value_pairs(df, key_col, value_col, batch_size):
        total_rows = len(df)

        for i in range(0, total_rows, batch_size):
            keys = df[key_col].iloc[i:i+batch_size].tolist()
            values = df[value_col].iloc[i:i+batch_size].tolist()
            yield dict(zip(keys, values))

    # Use the generator
    filtered_df = df[(df['label'] == 1) & (df['sentence'].apply(len) < 500)]
    # print(len(filtered_df))
    input_iterator = batched_key_value_pairs(filtered_df, 'hash', 'sentence', batch_size)
    num_batches = len(list(input_iterator))
    input_iterator = batched_key_value_pairs(filtered_df, 'hash', 'sentence', batch_size)

    # get token
    config = configparser.ConfigParser()
    config.read(api_config_file)
    token = config['API']['token']
    openai.api_key = token

    # Do Augmentation
    instruction = textwrap.dedent(f"""You are an advanced AI writer to tasked with data augmentation to help detect check-worthy claims in sentences. Inaccuracies in data are not a problem.

    For each input sentence, Generate {augment_size} additional semantically similar sentences with the following constraints: numbers, percentages, and entities are changed  (e.g., '40%' could be changed to '35%', '45%', etc. - Note that inaccuracies are NOT a problem), and rephrasing is allowed. Changing to a similar topic is also allowed. Ensure there is substantial variety: for example, do not repeat the same company name or a percentage number n times. Input is formatted as [hash]:[original sentence] pairs. Format response as a json object, with hashes of the original sentences as keys and the {augment_size} generated sentences as values. Ensure that each original sentence has {augment_size} generated sentences.""")
    logger.info(f'Instruction: {instruction}')

    responses = []
    for counter, sentences in enumerate(input_iterator):
        logger.info(f'Processing batch {counter+1} of {num_batches}')

        # check existing
        if skip_existing:
            already_existing = [h in out_dict for h in sentences.keys()]
            if all(already_existing):
                logger.info(f'All items in batch {counter+1} are already present. Continuing...')
                continue
            elif any(already_existing):
                logger.info(f'{sum(already_existing)} found to already exist')
            # filter out existing ones
            sentences = {k: v for k, v in sentences.items() if k not in out_dict}

        if up_to and counter >= up_to:
            logger.info(f'Maximum batch number {up_to} reached. Ending...')
            break
        temp_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Sentences: {sentences}"}
            ],
        temperature = 0.8,
        frequency_penalty = 0.5,
        )

        responses.append(temp_response)

        for index, choice_element in enumerate(temp_response.choices):
            if choice_element['finish_reason'] != 'stop':
                logger.warning(f"Choice {index} of Completion ID {temp_response.id} finished because of: {choice_element['finish_reason']}")

            try:
                this_response = json.loads(choice_element['message']['content'])
            except json.decoder.JSONDecodeError as e:
                logger.error(choice_element['message']['content'])
                logger.error(e)
                break

            for k, v in this_response.items():
                if len(v) < 10:
                    logger.warning(f'Sentence {k} had {len(v)} generated sentences')
                if overwrite and len(out_dict[k]) > 0:
                    out_dict[k] = v
                else:
                    out_dict[k] = out_dict[k] + v
        else:
            with open(outfile, 'w' ) as f:
                json.dump(out_dict, f)
            continue
        break


if __name__ == '__main__':

    consolidate_annots('/Users/hubert/Drive/DPhil/Projects/2022-08b-DSF/haystacks_expanded/data/03_processed/annotated', '*annotated.*', 0.4)