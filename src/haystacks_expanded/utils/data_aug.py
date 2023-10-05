import configparser
import glob
import hashlib
import json
import os
import textwrap
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from loguru import logger


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

def send_instruction(augment_size, sentences):

    # Do Augmentation
    instruction = textwrap.dedent(f"""You are an advanced AI writer to tasked with data augmentation to help detect check-worthy claims in sentences. Inaccuracies in data are not a problem.

    For each input sentence, Generate {augment_size} additional semantically similar sentences with the following constraints: numbers, percentages, and entities are changed  (e.g., '40%' could be changed to '35%', '45%', etc. - Note that inaccuracies are NOT a problem), and rephrasing is allowed. Changing to a similar topic is also allowed. Ensure there is substantial variety: for example, do not repeat the same company name or a percentage number n times. Input is formatted as [hash]:[original sentence] pairs.

    Format response as a valid json object with double quotes, with hashes of the original sentences as keys and the {augment_size} generated sentences as values. Ensure that each original sentence has {augment_size} generated sentences.""")

    r = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Sentences: {sentences}"}
            ],
        temperature = 0.8,
        frequency_penalty = 0.5,
    )

    return r

def api_augment(
        data_to_augment,
        api_config_file,
        outfile = None,
        batch_size = 10,
        augment_size = 10,
        up_to = None,
        overwrite = False,
        skip_existing = True,
        from_batch = None,
        supplement = None
    ):
    """Create Data Augmentation Using OpenAI API

    Args:
        data_to_augment (str): csv of hash, sentence, label
        api_config_file (str): config file with API token.
        outfile (str, optional): savename for output file. Defaults to None.
        batch_size (int, optional): Number of sentences to send each time to API. Defaults to 10.
        augment_size (int, optional): How many sentences to generate for each sentence. Defaults to 10.
        up_to (int, optional): Process up to this batch number. For debugging. Defaults to None.
        overwrite (bool, optional): Boolean flag to overwrite existing augmentations. Defaults to False.
        skip_existing (bool, optional): Skip existing. Defaults to True.
        from_batch (int, optional): Start from this batch number. Defaults to None.
        supplement (str, optional): Existing augmentation json file to supplement. Defaults to None.

    Yields:
        None: Saves data to outfile
    """

    # Verbose logging
    logger.info('Received Parameters:\n')
    logger.info(f'Data file: {data_to_augment}')
    logger.info(f'API Config File: {api_config_file}')
    logger.info(f'Specified outfile: {outfile}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Augment size: {augment_size}')
    logger.info(f'Starting from batch number: {from_batch}')
    logger.info(f'(for debug only) number of batches to process up to: {up_to}')
    logger.info(f'Supplement: {supplement}')
    logger.info(f'Overwrite: {overwrite}')
    logger.info(f'Skip Existing: {skip_existing}')

    if skip_existing and supplement:
        logger.warning('Supplement and skip_existing cannot both be set to True. Setting skip_existing to False and continuing...')
        skip_existing=False

    # Get Token and setup API
    config = configparser.ConfigParser()
    config.read(api_config_file)
    token = config['API']['token']
    openai.api_key = token

    # Read in data
    df = pd.read_csv(data_to_augment)

    # if supplement is true, read in existing data
    if supplement:
        with open(supplement, 'r') as f:
            supp_json = json.load(f)

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
    def batched_key_value_pairs(df, key_col, value_col, batch_size, supp_size = None):
        """Generate iterator to feed into API

        Args:
            df: Dataframe with 'hash', 'label', and 'sentence'
            key_col (str): column for keys
            value_col (_type_): column for values
            batch_size (_type_): size of each output batch
            supp_size (_type_): how many to supplement for this set of batched outputs

        Yields:
            tuple: batch_dict, supp_size
        """
        total_rows = len(df)

        for i in range(0, total_rows, batch_size):
            keys = df[key_col].iloc[i:i+batch_size].tolist()
            values = df[value_col].iloc[i:i+batch_size].tolist()
            yield dict(zip(keys, values)), supp_size

    # GET INPUT DATA
    if supplement:

        # Read in existing augmentations and how many each sentence has
        existing_ids = defaultdict(list)
        for k, v in supp_json.items():
            if len(v) < augment_size:
                existing_ids[len(v)].append(k)

        gen_to_chain = []
        num_batches = 0
        # get iterators for each set of sentences
        for existing_aug_count, associated_ids in existing_ids.items():
            logger.info(f'Creating generators for supplement size {augment_size-existing_aug_count}: {len(associated_ids)} sentences')
            for_counting = batched_key_value_pairs(
                df = df[df['hash'].isin(associated_ids)],
                key_col='hash',
                value_col='sentence',
                batch_size=batch_size,
                supp_size=augment_size-existing_aug_count
            )
            num_batches += len(list(for_counting))
            gen_to_chain.append(batched_key_value_pairs(
                df = df[df['hash'].isin(associated_ids)],
                key_col='hash',
                value_col='sentence',
                batch_size=batch_size,
                supp_size=augment_size-existing_aug_count
            ))

        # chain created generators together
        input_iterator = chain(*gen_to_chain)
        logger.debug(f'Number of batches: {num_batches}')

    else:

        # Use the generator
        filtered_df = df[(df['label'] == 1) & (df['sentence'].apply(len) < 500)]
        # print(len(filtered_df))
        input_iterator = batched_key_value_pairs(filtered_df, 'hash', 'sentence', batch_size)
        num_batches = len(list(input_iterator))
        logger.debug(num_batches)
        input_iterator = batched_key_value_pairs(filtered_df, 'hash', 'sentence', batch_size)

    responses = []
    for counter, (sentences, ss) in enumerate(input_iterator, start=1):

        if from_batch and counter < from_batch:
            continue

        logger.info(f'Processing batch {counter} of {num_batches}')

        # check existing. Note check existing CANNOT be true when we are supplementing, because the whole point is to supplement EXISTING augmentations that do not have enough.
        if skip_existing:
            already_existing = [h in out_dict for h in sentences.keys()]
            if all(already_existing):
                logger.info(f'All items in batch {counter} are already present. Continuing...')
                continue
            elif any(already_existing):
                logger.info(f'{sum(already_existing)} found to already exist')
            # filter out existing ones
            sentences = {k: v for k, v in sentences.items() if k not in out_dict}

        if up_to and counter > up_to:
            logger.info(f'Maximum batch number {up_to} reached. Ending...')
            break
        to_augment = int(f'{augment_size if not ss else ss}')
        logger.debug(f'Batch augment instruction is {to_augment}')
        temp_response = send_instruction(
            augment_size=to_augment,
            sentences=sentences
        )
        responses.append(temp_response)

        for index, choice_element in enumerate(temp_response.choices):
            if choice_element['finish_reason'] != 'stop':
                logger.warning(f"Choice {index} of Completion ID {temp_response.id} finished because of: {choice_element['finish_reason']}")

            try:
                this_response = json.loads(choice_element['message']['content'], strict=False)
            except json.decoder.JSONDecodeError as e:
                # logger.error(choice_element['message']['content'])
                logger.error(e)
                this_response = {}
            for k, v in this_response.items():
                # if augmenting of size 1 the output from api will be a string and not a list.
                if isinstance(v, str):
                    v = [v]
                if overwrite and len(out_dict[k]) > 0:
                    out_dict[k] = v
                else:
                    out_dict[k] = out_dict[k] + v
                if len(out_dict[k]) < augment_size:
                    logger.warning(f'Sentence {k} has {len(out_dict[k])} generated sentences, with {len(v)} from this request.')

        else:
            with open(outfile, 'w' ) as f:
                json.dump(out_dict, f)
            time.sleep(60)
            continue
        break


if __name__ == '__main__':

    pass