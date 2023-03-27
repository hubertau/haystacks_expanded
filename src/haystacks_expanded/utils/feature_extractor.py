'''
Extract features from TikTok video
'''

import whisper
from pathlib import Path
import json
import os
import pickle
from loguru import logger
import jsonlines
import pandas as pd
import numpy as np
import glob
import video_ocr
from tqdm import tqdm
from dataclasses import asdict
from nltk import sent_tokenize
from transformers import pipeline

from . import audio_to_spectrogram
from .search import video_info


class HaystacksFeatureExtractor:

    def __init__(self, metadata, video_path, feature_output_path):
        self.metadata = Path(metadata)
        self.video_path = Path(video_path)
        self.feature_output_path = Path(feature_output_path)

        self._consolidate_data()

    def _metadata_from_one_json(self, json_file):

        output = []
        with open(json_file, 'r') as f:
            x = json.load(f)
            for row in x['data']:
                video_location = self.video_path.absolute() / f"{row['aweme_info']['aweme_id']}.mp4"
                output.append(video_info(
                    id = row['aweme_info']['aweme_id'],
                    username= row['aweme_info']['author']['unique_id'],
                    downloaded = os.path.isfile(video_location),
                    video_location = video_location,
                    comment_count= row['aweme_info']['statistics']['comment_count'],
                    digg_count= row['aweme_info']['statistics']['digg_count'],
                    play_count= row['aweme_info']['statistics']['play_count'],
                    share_count= row['aweme_info']['statistics']['share_count'],
                    whatsapp_share_count= row['aweme_info']['statistics']['whatsapp_share_count'],
                    description=row['aweme_info']['search_desc']
                ))
        return output

    def _consolidate_data(self):

        self.all_videos = []
        if os.path.isdir(self.metadata):
            # if just the metadata dir is specified, then scan through all query files
            existing_query_outputs = glob.glob(os.path.join(self.metadata, '*.json'))
            self.all_videos = []
            for json_file in existing_query_outputs:
                self.all_videos.extend(self._metadata_from_one_json(json_file))
            # existing_videos = [Path(f).stem for f in existing_videos]
            # existing_features = glob.glob(os.path.join(self.feature_output_path, '*.pkl'))
            # existing_features = [Path(f).stem for f in existing_features]

        else:
            self.all_videos = self._metadata_from_one_json(self.metadata)
            self.valid_videos = [video for video in self.all_videos if video.downloaded]

        self.valid_videos = [video for video in self.all_videos if video.downloaded]


    def _preprocess_audio(self):

        for video in self.valid_videos:

            audio_output_file = video.video_location.parent / f'{video.id}.wav'
            spec_output       = video.video_location.parent / f'{video.id}.npz'

            audio_to_spectrogram.extract_audio(video.video_location, audio_output_file)
            audio_to_spectrogram.stereo_to_mono_downsample(audio_output_file, audio_output_file)
            audio_to_spectrogram.LoadAudio()


    def extract_features(self, overwrite = False):

        corrector = pipeline("text2text-generation", model='oliverguhr/spelling-correction-english-base')

        whisper_model = whisper.load_model('small')

        if overwrite:
            self.videos_to_process = self.valid_videos
        else:
            self.existing_features = glob.glob(os.path.join(self.feature_output_path, '*.pkl'))
            self.existing_features = [Path(i).stem for i in self.existing_features]

            self.videos_to_process = [i for i in self.valid_videos if i.id not in self.existing_features]

        logger.info(f'Savepath: {self.feature_output_path}')
        for video in tqdm(self.videos_to_process):
            # logger.info(f'Processing {video.id}')
            videoinfosavepath = self.feature_output_path / f'{video.id}.pkl'

            ocr_temp = video_ocr.perform_video_ocr(
                str(video.video_location)
            )
            video.ocr_text = '. '.join([corrector(frame.text.strip().replace('\n', ''), max_length=2048)[0]['generated_text'] for frame in ocr_temp])
            video.whisper_text = whisper_model.transcribe(str(video.video_location), fp16=False).get('text', '').strip()

            to_combine = [i for i in [video.description, video.ocr_text, video.whisper_text] if len(i)>0]
            video.concatenated = '. '.join(to_combine)

            with open(videoinfosavepath, 'wb') as f:
                pickle.dump(video, f)

    def save_to(self, savename, overwrite=False):
        """Save all features files to directory
        """

        if os.path.isfile(savename) and not overwrite:
            existing_df = pd.read_csv(savename)
            logger.info('Existing save file found, loading in...')

        # collect all feature files from feature output path
        all_features = glob.glob(os.path.join(self.feature_output_path, '*.pkl'))

        # extract ids from these items
        all_ids = [Path(item).stem for item in all_features]

        # remove ids that already exist in the existing_df
        if os.path.isfile(savename) and not overwrite:
            all_ids = [i for i in all_ids if i in existing_df['id'].unique()]

        # rehydrate load paths for unpickling
        all_ids = [Path(self.feature_output_path / f'{id}.pkl') for id in all_ids]

        # iterate over features and add them to records
        to_add = []
        for video_feature in all_ids:
            with open(video_feature, 'rb') as f:
                features = pickle.load(f)
            to_add.append(asdict(features))

        # create dataframe from records
        df = pd.DataFrame.from_records(to_add)

        # do configs
        # df['concatenated'] = df.apply(lambda row: '. '.join([row['description'], row['ocr']['text'].strip(), row['whisper_text']]))

        # IMPORTANT: The subsequent steps in the pipeline will take the concatenated text and split into sentences in each row.
        s = df["concatenated"].apply(lambda x : sent_tokenize(x)).apply(pd.Series,1).stack()
        s.index = s.index.droplevel(-1)
        s.name = 'sentence'

        # There are blank or emplty cell values after above process. Removing them
        s.replace('', np.nan, inplace=True)
        s.dropna(inplace=True)

        df = df.join(s)

        # append to any existing dataframe
        if os.path.isfile(savename) and not overwrite:
            df = pd.concat([existing_df, df])

        # save
        df.to_csv(savename)

