'''
Extract features from TikTok video
'''

import video_ocr
import whisper
from pathlib import Path
import json
import os
import pickle
from loguru import logger
import jsonlines
import glob
from tqdm import tqdm

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


    def extract_features(self):

        whisper_model = whisper.load_model('base')

        logger.info(f'Savepath: {self.feature_output_path}')
        for video in tqdm(self.valid_videos):
            logger.info(f'Processing {video.id}')
            videoinfosavepath = self.feature_output_path / f'{video.id}.pkl'

            video.ocr_text = video_ocr.perform_video_ocr(str(video.video_location))
            video.whisper_text = whisper_model.transcribe(str(video.video_location))

            with open(videoinfosavepath, 'wb') as f:
                pickle.dump(video, f)


