import requests
import json
import os
import datetime
import glob
from loguru import logger
import yt_dlp
from dataclasses import dataclass, field


@dataclass
class video_info:
    id: str
    username: field(default_factory=str, compare=False)

    def as_url(self):
        return f'https://www.tiktok.com/@{self.username}/video/{self.id}'

def query(query_name, query, savepath, token = None, period = 1, platform='TikTok')->None:
    '''Function to collect data from API given a query.
    '''
    assert token is not None, 'API token cannot be none'
    assert os.path.exists(savepath), f'{savepath} is not a valid path'

    platform = platform.lower()
    if platform not in ['tiktok', 'instagram']:
        raise ValueError(f"platform argument must be either tiktok or instagram. Received {platform} instead")

    root = "https://www.ensembledata.com/apis"
    if platform == 'tiktok':
        endpoint = "/tt/keyword/full-search"
    # elif platform == 'instagram':
        # endpoint = "/tt/"

    params = {
        'name': query,
        'period': period,
        'country': 'uk',
        'token': token
    }

    res = requests.get(root+endpoint, params=params)
    if res.status_code != 200:
        raise RuntimeWarning(f'Status code was not 200. Received {res.status_code}')
    else:
        logger.info('Collection OK. Status code 200')


    savename = os.path.join(savepath, f'{query_name}_{datetime.datetime.today().strftime("%Y-%m-%d")}.json')

    with open(savename, 'w') as f:
        json.dump(res.json(), f)
    logger.info(f'Saved to {savename}')

def download(file, savepath, overwrite = False, max_download = None):

    assert os.path.exists(savepath)
    if not os.path.exists('videos'):
        os.makedirs('videos')

    if max_download:
        max_download = int(max_download)

    all_ids = []
    with open(file, 'r') as f:
        x = json.load(f)
        for row in x['data']:
            all_ids.append(video_info(
                id = row['aweme_info']['aweme_id'],
                username= row['aweme_info']['author']['unique_id']
            ))

    existing_videos = glob.glob(os.path.join(savepath, '*.mp4'))
    existing_videos = [os.path.split(i)[-1].split('.')[0] for i in existing_videos]

    ydl_opts= {
        'outtmpl': os.path.join(savepath, f"videos/%(id)s.%(ext)s"),
        'overwrites': overwrite,
        'logger': logger
    }
    skipped = 0
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for counter, video in enumerate(all_ids):
            if video.id in existing_videos:
                logger.info(f'{video.id} already downloaded. Continuing...')
                skipped += 1
            logger.info(f"{counter}, {skipped}, {max_download}")
            if max_download is not None and counter-skipped >= max_download:
                logger.info(f'Max download of {max_download} reached. Terminating...')
                break
            try:
                ydl.download(video.as_url())
            except:
                logger.error('Something went wrong')
                skipped += 1
