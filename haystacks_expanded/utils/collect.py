from TikTokApi import TikTokApi
import logging
import json


logging.basicConfig(level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# logging.getLogger('').addHandler(console)


def get_cookies_from_file():
    with open('cookies.json') as f:
        cookies = json.load(f)

    cookies_kv = {}
    for cookie in cookies:
        cookies_kv[cookie['name']] = cookie['value']

    return cookies_kv

cookies = get_cookies_from_file()
# print(cookies)

def get_cookies(**kwargs):
    return cookies

logging.info('Collecting...')
with TikTokApi(logging_level=logging.INFO) as api: # .get_instance no longer exists
    api._get_cookies = get_cookies
#     # print(api._get_cookies())
#     logging.info('Start collection')
#     for video in api.search.videos('therock'):
#         print(video.id)
#     # for trending_video in api.trending.videos(count=50):
#         # print(trending_video.author.username)
    video_bytes = api.video(id='7194841883313491242').bytes()

    # Saving The Video
    with open('saved_video.mp4', 'wb') as output:
        output.write(video_bytes)