from TikTokAPI import TikTokAPI
cookie = {
  "s_v_web_id": "verify_ld1w40f8_wEqFSceA_rSaW_4rnP_9kFx_WHvS1El6aLec",
  "tt_webid": "7CnNyEIvr9f61pz_sUqC6Rc_zSiP3za1UfFm_27NEnQwc"
}
api = TikTokAPI(cookie=cookie)
retval = api.getTrending(count=5)
# print(retval)


savepath = '/home/hubert/DPhil_Studies/2023-01-DSF/haystacks_expanded/test_video.mp4'
api.downloadVideoById('7194841883313491242',savepath)

