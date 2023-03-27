"""
This module offers functionality to OCR the frames of a video, while trying to be
computationally efficient by ignoring frames that are similar to their adjacent
frames.
"""
import cv2 as cv
import os

import scipy.fft
from contextlib import contextmanager
from itertools import tee
import numpy as np
import tesserocr
import pytesseract
from multiprocessing.pool import ThreadPool
import multiprocessing
import tqdm
from PIL import Image
import click
from functools import wraps


IS_CL = False
FILEPATH_DOC = "Path to the input video file"
SAMPLE_RATE_DOC = "Number of frames to sample per second"
DEBUG_DIR_DOC = (
    "If provided, writes frame and their respective texts here, for debugging"
)


def _only_if_cl(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if IS_CL:
            return f(*args, **kwargs)

    return wrapper


@_only_if_cl
def _error_log(text, *args, **kwargs):
    click.echo(click.style(text, fg="red"), err=True, *args, **kwargs)


@_only_if_cl
def _info_log(text, *args, **kwargs):
    click.echo(text, *args, **kwargs)


class _NoOpProgressBar:
    def update(self):
        pass

    def total(self, n):
        pass


pbar = _NoOpProgressBar()


def phash(image, hash_size=8, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    image = cv.resize(image, (img_size, img_size), interpolation=cv.INTER_LINEAR)
    dct = scipy.fft.dct(scipy.fft.dct(image, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff


class Frame:
    def __init__(self, frame_number, image, ts_second):
        self.frame_number = frame_number
        self.image = image
        self.ts_second = ts_second


@contextmanager
def _open_cv_video(filepath):
    cap = cv.VideoCapture(filepath)
    try:
        yield cap
    finally:
        cap.release()


def _get_frames(video_capture, sample_rate):
    fps = int(video_capture.get(cv.CAP_PROP_FPS))
    pbar.total = (
        video_capture.get(cv.CAP_PROP_FRAME_COUNT) // (fps // sample_rate)
    ) - 1
    frame_number = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_number += 1
        if frame_number % (fps // sample_rate) != 0:
            continue

        # GRAYSCALE
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # THRESHOLDING
        frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        # noise removal
        # frame = cv.medianBlur(frame, 5)

        # frame = cv.normalize(frame, np.zeros((frame.shape[0], frame.shape[1])), 0, 255, cv.NORM_MINMAX)

        # im = Image.fromarray(frame, mode='L')
        # length_x, width_y = im.size
        # factor = min(1, float(1024.0 / length_x))
        # size = int(factor * length_x), int(factor * width_y)
        # im_resized = im.resize(size, Image.ANTIALIAS)
        # frame = np.array(im_resized)


        # kernel = np.ones((5,5),np.uint8)
        # frame = cv.erode(frame, kernel, iterations = 1)

        # Edge detection
        # frame = cv.Canny(frame, 100, 200)

        # gamma=0.6
        # lookUpTable = np.empty((1,256), np.uint8)
        # for i in range(256):
        #     lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        # frame = cv.LUT(frame, lookUpTable)

        # kernel = np.ones((5,5),np.uint8)
        # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)


        yield Frame(frame_number, frame, frame_number // fps)


def _pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _are_similar_frame(f1, f2):
    diff = np.count_nonzero(phash(f1.image) != phash(f2.image))
    return diff <= 15


def _filter_redundant_frames(frames):
    for f1, f2 in _pairwise(frames):
        if not _are_similar_frame(f1, f2):
            yield f1
        else:
            pbar.update()


def _ocr(frame):
    """
    Page segmentation modes:
    0    Orientation and script detection (OSD) only.
    1    Automatic page segmentation with OSD.
    2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
    3    Fully automatic page segmentation, but no OSD. (Default)
    4    Assume a single column of text of variable sizes.
    5    Assume a single uniform block of vertically aligned text.
    6    Assume a single uniform block of text.
    7    Treat the image as a single text line.
    8    Treat the image as a single word.
    9    Treat the image as a single word in a circle.
    10    Treat the image as a single character.
    11    Sparse text. Find as much text as possible in no particular order.
    12    Sparse text with OSD.
    13    Raw line. Treat the image as a single text line,
        bypassing hacks that are Tesseract-specific.
    
    OCR Engine modes:
    0    Legacy engine only.
    1    Neural nets LSTM engine only.
    2    Legacy + LSTM engines.
    3    Default, based on what is available.
 
    """
    pil_image = Image.fromarray(frame.image)
    # text = tesserocr.image_to_text(pil_image)
    custom_config = r'--oem 3 --psm 11'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    frame.text = text
    pbar.update()
    return frame


def _parallel_ocr(frames):
    with ThreadPool(multiprocessing.cpu_count()) as pool:
        return pool.map(_ocr, frames, chunksize=multiprocessing.cpu_count())


def _write_if_debug(frames, debug_dir):
    if not debug_dir:
        return
    for frame in frames:
        cv.imwrite(os.path.join(debug_dir, f"{frame.frame_number}.png"), frame.image)
        with open(os.path.join(debug_dir, f"{frame.frame_number}.txt"), "w") as f:
            f.write(frame.text)


def perform_video_ocr(filepath: str, sample_rate: int = 1, debug_dir: str = ""):
    f"""
    :param filepath: {FILEPATH_DOC}
    :param sample_rate: {SAMPLE_RATE_DOC}
    :param debug_dir: {DEBUG_DIR_DOC}
    """
    frames = []
    with _open_cv_video(filepath) as cap:
        frames = _parallel_ocr(_filter_redundant_frames(_get_frames(cap, sample_rate)))
    frames.sort(key=lambda frame: frame.frame_number)
    non_empty_frames = []
    for frame in frames:
        if frame.text.strip():
            non_empty_frames.append(frame)
    _write_if_debug(non_empty_frames, debug_dir)
    return non_empty_frames


def _get_time_stamp(seconds):
    rem_seconds = seconds
    hours = rem_seconds // 3600
    rem_seconds %= 3600
    mins = rem_seconds // 60
    rem_seconds %= 60
    return "{:02}:{:02}:{:02}".format(int(hours), int(mins), int(rem_seconds))


def _display_frames(frames):
    terminal_width = os.get_terminal_size().columns
    _info_log("")
    for frame in frames:
        _info_log("-" * terminal_width)
        _info_log(f"Timestamp = {_get_time_stamp(frame.ts_second)}")
        _info_log(frame.text)
    _info_log("-" * terminal_width)


@click.command()
@click.argument(
    "filepath",
    type=click.Path(
        exists=True,
        readable=True,
    ),
)
@click.option("--sample_rate", type=int, help=SAMPLE_RATE_DOC, default=1)
@click.option(
    "--debug_dir",
    type=click.Path(exists=True, writable=True, file_okay=False, dir_okay=True),
    help=DEBUG_DIR_DOC,
)
def main(filepath, sample_rate, debug_dir):
    global IS_CL
    global pbar
    IS_CL = True
    with tqdm.tqdm() as progress_bar:
        pbar = progress_bar
        frames = perform_video_ocr(
            filepath, sample_rate=sample_rate, debug_dir=debug_dir
        )
    _display_frames(frames)


if IS_CL:
    main()
