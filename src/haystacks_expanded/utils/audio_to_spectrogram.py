"""
From https://github.com/roudimit/AVLnet/blob/main/audio_to_spectrograms.py

as part of feature extraction for https://github.com/ninatu/everything_at_once

N.B. Instructions for feature extraction:

Use the model on your own videos

* If you want to use smaller clips within your videos, you'll have to split them up at some point. You can do that either before you extract the features, or after. We recommend the former.
* You can use the functions in audio_to_spectrograms.py to extract audio, and then spectrograms, from videos.
* You can use this video feature extractor to extract the visual features. We used the default settings. The model weights for the ResNext-101 are provided in the AVLnet model weights folder.
* We recommend extracting all of the features first, creating a pickle file to match the one for YouCook2, and then modifying the fine-tuning commands for YouCook2.
* Specifically, the pickle file should be a Python list, where each element is a dictionary and represents a video clip. The keys of each dictionary should be '2d' which maps to a single feature vector representing the pooled 2d features for the entire clips (same for the '3d' key), and 'audio' which maps to the spectrogram for the clip. Please review youcook_train_audio.pkl and youcook_val_audio.pkl for more details.

"""

import subprocess

import librosa
import numpy as np
import scipy


def extract_audio(input_file, output_file, log_file):
    """ Step 1
        Extracts audio at the native sampling rate into a separate wav file. 
    """
    subprocess.call(['ffmpeg', '-i', input_file, '-vn', output_file], stdout = log_file, stderr = log_file)

def stereo_to_mono_downsample(input_file, output_file, sample_rate=16000):
    """ Step 2
        Resamples wav file (we use 16 kHz).
        Convert from stereo to mono.
        Apply a gain of -4 to avoid clipping for mono to stereo conversion.
    """
    subprocess.call(['sox', input_file, output_file, 'gain', '-4', 'channels', '1', 'rate', str(sample_rate)])

def LoadAudio(path, target_length=2048, use_raw_length=False):
    """ Step 3
        Convert audio wav file to mel spec feats
        target_length is the maximum number of frames stored (disable with use_raw_length)
        # NOTE: assumes audio in 16 kHz wav file
    """
    audio_type = 'melspectrogram'
    preemph_coef = 0.97
    sample_rate = 16000
    window_size = 0.025
    window_stride = 0.01
    window_type = 'hamming'
    num_mel_bins = 40
    padval = 0
    fmin = 20
    n_fft = int(sample_rate * window_size)
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)
    windows = {'hamming': scipy.signal.hamming}
    # load audio, subtract DC, preemphasis
    # NOTE: sr=None to avoid resampling (assuming audio already at 16 kHz sr
    y, sr = librosa.load(path, sr=None)
    if y.size == 0:
        y = np.zeros(200)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length,
        window=windows[window_type])
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        feats = librosa.power_to_db(melspec, ref=np.max)
    n_frames = feats.shape[1]
    if use_raw_length:
        target_length = n_frames
    p = target_length - n_frames
    if p > 0:
        feats = np.pad(feats, ((0,0),(0,p)), 'constant',
            constant_values=(padval,padval))
    elif p < 0:
        feats = feats[:,0:p]
        n_frames = target_length
    return feats, n_frames

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])