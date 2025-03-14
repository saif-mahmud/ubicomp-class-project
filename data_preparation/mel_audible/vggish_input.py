# MFCC Spectrogram conversion code from VGGish, Google Inc.
# https://github.com/tensorflow/models/tree/master/research/audioset

import numpy as np
from scipy.io import wavfile
from .mel_features import *
from .params import *


def wavfile_to_examples(wav_file, lower_edge_hertz=MEL_MIN_HZ,
                        upper_edge_hertz=MEL_MAX_HZ):
    sr, wav_data = wavfile.read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    data = wav_data / 32768.0    # Convert to [-1.0, +1.0]
    # print("sampling rate:", sr, "length", wav_data.shape)

    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Compute log mel spectrogram features.
    log_mel = log_mel_spectrogram(data,
                                  audio_sample_rate=sr,
                                  log_offset=LOG_OFFSET,
                                  window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
                                  hop_length_secs=STFT_HOP_LENGTH_SECONDS,
                                  num_mel_bins=NUM_MEL_BINS,
                                  lower_edge_hertz=lower_edge_hertz,
                                  upper_edge_hertz=upper_edge_hertz)

    # print(log_mel.shape)
    # Frame features into examples.
    features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
    example_window_length = int(
        round(EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(
        round(EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = frame(
        log_mel, window_length=example_window_length, hop_length=example_hop_length)
    return log_mel_examples


def wavform_to_examples(wav_data, lower_edge_hertz=MEL_MIN_HZ, upper_edge_hertz=MEL_MAX_HZ, sr=16000):
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    data = wav_data / 32768.0    # Convert to [-1.0, +1.0]
    print("[DEBUG] sampling rate:", sr, "length", wav_data.shape)

    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Compute log mel spectrogram features.
    log_mel = log_mel_spectrogram(data,
                                  audio_sample_rate=sr,
                                  log_offset=LOG_OFFSET,
                                  window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
                                  hop_length_secs=STFT_HOP_LENGTH_SECONDS,
                                  num_mel_bins=NUM_MEL_BINS,
                                  lower_edge_hertz=lower_edge_hertz,
                                  upper_edge_hertz=upper_edge_hertz)

    print('[DEBUG] log mel feature shape:', log_mel.shape)
    # Frame features into examples.
    features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
    print(f'[DEBUG] Feature Sampling Rate: {features_sample_rate}')
    example_window_length = int(
        round(EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(
        round(EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = frame(
        log_mel, window_length=example_window_length, hop_length=example_hop_length)

    print(
        f'[DEBUG] window len: {example_window_length}, hop len: {example_hop_length}')
    print(f'[DEBUG] windowed mel feature shape: {log_mel_examples.shape}')
    return log_mel_examples, log_mel
