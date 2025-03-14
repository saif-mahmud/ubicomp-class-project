# MFCC Spectrogram conversion code from VGGish, Google Inc.
# https://github.com/tensorflow/models/tree/master/research/audioset

SAMPLE_RATE = 16000

STFT_WINDOW_LENGTH_SECONDS = 0.6
STFT_HOP_LENGTH_SECONDS = 0.03
# Each example contains 96 10ms frames
EXAMPLE_WINDOW_SECONDS = 100 * STFT_HOP_LENGTH_SECONDS
EXAMPLE_HOP_SECONDS = 100 * STFT_HOP_LENGTH_SECONDS     # with zero overlap.
# Frames in input mel-spectrogram patch.
NUM_FRAMES = 1 + int(EXAMPLE_WINDOW_SECONDS / STFT_HOP_LENGTH_SECONDS)

MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.001  # Offset used for stabilized log of input mel-spectrogram.

NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
NUM_MEL_BINS = NUM_BANDS
