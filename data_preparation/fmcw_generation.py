'''
Generate FMCW signals and write to .wav file
4/29/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import wave
import argparse
import numpy as np
import matplotlib.pyplot as plt
from filters import butter_bandpass_filter

def fmcw_generation(fc, B, fs, L, save_name, blank_l=0, blank_r=0):
    assert(fc + B < fs / 2)     # Nyquist freq
    t = np.arange(L) / fs
    u = 2 * np.pi * (fc * t + B * (t ** 2) / (2 * L / fs))
    v = np.cos(u)

    if blank_l > 0:
        v = np.concatenate([np.zeros((blank_l)), v])
        L += blank_l
    if blank_r > 0:
        v = np.concatenate([v, np.zeros((blank_r))])
        L += blank_r

    v = np.tile(v, 1001)
    v = butter_bandpass_filter(v, fc - B * 0.0, fc + B + B * 0.0, fs, order=6)
    print(v)

    fmcw_seq = v[500 * L: 501 * L]
    fmcw_seq /= np.max(np.abs(fmcw_seq))

    fmcw_seq *= (1 << 15) - 1
    fmcw_seq = fmcw_seq.astype(np.int16)

    plt.figure()
    plt.plot(fmcw_seq)
    plt.show()

    wavfile = wave.open(os.path.join('tx_signals', save_name), 'wb')
    wavfile.setframerate(44100)
    wavfile.setsampwidth(2)
    wavfile.setnchannels(1)
    wavfile.writeframes(fmcw_seq.tobytes())
    wavfile.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-fc', '--carrier_freq', help='carrier_freq', type=float, default=19000)
    parser.add_argument('-B', '--bandwidth', help='bandwidth', type=float)
    parser.add_argument('-fs', '--samplingrate', help='samplingrate', type=float, default=50000)
    parser.add_argument('-L', '--frame_length', help='frame_length', type=int, default=600)
    parser.add_argument('-o', '--output', help='intended file name')
    parser.add_argument('--blank_l', help='number of blank samples to be added on the left', type=int, default=0)
    parser.add_argument('--blank_r', help='number of blank samples to be added on the right', type=int, default=0)

    args = parser.parse_args()
    fmcw_generation(args.carrier_freq, args.bandwidth, args.samplingrate, args.frame_length, args.output, args.blank_l, args.blank_r)
