import librosa
import numpy as np

SAMPLE_RATE = 22050
N_MFCC = 40

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = np.mean(mfcc.T, axis=0)

    # reshape according to CNN input
    mfcc = mfcc.reshape(1, N_MFCC, 1, 1)

    return mfcc