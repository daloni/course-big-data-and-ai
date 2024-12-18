import librosa
import matplotlib.pyplot as plt
import os

def Audio2Spectrogram(audio_file, dict_cfg):
    audio, Fs = librosa.load(audio_file)
    spectrogram = librosa.feature.melspectrogram(y = audio, sr = Fs,**dict_cfg)
    spectrogram = librosa.power_to_db(spectrogram)
    return librosa.util.normalize(spectrogram)

def main():
    directoryReader = os.walk('../data/ESC-50-master/audio/')
    files = 
    [x[0] for x in os.walk(directory)]
    AUDIO_FILE = "./data/ESC-50-master/audio/1-100210-A-36.wav"
    audio, Fs = librosa.load(AUDIO_FILE)
    fig, ax = plt.subplots(figsize=(14,5))
    # librosa.display.waveshow(audio, sr=Fs, ax=ax, color="blue")

    sample_rate = Fs
    n_mels = 500 # number of mel frequency bands
    hop_length = 100 # number of samples between consecutive analysis frames
    win_length = 200 #(smaller than n_fft) Each frame of audio is windowed by window().
    n_fft = 1024 #length of the FFT window
    fmax = 20000 # the frequency upper bound used to create mel bands

    spectrogram = librosa.feature.melspectrogram(y=audio,
                                                sr=sample_rate,
                                                n_mels=n_mels,
                                                hop_length=hop_length,
                                                win_length = win_length,
                                                n_fft=n_fft,
                                                fmax=fmax)

    spectrogram = librosa.power_to_db(spectrogram)
    librosa.display.specshow(spectrogram)

main()
