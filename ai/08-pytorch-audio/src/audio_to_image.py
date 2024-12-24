import librosa
import matplotlib.pyplot as plt
import os
import numpy as np

DEBUG = False
AUDIOS_PATH = '../data/ESC-50-master/audio/'
SPECTOGRAMS_PATH = '../data/spectograms/'

def Audio2Spectrogram(audio_file, dict_cfg):
    audio, Fs = librosa.load(audio_file)

    # Extract fmax from the dictionary and remove it from the dictionary
    fmax = dict_cfg.get('fmax', Fs/2)
    del dict_cfg['fmax']
    fmax = min(fmax, Fs // 2)

    spectrogram = librosa.feature.melspectrogram(y=audio, sr=Fs, **dict_cfg, fmax=fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    return librosa.util.normalize(spectrogram)

def main(path=AUDIOS_PATH, output_path=SPECTOGRAMS_PATH):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if DEBUG:
        files = files[:1]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in files:
        audio_cfg = dict(
            n_mels=256,         # Higher number of Mel bands for higher spectral resolution
            hop_length=50,      # Lower hop_length for higher temporal resolution
            win_length=512,     # Larger window size for higher temporal resolution
            n_fft=4096,         # Larger FFT for higher spectral resolution
            fmax=22000          # Keep fmax the same for the same maximum frequency
        )

        filepath = os.path.join(path, file)
        spec = Audio2Spectrogram(filepath, audio_cfg)

        # Create figure and axis explicitly with high resolution
        fig, ax = plt.subplots(figsize=(14, 6))  # Larger figure for better quality

        # Display the spectrogram with a better colormap (e.g., 'inferno')
        img = librosa.display.specshow(spec, sr=22050, hop_length=50, x_axis='time', y_axis='mel', ax=ax, cmap='inferno')

        if DEBUG:
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            plt.show()

        # Remove axes for cleaner image
        ax.axis('off')

        # Save the spectrogram as PNG with high DPI and without axes
        file_name = file.replace('.wav', '.png')
        output_file_path = os.path.join(output_path, file_name)
        
        # Save the image with 300 DPI for better quality
        plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')

        if DEBUG:
            continue

        plt.close()

main()
