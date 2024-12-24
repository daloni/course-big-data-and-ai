import librosa
import matplotlib.pyplot as plt
import os
import shutil
from target_relation import TARGET

DEBUG = False
SPECTOGRAMS_PATH = '../data/spectograms/'
RANDOMIZED_SPECTOGRAMS_PATH = '../data/spectograms-train/'

def main(path=SPECTOGRAMS_PATH, output=RANDOMIZED_SPECTOGRAMS_PATH):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if DEBUG:
        files = files[:1]

    # Delete the output folder if it exists
    if os.path.exists(output):
        shutil.rmtree(output)

    # Create the output folder
    os.makedirs(output)

    for file in files:
        # Filename {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.png extract the target
        myTarget = file.split('-')[-1].replace('.png', '')
        myTarget = TARGET[int(myTarget)]

        # Create folder inside the output folder
        output_folder = os.path.join(output, myTarget)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Count the number of files in the output folder
        output_files = len(os.listdir(output_folder))

        # Copy the file to the output folder with a new name
        output_file = os.path.join(output_folder, 'file-' + str(output_files) + '.png')
        shutil.copy(os.path.join(path, file), output_file)

main()
