import os
from model.speaker_diarization import speaker_diarization
from utils import read_wav

def main(inputs):
    labels = os.listdir(inputs)
    input_files = [inputs + '/' + d for d in labels]
    for i in input_files:
        print(i)
        fs, signal = read_wav(i)
        n, cls = speaker_diarization(fs, signal)
        print(n)


if __name__ == "__main__":
    FILE = '../dataset/test_data'
    main(FILE)