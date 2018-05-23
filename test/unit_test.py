"""
unit test
"""
import unittest
import os
from model.speaker_diarization import speaker_diarization
from model.speaker_recognition import task_enroll, task_predict
from utils import read_wav


class TestFunc(unittest.TestCase):
    """Test mathfuc.py"""

    def test_speaker_diarization(self):
        """Test method speaker_diarization"""
        file_name = os.path.join(os.getcwd(), "data", "diarization", '3.wav')
        [fs, signal] = read_wav(file_name)
        self.assertEqual(int(os.path.basename(file_name)[:-4]), speaker_diarization(fs, signal))

    def test_speaker_recognition(self):
        """Test function speaker_recognition"""
        model_file = os.path.join(os.getcwd(), "data", "recognition", "model.out")
        inputs = os.path.join(os.getcwd(), "data", "recognition", "training_data")
        labels = os.listdir(inputs)
        input_dirs = [os.path.join(inputs, d) for d in labels]
        input_dirs = " ".join(input_dirs)
        task_enroll(input_dirs, model_file)

        inputs = os.path.join(os.getcwd(), "data", "recognition", "test_data")
        labels = os.listdir(inputs)
        input_files = [os.path.join(inputs, d) for d in labels]
        self.assertIsNone(task_predict(input_files, model_file))


if __name__ == '__main__':
    unittest.main()
