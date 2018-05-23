"""
demo
"""
import os
from model.speaker_recognition import task_enroll, task_predict


def main(mod_type, mdl_file):
    """
    main function
    """
    if mod_type == 'enroll':
        inputs = '../dataset/training_data'
        labels = os.listdir(inputs)
        input_dirs = [inputs + '/' + d for d in labels]
        input_dirs = " ".join(input_dirs)
        task_enroll(input_dirs, mdl_file)

    elif mod_type == 'predict':
        inputs = '../dataset/test_data'
        labels = os.listdir(inputs)
        input_files = [inputs + '/' + d for d in labels]
        task_predict(input_files, mdl_file)


if __name__ == "__main__":
    # MODE_TYPE = "enroll"
    MODE_TYPE = "predict"

    MODEL_FILE = '../dataset/model.out'
    main(MODE_TYPE, MODEL_FILE)
