"""
有监督识别的模型
"""
import pickle
from collections import defaultdict
import time
from utils.utils import GMMSet
from utils.feature_extraction import get_feature


class ModelInterface:
    """
    Supervised Interface Class：
    enroll: 注册新ID
    train: 训练模型
    dump: 模型序列化
    predict: 输出这段声音是谁说的
    load: 读取模型
    """
    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()

    def enroll(self, name, fs, signal):
        """
        注册新ID
        :param name:ID
        :param fs:采样率
        :param signal:信号
        """
        feat = get_feature(fs, signal)
        self.features[name].extend(feat)

    def train(self):
        """
        训练模型
        """
        self.gmmset = GMMSet()
        start_time = time.time()
        for name, feats in self.features.items():
            self.gmmset.fit_new(feats, name)
        print(time.time() - start_time, " seconds")

    def dump(self, fname):
        """
        保存模型
        :param fname: 保存的文件名
        """
        with open(fname, 'wb') as fid:
            pickle.dump(self, fid, -1)

    def predict(self, fs, signal):
        """
        判断说话人ID
        :param fs: 采样率
        :param signal: 信号
        :return:
        """
        feat = get_feature(fs, signal)
        return self.gmmset.predict_one(feat)

    @staticmethod
    def load(fname):
        """
        读取模型
        :param fname: 文件名
        """
        with open(fname, 'rb') as fid:
            model_class = pickle.load(fid)
            return model_class
