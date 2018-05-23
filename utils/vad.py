"""
voice action detection
"""
import numpy
import sklearn.svm
from utils.utils import normalize_features
EPS = numpy.finfo(float).eps


def vad(short_term_features, st_step=0.05, smooth_window=0.5, weight=0.5):
    """
    输入一段音频，输出segment list一段段有声音的区间
    :param short_term_features:
    :param st_step:
    :param smooth_window: 平滑窗口
    :param weight: 有声音无声音半监督的权重
    :return: segment
    """
    if weight >= 1:
        weight = 0.99
    if weight <= 0:
        weight = 0.01

    energy_st = short_term_features[1, :]
    energy_tmp = numpy.sort(energy_st)
    len_one = int(len(energy_tmp) / 10)
    len_two = int(len(energy_tmp) / 10)
    thre_one = numpy.mean(energy_tmp[0:len_one]) + EPS
    thre_two = numpy.mean(energy_tmp[-len_two:-1]) + EPS
    class_one = short_term_features[:, numpy.where(energy_st <= thre_one)[0]]
    class_two = short_term_features[:, numpy.where(energy_st >= thre_two)[0]]
    features_sample = [class_one.T, class_two.T]

    [features_norm_sample, mean_sample, std_sample] = normalize_features(features_sample)
    tred_svm = train_svm(features_norm_sample, 1.0)

    prob_onset = []
    for i in range(short_term_features.shape[1]):
        cur_fv = (short_term_features[:, i] - mean_sample) / std_sample
        prob_onset.append(tred_svm.predict_proba(cur_fv.reshape(1, -1))[0][1])
    prob_onset = numpy.array(prob_onset)
    prob_onset = smooth_moving_avg(prob_onset, int(smooth_window / st_step))  # smooth probability

    prob_onset_sorted = numpy.sort(prob_onset)
    num_tmp = int(prob_onset_sorted.shape[0] / 10)
    thre_tmp = (numpy.mean((1 - weight) * prob_onset_sorted[0:num_tmp]) +
                weight * numpy.mean(prob_onset_sorted[-num_tmp::])) #阈值

    max_idx = numpy.where(prob_onset > thre_tmp)[0]
    i = 0
    time_clusters = []
    segment_limits = []
    combine_thd = 2

    while i < len(max_idx):
        cur_cluster = [max_idx[i]]
        if i == len(max_idx)-1:
            break
        while max_idx[i+1] - cur_cluster[-1] <= combine_thd: # 连着的话合并
            cur_cluster.append(max_idx[i+1])
            i += 1
            if i == len(max_idx)-1:
                break
        i += 1
        time_clusters.append(cur_cluster)
        segment_limits.append([cur_cluster[0] * st_step, cur_cluster[-1] * st_step])

    min_duration = 0.1
    segment_limits_two = []
    for seg_tmp in segment_limits:
        if seg_tmp[1] - seg_tmp[0] > min_duration:
            segment_limits_two.append(seg_tmp)
    segment_limits = segment_limits_two

    return segment_limits


def train_svm(features, cparam):
    """
    vad辅助函数，生成一个SVM判断有无声音
    """
    data_tmp = numpy.array([])
    label_tmp = numpy.array([])
    for i, feature_tmp in enumerate(features):
        if i == 0:
            data_tmp = feature_tmp
            label_tmp = i * numpy.ones((len(feature_tmp), 1))
        else:
            data_tmp = numpy.vstack((data_tmp, feature_tmp))
            label_tmp = numpy.append(label_tmp, i * numpy.ones((len(feature_tmp), 1)))
    svm = sklearn.svm.SVC(C=cparam, kernel='linear', probability=True)
    svm.fit(data_tmp, label_tmp)
    return svm


def smooth_moving_avg(input_signal, window_len=11):
    """
    平滑滤波
    """
    window_len = int(window_len)
    if input_signal.ndim != 1:
        raise ValueError("")
    if input_signal.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return input_signal
    sig_tmp = numpy.r_[2*input_signal[0] - input_signal[window_len-1::-1],
                       input_signal, 2*input_signal[-1]-input_signal[-1:-window_len:-1]]
    window_tmp = numpy.ones(window_len, 'd')
    result = numpy.convolve(window_tmp/window_tmp.sum(), sig_tmp, mode='same') # 卷积，相当于平均
    return result[window_len:-window_len+1]


def ivad(segment_limits, mt_step, reserved_time, num_of_windows):
    """
    segment_limits --> list pos
    """
    count = 0
    i_vad = numpy.array([])
    before_seg = numpy.array([])
    for seg in segment_limits:
        if count == 0:
            i_vad = numpy.array(range(max(1, int(seg[0] / mt_step - reserved_time / mt_step)),
                                      min(int(seg[1] / mt_step + reserved_time / mt_step),
                                          num_of_windows - 1)))
            count += 1
            before_seg = min(int(seg[1] / mt_step + reserved_time / mt_step), num_of_windows - 1)
        else:
            i_vad = numpy.append(i_vad, numpy.array(
                range(max(before_seg, int(seg[0] / mt_step - reserved_time / mt_step)),
                      min(int(seg[1] / mt_step + reserved_time / mt_step), num_of_windows - 1))))
            before_seg = min(int(seg[1] / mt_step + reserved_time / mt_step), num_of_windows - 1)
    return i_vad
