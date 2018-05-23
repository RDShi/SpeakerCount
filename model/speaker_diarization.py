"""
unsupervised: process： feature extraction --> kmeans(silhouette)
"""
import os
import numpy
from scipy.spatial import distance
import scipy.signal
from utils import normalize_features, vad, mt_feature_extraction, ivad, write_wav
from model.unsupervised_model import kmeans_silhouette
EPS = numpy.finfo(float).eps


def speaker_diarization(fs, signal, mt_size=2.0, mt_step=0.2, st_win=0.05):
    """
    unsupervised speaker count
    """
    st_step = st_win

    [mid_term_features, short_term_features] = mt_feature_extraction(signal, fs, mt_size * fs,
                                                                     mt_step * fs,
                                                                     round(fs * st_win))
    [mid_term_features_norm, _, _] = normalize_features([mid_term_features.T])
    mid_term_features_norm = mid_term_features_norm[0].T
    num_of_windows = mid_term_features.shape[1]

    # VAD：
    reserved_time = 1
    segment_limits = vad(short_term_features, st_step, smooth_window=0.5, weight=0.3)
    i_vad = ivad(segment_limits, mt_step, reserved_time, num_of_windows)
    mid_term_features_norm = mid_term_features_norm[:, i_vad]

    # remove outliers:
    distances_all = numpy.sum(distance.squareform(distance.pdist(mid_term_features_norm.T)), axis=0)
    m_distances_all = numpy.mean(distances_all)
    i_non_outliers = numpy.nonzero(distances_all < 1.2 * m_distances_all)[0]

    mid_term_features_norm = mid_term_features_norm[:, i_non_outliers]
    i_features_select = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45,
                         46, 47, 48, 49, 50, 51, 52, 53]
    mid_term_features_norm = mid_term_features_norm[i_features_select, :]

    num_range = range(2, 10)##人数范围[2,10)
    [n_speakers_final, imax, num_speaker_cls] = \
        kmeans_silhouette(mid_term_features_norm, num_range)

    cls = numpy.zeros((num_of_windows,))-1
    valid_pos = i_vad[i_non_outliers]
    for i in range(num_of_windows):
        if i in valid_pos:
            j = numpy.argwhere(valid_pos == i)[0][0]
            cls[i] = num_speaker_cls[imax][j]

    # median filtering:
    cls = scipy.signal.medfilt(cls, 11)
    start = 0
    end = 0
    for i in range(1, len(cls)):
        if cls[i] == cls[i-1]:
            end = i
        else:
            write_wav(os.path.join(os.path.pardir, "result", "result_wav",
                                   str(cls[i-1]) + "-" + str(start*mt_step) + "-" +
                                   str(end*mt_step) + ".wav"),
                      fs, signal[int(start * mt_step * fs):int(end * mt_step * fs)])
            start = i
    return n_speakers_final, cls
