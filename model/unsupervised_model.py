"""
unsupervised function
"""
import numpy
from scipy.spatial import distance
import sklearn.cluster
import sklearn.discriminant_analysis
import sklearn


def kmeans_silhouette(mid_term_features_norm, num_range):
    """
    利用kmeans和轮廓系数
    """
    s_range = list(num_range)
    num_speaker = [0 for _ in num_range]
    num_speaker_cls = [[] for _ in num_range]
    for _ in range(5):  ##重复5次
        cls_all = []
        sil_all = []
        centers_all = []

        for i_speakers in s_range:
            k_means = sklearn.cluster.KMeans(n_clusters=i_speakers)
            k_means.fit(mid_term_features_norm.T)
            cls = k_means.labels_
            means = k_means.cluster_centers_
            cls_all.append(cls)
            centers_all.append(means)
            sil_a = []
            sil_b = []
            for c_speaker in range(i_speakers):
                cluster_percent = numpy.nonzero(cls == c_speaker)[0].shape[0] / float(len(cls))
                if cluster_percent < 0.020:
                    sil_a.append(0.0)
                    sil_b.append(0.0)
                else:
                    mid_term_features_norm_temp = mid_term_features_norm[:, cls == c_speaker]
                    y_t = distance.pdist(mid_term_features_norm_temp.T)
                    sil_a.append(numpy.mean(y_t) * cluster_percent)
                    sil_bs = []
                    for c_two in range(i_speakers):
                        if c_two != c_speaker:
                            cluster_percent_two = numpy.nonzero(cls == c_two)[0].shape[0] \
                                                  / float(len(cls))
                            mid_term_features_norm_temp_two = \
                                mid_term_features_norm[:, cls == c_two]
                            y_t = distance.cdist(mid_term_features_norm_temp.T,
                                                 mid_term_features_norm_temp_two.T)
                            sil_bs.append(numpy.mean(y_t) *
                                          (cluster_percent + cluster_percent_two) / 2.0)
                    sil_bs = numpy.array(sil_bs)
                    sil_b.append(min(sil_bs))
            sil_a = numpy.array(sil_a)
            sil_b = numpy.array(sil_b)
            sil = []
            for c_speaker in range(i_speakers):
                sil.append((sil_b[c_speaker] - sil_a[c_speaker]) /
                           (max(sil_b[c_speaker], sil_a[c_speaker]) + 0.00001))
            sil_all.append(numpy.mean(sil))
        imax = int(numpy.argmax(sil_all))
        num_speaker[imax] = num_speaker[imax] + 1
        num_speaker_cls[imax] = cls_all[imax]

    imax = num_speaker.index(max(num_speaker))
    n_speakers_final = s_range[imax]
    return n_speakers_final, imax, num_speaker_cls
