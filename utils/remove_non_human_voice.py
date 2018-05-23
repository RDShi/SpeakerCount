"""
判别是否为人声
"""
import pickle

def remove_nohuman(mid_term_features, threshold=-0):
    """
    输入MidTerm特征，用GMM判别是否为人声
    """
    mid_term_features = mid_term_features.T
    model_name = '..\\dataset\\voice_gmm'
    fid = open(model_name, 'rb')
    gmm_set = pickle.load(fid)
    fid.close()
    flags_ind = []
    for feature in mid_term_features:
        if gmm_set.score([feature]) > threshold:
            flags_ind.append(1)
        else:
            flags_ind.append(0)
    return flags_ind
