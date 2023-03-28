import numpy  as np
import copy


def wst_reduction(stats, J, Q, orientations, center_vertex_id, m_norm=0.0, s_norm=1.0):
    #
    # Initial formatting check
    #
    for key in stats.keys():
        if key not in ['S0', 'S1', 'S2']:
            raise Exception(f"Invalid keys {stats.keys()}")
    assert 'S0' in stats.keys() and 'S1' in stats.keys()
    assert stats['S0'].ndim == 2
    assert stats['S1'].ndim == 4
    if 'S2' in stats.keys():
        assert stats['S2'].ndim == 5
    X = copy.deepcopy(stats)
    X_red = {}

    #
    # Normalization
    #
    def S2_normalization(x_s1, x_s2):
        ret = x_s2.copy()
        cnt = 0
        for j1 in range(J*Q):
            for j2 in range(j1 + 1, J*Q):
                ret[..., cnt, :, :] /= x_s1[..., j1, :, np.newaxis]
                cnt += 1
        return ret
    if 'S2' in X.keys():
        X['S2'] = S2_normalization(X['S1'], X['S2'])
    X['S1'] = X['S1'] / X['S0'][..., np.newaxis, np.newaxis]
    
    # 
    # Standardization
    #

    X['S0'] = np.log10(X['S0'])
    X['S1'] = np.log10(X['S1'])
    if 'S2' in X.keys():
        X['S2'] = np.log10(X['S2'])

    X['S0'] = (X['S0'] - m_norm) / s_norm
    X['S1'] = (X['S1'] - m_norm) / s_norm
    if 'S2' in X.keys():
        X['S2'] = (X['S2'] - m_norm) / s_norm

    # NaN check
    for key in X.keys():
        assert np.isnan(X[key]).any() == False

    # 
    # Reduction
    #

    # Orientations variables
    nb_vertices = len(orientations)
    cos_centered_bins = {}
    for i in range(nb_vertices):
        cos_centered = np.dot(orientations[i], orientations[center_vertex_id]).round(5)
        if cos_centered not in cos_centered_bins.keys():
            cos_centered_bins[cos_centered] = []
        cos_centered_bins[cos_centered].append(i)
    cos_pairs_bins = {}
    for i in range(nb_vertices):
        for j in range(nb_vertices):
            cos_pair = np.dot(orientations[i], orientations[j]).round(5)
            if cos_pair not in cos_pairs_bins.keys():
                cos_pairs_bins[cos_pair] = []
            cos_pairs_bins[cos_pair].append((i, j))
    cos_pairs_centered_bins = {}
    for i in range(nb_vertices):
        for j in range(nb_vertices):
            cos_centered_i = np.dot(orientations[i], orientations[center_vertex_id]).round(5)
            cos_centered_j = np.dot(orientations[j], orientations[center_vertex_id]).round(5)
            cos_pair = np.dot(orientations[i], orientations[j]).round(5)
            triplet = (cos_centered_i, cos_centered_j, cos_pair)
            if triplet not in cos_pairs_centered_bins.keys():
                cos_pairs_centered_bins[triplet] = []
            cos_pairs_centered_bins[triplet].append((i, j))

    def S1_reduction(x):
        ret_0 = x.mean(axis=-1)

        ret_1 = np.zeros(x.shape[:-1] + (len(cos_centered_bins.keys()),))
        for i, key in enumerate(cos_centered_bins.keys()):
            ret_1[..., i] = x[..., cos_centered_bins[key]].mean(axis=-1)
        
        return ret_0, ret_1

    def S2_reduction(x):
        ret_0 = x.mean(axis=(-1, -2))

        ret_1 = np.zeros(x.shape[:-2] + (len(cos_pairs_bins.keys()),))
        for i, key in enumerate(cos_pairs_bins.keys()):
            for elt in cos_pairs_bins[key]:
                ret_1[..., i] += x[..., elt[0], elt[1]]
            ret_1[..., i] /= len(cos_pairs_bins[key])

        ret_2 = np.zeros(x.shape[:-2] + (len(cos_pairs_centered_bins.keys()),))
        for i, key in enumerate(cos_pairs_centered_bins.keys()):
            for elt in cos_pairs_centered_bins[key]:
                ret_2[..., i] += x[..., elt[0], elt[1]]
            ret_2[..., i] /= len(cos_pairs_centered_bins[key])
        
        return ret_0, ret_1, ret_2

    X_red["S0"] = X['S0']
    X_red["S1iso"], X_red["S1rsd"] = S1_reduction(X['S1'])
    if 'S2' in X.keys():
        X_red["S2iso1"], X_red["S2iso2"], X_red["S2rsd"] = S2_reduction(X['S2'])

    return X_red
