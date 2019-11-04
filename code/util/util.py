# Author: Aria Wang
import os
import numpy as np
import pickle
import json
import re
from math import sqrt
import numpy as np


ROIS = ["OPA", "PPA", "LOC", "EarlyVis", "RSC"]
SIDE = ["LH", "RH"]

cortical_dir = "../BOLD5000_cortical"

def zero_strip(s):
    if s[0] == "0":
        s = s[1:]
        return zero_strip(s)
    else:
        return s


def zscore(mat, axis=None):
    if axis is None:
        return (mat - np.mean(mat)) / np.std(mat)
    else:
        return (mat - np.mean(mat, axis=axis, keepdims=True)) / np.std(
            mat, axis=axis, keepdims=True
        )


def pearson_corr(X, Y, rowvar=True):
    if rowvar:
        return np.mean(zscore(X, axis=1) * zscore(Y, axis=1), axis=1)
    else:
        return np.mean(zscore(X, axis=0) * zscore(Y, axis=0), axis=0)

def empirical_p(acc, dist, dim=2):
    # dist is permute times x num_voxels
    # acc is of length num_voxels
    if dim == 1:
        return np.sum(dist > acc) / dist.shape[0]
    elif dim == 2:
        assert len(acc) == dist.shape[1]
        ps = list()
        for i, r in enumerate(acc):
            ps.append(np.sum(dist[:,i] > r)/dist.shape[0])
        return ps

def extract_dataset_index(stim_list, dataset="all", rep=False, return_filename=False):
    with open("../outputs/stimuli_info/imgnet_imgsyn_dict.pkl", "rb") as f:
        stim_imgnet = pickle.load(f)

    with open("../outputs/stimuli_info/COCO_img2cats.json") as f:
        COCO_img2cats = json.load(f)  # categories info

    dataset_labels = stim_list.copy()
    COCO_idx, imagenet_idx, SUN_idx = list(), list(), list()
    COCO_cat_list, imagenet_cat_list, SUN_cat_list = list(), list(), list()
    for i, n in enumerate(stim_list):
        if "COCO_" in n:
            if "rep_" in n and rep is False:
                continue
            dataset_labels[i] = "COCO"
            COCO_idx.append(i)
            n.split()  # takeout \n
            COCO_id = zero_strip(str(n[21:-5]))
            COCO_cat_list.append(COCO_img2cats[COCO_id])
        elif "JPEG" in n:  # imagenet
            dataset_labels[i] = "imagenet"
            if "rep_" in n:
                if not rep:
                    continue
                else:
                    n = n[4:]
            syn = stim_imgnet[n[:-1]]
            imagenet_idx.append(i)
            imagenet_cat_list.append(syn)
        else:
            dataset_labels[i] = "SUN"
            name = n.split(".")[0]
            if "rep_" in name:
                if not rep:
                    continue
                else:
                    name = name[4:]
            SUN_idx.append(i)
            if return_filename:
                SUN_cat_list.append(name)
            else:
                SUN_cat_list.append(re.split("[0-9]", name)[0])
    if rep:
        assert len(stim_list) == len(COCO_idx) + len(imagenet_idx) + len(SUN_idx)

    if dataset == "COCO":
        return COCO_idx, COCO_cat_list
    elif dataset == "imagenet":
        return imagenet_idx, imagenet_cat_list
    elif dataset == "SUN":
        return SUN_idx, SUN_cat_list
    else:
        return dataset_labels


def grab_brain_data(datamat, rois, side=None, zscore_flag=False):
    """
	grab brain data matrix based on ROIS and sides of the brain
	"""
    if side is None:
        name = rois
    else:
        name = side + rois
    X = datamat[name]
    if zscore_flag:
        return zscore(X, axis=1)
    else:
        return X


def load_brain_data(br, side, roi, subset_idx=None):
    """
    :param br: Brain data matrix. Could be a single dictionary where names are ROI names or a list of dictionaries.
    :param side: side of the brain to extract
    :param roi: roi of the brain to extract
    :param subset_idx: subsetting brain trials to get non-repete trials or specific datasets
    :return: the roi data
    """
    if type(br) is list:
        bd = [grab_brain_data(b, roi, side, zscore_flag=False) for b in br]
        roi_data = (bd[0] + bd[1]) / 2  # average across TRs
    else:
        roi_data = grab_brain_data(br, roi, side, zscore_flag=False)

    if (
        subset_idx is None
    ):  # subset to get rid of the repetition trials or specific dataset
        return roi_data
    else:
        return roi_data[subset_idx, :]


def count_repetition(stim_list, br_idx):
    count = 0
    for item in np.array(stim_list)[br_idx]:
        if "rep" in item:
            count += 1
    return count / len(np.array(stim_list)[br_idx])


def pool_size(fm, dim, adaptive=True):
    import torch

    if adaptive:  # use adaptive avgpool instead
        c = fm.shape[1]
        k = int(np.floor(np.sqrt(dim / c)))
    else:
        k = 1
        tot = torch.numel(torch.Tensor(fm.view(-1).shape))
        ctot = tot
        while ctot > dim:
            k += 1
            ctot = tot / k / k

    return k


def find_overlap(cat_list, wv_list, br_idx_list, unique=False):
    """
	find overlap categories in brain space and co-occurences space;
	replicate: have a list of list for the repetition trials output:
	wv_idx: list of the categories that are presented in the brain data
	br_idx: a list of lists of trials that are associated with categories in wv_idx

	replicate: replicating the categories or object vectors according to repetition in the brain data
	unique: only include one of the trials among the repetition of same images in brain data
	when both are false: the br_idx will return a list of list that contains the index of all repeated trials
	"""
    br_idx = []  # brain
    wv_idx = []  # picture
    assert len(cat_list) == len(
        br_idx_list
    )  # length of categories list should be same as brain trials
    for s in wv_list:
        if s in cat_list:
            wv_pos = wv_list.index(s)  # find its index in the wordvectors
            if unique:
                br_idx.append(
                    br_idx_list[cat_list.index(s)]
                )  # find the corresponding index for brain response
                wv_idx.append(wv_pos)  # find the corresponding index for word vectors
                continue
            else:
                # otherwise get all the brain response index corresponding to that category
                tmp_idx = [br_idx_list[i] for i, x in enumerate(cat_list) if x == s]
                br_idx += tmp_idx
                wv_idx += [wv_pos] * len(tmp_idx)

    assert len(br_idx) == len(wv_idx)
    return br_idx, wv_idx


def get_nonrep_index(stim_list):
    idx_list = []
    for index, img_name in enumerate(stim_list):
        if "rep_" in img_name:
            continue  # repeated images are NOT included in the training and testing sets
        idx_list.append(index)
    return idx_list


def check_nans(data, clean=False):
    if np.sum(np.isnan(data)) > 0:
        print("NaNs in the data")
        if clean:
            nan_sum = np.sum(np.isnan(data), axis=1)
            new_data = data[nan_sum < 1, :]
            print("Original data shape is " + data.shape)
            print("NaN free data shape is " + new_data.shape)
            return new_data
    else:
        return data


def pytorch_pca(x):
    x_mu = x.mean(dim=0, keepdim=True)
    x = x - x_mu

    _, s, v = x.svd()
    s = s.unsqueeze(0)
    nsqrt = sqrt(x.shape[0] - 1)
    xp = x @ (v / s * nsqrt)

    return xp


def pca_test(x):
    from sklearn.decomposition import PCA

    pca = PCA(whiten=True, svd_solver="full")
    pca.fit(x)
    xp = pca.transform(x)
    return xp


def sum_squared_error(x1, x2):
    return np.sum((x1 - x2) ** 2, axis=0)


def generate_rdm(mat, idx=None, avg=False):
    """
	Generate rdm based on data selected by the idx
	idx: lists of index if averaging is not needed; list of list of index if averaging is needed
	"""
    from scipy.spatial.distance import squareform
    from scipy.spatial.distance import pdist

    if idx is None:
        idx = np.arange(mat.shape[0])

    if type(mat) == list:
        data = np.array(mat)[idx]
        return np.corrcoef(data)
    if avg:
        data = np.zeros((len(idx), mat.shape[1]))
        for i in range(len(idx)):
            data[i] = np.mean(mat[idx[i], :], axis=0)
    else:
        data = mat[idx, :]

    dist = squareform(pdist(data, "cosine"))
    return dist


if __name__ == "__main__":
    import torch
    from scipy.stats import pearsonr

    # PCA test
    x = np.array([[12.0, -51, 4, 99], [6, 167, -68, -129], [-4, 24, -41, 77]])
    x = torch.from_numpy(x).to(dtype=torch.float64)
    xp1 = pytorch_pca(x)

    xp2 = pca_test(x)
    assert np.sum(abs(xp1.numpy() - xp2) > 0.5) == 0

    # correlation test
    a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    b = np.array([[12.0, -51, 4], [99, 6, 167],[-68, -129, -4], [24, -41, 77]])

    corr_row_1 = pearson_corr(a, b, rowvar=True).astype(np.float32)
    corr_row_2 = []
    for i in range(a.shape[0]):
        corr_row_2.append(pearsonr(a[i, :], b[i, :])[0].astype(np.float32))
    assert [corr_row_1[i] == corr_row_2[i] for i in range(len(corr_row_2))]

    corr_col_1 = pearson_corr(a, b, rowvar=False).astype(np.float32)
    corr_col_2 = []
    for i in range(a.shape[1]):
        corr_col_2.append(pearsonr(a[:, i], b[:, i])[0].astype(np.float32))
    assert [corr_col_1[i] == corr_col_2[i] for i in range(len(corr_col_2))]



