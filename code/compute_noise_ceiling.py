"""
This scripts runs ridge regressino model to predict from one subjects'
data to the other to estimate a noise ceiling.
"""

from scipy.io import loadmat
import torch
from scipy import stats
import numpy as np
from tqdm import tqdm
import argparse
import os
from sklearn.model_selection import ShuffleSplit, KFold, PredefinedSplit
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from encodingmodel.encoding_model import ridge_cv
from util.util import *

subj_n = 3
if os.path.isdir("../BOLD5000_cortical"):
    cortical_dir = "../BOLD5000_cortical"
else:
    cortical_dir = "/data2/tarrlab/common/datasets/BOLD5000_cortical"

parser = argparse.ArgumentParser()
parser.add_argument("--whole_brain", action="store_true")
parser.add_argument("--kernel", action="store_true")
parser.add_argument("--no_cv", action="store_true")
args = parser.parse_args()


def load_single_brain_data(subj_id, whole_brain=True):
    with open("../BOLD5000/CSI0{}_stim_lists.txt".format(subj_id)) as f:
        sl = f.readlines()
    stims = [item.strip("\n") for item in sl]
    subset_idx = get_nonrep_index(stims)
    assert len(subset_idx) == 4916

    if whole_brain:
        # Load whole brain data (the one that is zscored across runs)
        br_data_tr3 = np.load(
            "{}/CSI{}_TR{}_zscore.npy".format(cortical_dir, subj_id, 3)
        )
        br_data_tr4 = np.load(
            "{}/CSI{}_TR{}_zscore.npy".format(cortical_dir, subj_id, 4)
        )
        data = (br_data_tr3 + br_data_tr4) / 2
    else:
        data = [
            loadmat("../BOLD5000/CSI{}_ROIs_TR{}.mat".format(subj_id, tr))
            for tr in [3, 4]
        ]
    return data[subset_idx, :]


def load_data(subj=None):
    if subj is None:
        br_data = list()
        for subj in range(subj_n):
            # Load brain data
            br_data.append(load_single_brain_data(subj + 1))
        assert len(br_data) == subj_n
    return br_data


def load_kernel(subj_id):
    print("loading kernal for subject {}".format(subj_id))
    kfname = "../outputs/cross_subject_corr/subj{}_response_kernel.npy".format(subj_id)
    try:
        k = np.load(kfname)
    except FileNotFoundError:
        print("Kernel not found. Computing one.")
        br_data = load_single_brain_data(subj_id)
        k = br_data @ br_data.T
        np.save(kfname, k)
    return k


def compute_corr_of_subject_using_normal_ridge():
    corr_mat = np.zeros((subj_n, subj_n))
    rsq_mat = corr_mat.copy()
    br_data = load_data()
    for i in tqdm(range(subj_n)):
        for j in tqdm(range(subj_n - 1)):
            if args.whole_brain:
                X = br_data[i]
                y = br_data[j]
                output = ridge_cv(X, y, split_by_runs=False)
                np.save("../outputs/cross_subject_corr/subj{}_to_subj{}.npy", output)
                corr_mat[i, j + 1] = output[0]
                rsq_mat[i, j + 1] = output[1]
            else:
                print("Haven't implemented for ROIs yet")
                pass
    np.save("../outputs/cross_subject_corr/corr_mat.npy", corr_mat)
    np.save("../outputs/cross_subject_corr/rsq_mat.npy", rsq_mat)


def fit(K, y, alpha):
    return np.linalg.inv(K + alpha * np.eye(K.shape[0])) @ y


def kernel_ridge_cv(K, y, cv=True, nfold=7, tol=8):
    if args.no_cv:
        cv = False
    print(K.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    scoring = lambda y, yhat: -mean_squared_error(yhat, y)

    alphas = np.logspace(-tol, 1 / 2 * np.log10(K.shape[1]) + tol, 10)

    # split train and test set
    train_index, test_index = next(ShuffleSplit(test_size=0.15).split(K, y))
    K_train = K[train_index, :][:, train_index]
    y_train = y[train_index, :]
    y_test = y[test_index, :]

    if cv:
        kfold = KFold(n_splits=nfold)
    else:
        tr_index, val_index = next(ShuffleSplit(test_size=0.15).split(K_train, y_train))
        test_fold = np.zeros(K_train.shape[0])
        test_fold[tr_index] = -1
        kfold = PredefinedSplit(test_fold)
        assert kfold.get_n_splits() == 1

    cv_mean = list()
    for alpha in tqdm(alphas):
        scores = list()
        for tr_index, val_index in kfold.split(K_train):
            print("Fitting ridge models...")
            yval = y_train[val_index]
            tmp = fit(K_train[tr_index, :][:, tr_index], y_train[tr_index], alpha)
            ypred = K_train[val_index, :][:, tr_index] @ tmp
            scores.append(scoring(yval, ypred))
        cv_mean.append(np.mean(scores))

    best_alpha = np.argmax(cv_mean)

    print("Making predictions using ridge models...")
    yhat = (
        K[test_index, :][:, train_index]
        @ np.linalg.inv(K_train + best_alpha * np.eye(K_train.shape[0]))
        @ y_train
    )
    rsqs = [r2_score(y_test[:, i], yhat[:, i]) for i in range(y_test.shape[1])]
    corrs = [pearsonr(y_test[:, i], yhat[:, i]) for i in range(y_test.shape[1])]

    return corrs, rsqs


def compute_whole_brain_corr_of_subject_using_kernel_ridge():
    # load/compute kernel first
    for i in tqdm(np.arange(subj_n) + 1):  # 1, 2, 3
        k = load_kernel(i)
        for j in tqdm(np.arange(subj_n) + 1):
            save_root = "../outputs/cross_subject_corr/subj{}_to_subj{}".format(i, j)
            exist = os.path.isfile(save_root + "_rsqs.npy")
            if exist:
                print("Already exist results from subj {} to subj {}".format(i, j))
            else:
                y = load_single_brain_data(j)
                corrs, rsqs = kernel_ridge_cv(k, y)
                np.save(save_root + "_corrs.npy", corrs)
                np.save(save_root + "_rsqs.npy", rsqs)


if __name__ == "__main__":
    if args.kernel and args.whole_brain:
        compute_whole_brain_corr_of_subject_using_kernel_ridge()
