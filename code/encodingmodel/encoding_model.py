import os
import torch
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    PredefinedSplit,
    train_test_split,
    GroupShuffleSplit,
    ShuffleSplit,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

from util.util import *
from util.model_config import *
from encodingmodel.ridge import RidgeCVEstimator


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if os.path.isdir("../BOLD5000_cortical"):
    cortical_dir = "../BOLD5000_cortical"
else:
    cortical_dir = "/data2/tarrlab/common/datasets/BOLD5000_cortical"


# def gelnet_cv(X, Xval, Xtest, ys, yval, ytest, ns):
# 	support = list()
# 	l_rs = torch.logspace(-10, 10, 100).tolist()
#
# 	X = [torch.from_numpy(x).to(dtype=torch.float64).to(device) for x in X]
# 	Xval = [torch.from_numpy(xval).to(dtype=torch.float64).to(device) for xval in Xval]
# 	Xtest = [torch.from_numpy(xtest).to(dtype=torch.float64).to(device) for xtest in Xtest]
#
# 	for i in tqdm(range(ys.shape[1])): #loop through the voxels
# 		y = torch.from_numpy(ys[:,i]).to(dtype=torch.float64).to(device)
# 		yval = torch.from_numpy(yval[:,i].to(dtype=torch.float64.to(device)))
# 		ytest = torch.from_numpy(ytest[:,i]).to(dtype=torch.float64).to(device)
#
# 		summaries = gelnet(X, y, Xval, yval, Xtest, ytest, l_rs, ns, verbose=True, device=device)
# 		best_summ = min(summaries.items(), key=lambda z: z[1][1])
#
# 		support.append(best_summ[1][0])
# 	return support


def ridge_cv(
        X,
        y,
        split_by_runs,
        run_group=None,
        pca=False,
        tol=8,
        nfold=7,
        cv=False,
        fix_testing=False,
        permute_y=False,
):
    # fix_tsesting can be True (42), False, and a seed
    if fix_testing is True:
        fix_testing_state = 42
    elif fix_testing is False:
        fix_testing_state = None
    else:
        fix_testing_state = fix_testing

    scoring = lambda y, yhat: -torch.nn.functional.mse_loss(yhat, y)

    alphas = torch.from_numpy(
        np.logspace(-tol, 1 / 2 * np.log10(X.shape[1]) + tol, 100)
    )

    # split train and test set
    if split_by_runs:
        train_index, test_index = next(
            GroupShuffleSplit(test_size=0.15, random_state=fix_testing_state).split(X, y, run_group)
        )
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        run_group_train = np.array(run_group)[train_index]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=fix_testing_state
        )

    if pca:
        print("Running PCA...")
        pca = PCA()
        X_train = pca.fit_transform(X_train.astype(np.float32))
        X_test = pca.transform(X_test)
        print("PCA Done.")

    X_train = torch.from_numpy(X_train).to(dtype=torch.float64).to(device)
    y_train = torch.from_numpy(y_train).to(dtype=torch.float64).to(device)
    X_test = torch.from_numpy(X_test).to(dtype=torch.float64).to(device)

    # model selection
    if split_by_runs:
        if cv:
            kfold = GroupKFold(n_splits=nfold)
        else:
            tr_index, _ = next(
                GroupShuffleSplit(n_splits=nfold).split(
                    X_train, y_train, run_group_train
                )
            )
            test_fold = np.zeros(X_train.shape[0])
            test_fold[tr_index] = -1
            kfold = PredefinedSplit(test_fold)
            assert kfold.get_n_splits() == 1

    else:
        if cv:
            kfold = KFold(n_splits=nfold)
        else:
            tr_index, _ = next(
                ShuffleSplit(test_size=0.15).split(X_train, y_train)  # split training and testing
            )
            # set predefined train and validation split
            test_fold = np.zeros(X_train.shape[0])
            test_fold[tr_index] = -1
            kfold = PredefinedSplit(test_fold)
            assert kfold.get_n_splits() == 1

    clf = RidgeCVEstimator(alphas, kfold, scoring, scale_X=False)

    print("Fitting ridge models...")
    if split_by_runs:
        clf.fit(X_train, y_train, groups=run_group_train)
    else:
        clf.fit(X_train, y_train)

    print("Making predictions using ridge models...")
    yhat = clf.predict(X_test).cpu().numpy()
    rsqs = [r2_score(y_test[:, i], yhat[:, i]) for i in range(y_test.shape[1])]
    corrs = [pearsonr(y_test[:, i], yhat[:, i]) for i in range(y_test.shape[1])]

    if not permute_y:
        return (
            corrs,
            rsqs,
            clf.mean_cv_scores.cpu().numpy(),
            clf.best_l_scores.cpu().numpy(),
            clf.best_l_idxs.cpu().numpy(),
            [yhat, y_test],
        )

    else:  # permutation testings
        print("running permutation test (permutating test labels 5000 times).")
        repeat = 5000
        corrs_dist = list()
        label_idx = np.arange(y_test.shape[0])
        for _ in tqdm(range(repeat)):
            np.random.shuffle(label_idx)
            y_test_perm = y_test[label_idx, :]
            perm_corrs = pearson_corr(y_test_perm, yhat, rowvar=False)
            corrs_dist.append(perm_corrs)
        corr_only = [r[0] for r in corrs]
        p = empirical_p(corr_only, np.array(corrs_dist))
        assert len(p) == y_test.shape[1]
        return corrs_dist, p, None


def fit_encoding_model(
        X,
        br,
        model_name=None,
        byROI=True,
        subset_idx=None,
        subj=1,
        split_by_runs=False,
        pca=False,
        fix_testing=False,
        cv=False,
        saving=True,
        permute_y=False,
        stacking=False,
        model_list=None,
):
    if cv:
        print("Running cross validation")

    outpath = "../outputs/encoding_results/subj{}/".format(subj)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    corrs_array, rsqs_array, cv_array, l_score_array, best_l_array, predictions_array = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    if split_by_runs:
        # length of run info should the same as the length of brain data
        # total of 141 runs, each has 37 images
        with open("{}/CSI{}_events.json".format(cortical_dir, subj)) as f:
            events = json.load(f)
        sessions = np.array(events["session"])[subset_idx]
        runs = np.array(events["runs"])[subset_idx]
        run_group = []
        for i, _ in enumerate(runs):
            run_group.append(sessions[i] + "_" + runs[i])
    else:
        run_group = None

    if byROI:
        wb = ""
        for j in range(len(SIDE)):
            for i in range(len(ROIS)):
                print(SIDE[j] + ROIS[i])
                y = load_brain_data(br, SIDE[j], ROIS[i], subset_idx=subset_idx)
                assert y.shape[0] == X.shape[0]

                # split train and test
                corrs, *cv_outputs = ridge_cv(
                    X,
                    y,
                    split_by_runs,
                    run_group,
                    pca=pca,
                    cv=False,
                    fix_testing=fix_testing,
                    permute_y=permute_y,
                )

                if permute_y:  # save permutation test results now to save space
                    outdir = "../outputs/permutation_results/subj{}/".format(subj)
                    if not os.path.isdir(outdir):
                        os.makedirs(outdir)
                    pickle.dump(
                        corrs,
                        open(
                            "../outputs/permutation_results/subj{}/permutation_test_on_test_data_corr_{}_{}{}.p".format(
                                subj, model_name, SIDE[j], ROIS[i]
                            ),
                            "wb",
                        ),
                    )
                    pickle.dump(
                        cv_outputs[0],
                        open(
                            "../outputs/permutation_results/subj{}/permutation_test_on_test_data_pvalue_{}_{}{}.p".format(
                                subj, model_name, SIDE[j], ROIS[i]
                            ),
                            "wb",
                        ),
                    )

                else:
                    corrs_array.append(corrs)  # append values of all ROIs
                    if len(cv_outputs) > 0:
                        rsqs_array.append(cv_outputs[0])
                        cv_array.append(cv_outputs[1])
                        l_score_array.append(cv_outputs[2])
                        best_l_array.append(cv_outputs[3])
                        predictions_array.append(cv_outputs[4])

    else:
        wb = "_whole_brain"
        if subset_idx is None:
            y = br
        else:
            y = br[subset_idx, :]  # subset to get rid of the repetition trials

        if stacking:
            if not os.path.isdir(outpath + "stacking/"):
                os.makedirs(outpath + "stacking/")
            err_list, yhats = list(), list()
            y_test = None

            n_features = len(X)
            # seed = np.random.randint(0, high=2**32-1, size=1).item()

            for k in range(n_features):
                assert (
                y.shape[0] == X[k].shape[0]
                )  # test that shape of features spaces and the brain are the same
                _, *cv_outputs = ridge_cv(
                    X[k],
                    y,
                    split_by_runs,
                    run_group,
                    pca=pca,
                    cv=False,
                    fix_testing=True, # to save fix train and test splits
                    permute_y=False #TODO: maybe add permutation support?
                )
                # check train test split are the same across datasets
                if y_test is None:
                    np.save("{}stacking/ytest.npy".format(outpath), cv_outputs[4][1])
                # else:
                #     assert y_test == cv_outputs[4][1]

                # err_list.append(cv_outputs[0])
                yhat = cv_outputs[4][0]
                np.save("{}stacking/yhat_{}.npy".format(outpath, model_list[k]), yhat)
                y_test = cv_outputs[4][1]
                yhats.append(yhat)

                err_list.append(mean_squared_error(yhat, y_test, multioutput='raw_values'))
                # print(np.array(err_list).shape)
                try:
                    assert np.array(err_list).shape == (k+1, y_test.shape[1])
                except AssertionError:
                    print("Err list has wrong shape.")

            S, stacked_r2s = stack(np.array(err_list), np.array(yhats), y_test)
            np.save("{}stacking/S_{}.npy".format(outpath, model_name), S)
            np.save("{}stacking/rsq_{}.npy".format(outpath, model_name), stacked_r2s)

        else:
            assert (
                y.shape[0] == X.shape[0]
            )  # test that shape of features spaces and the brain are the same

            corrs_array, *cv_outputs = ridge_cv(
                X,
                y,
                split_by_runs,
                run_group,
                pca=pca,
                cv=False,
                fix_testing=fix_testing,
                permute_y=permute_y,
            )

        if permute_y:  # if running permutation just return subsets of the output
            # save correaltions
            np.save(
                "../outputs/permutation_results/subj{}/permutation_test_on_test_data_corr{}_{}.npy".format(
                    subj, wb, model_name
                ),
                np.array(corrs_array),
            )
            # save p-values
            pickle.dump(
                cv_outputs[0],
                open(
                    "../outputs/permutation_results/subj{}/permutation_test_on_test_data_pvalue{}_{}.p".format(
                        subj, wb, model_name
                    ),
                    "wb",
                ),
            )
            # return np.array(corrs_array), cv_outputs[0]


    if saving:
        pickle.dump(
            corrs_array, open(outpath + "corr{}_{}.p".format(wb, model_name), "wb")
        )

        if len(cv_outputs) > 0:
            pickle.dump(
                cv_outputs[0], open(outpath + "rsq{}_{}.p".format(wb, model_name), "wb")
            )
            pickle.dump(
                cv_outputs[1],
                open(outpath + "cv_score{}_{}.p".format(wb, model_name), "wb"),
            )
            pickle.dump(
                cv_outputs[2],
                open(outpath + "l_score{}_{}.p".format(wb, model_name), "wb"),
            )
            pickle.dump(
                cv_outputs[3],
                open(outpath + "best_l{}_{}.p".format(wb, model_name), "wb"),
            )

            if fix_testing:
                pickle.dump(
                    cv_outputs[4],
                    open(outpath + "pred{}_{}.p".format(wb, model_name), "wb"),
                )

    return np.array(corrs_array), None


def permutation_test(
        X,
        y,
        model_name,
        repeat=5000,
        byROI=True,
        subset_idx=None,
        subj=1,
        split_by_runs=False,
        pca=False,
        permute_y=True,  # rather than permute training
):
    """
	Running permutation test (permute the label 5000 times).
	"""
    outdir = "../outputs/permutation_results/subj{}/".format(subj)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if byROI:
        wb = ""
    else:
        wb = "_whole_brain"

    print("Running permutation test of {} for {} times".format(model_name, repeat))
    corr_dists, rsq_dists = list(), list()
    if permute_y:  # permute inside ridge cv
        print("Permutation testing by permuting test data.")
        _ = fit_encoding_model(
            X,
            y,
            model_name=model_name,
            subj=subj,
            byROI=byROI,
            subset_idx=subset_idx,
            split_by_runs=split_by_runs,
            pca=pca,
            cv=False,
            saving=False,
            permute_y=True,
            fix_testing=False,
        )
        # if not byROI:  # ROIS results are already saved before this line is run #TODO： test permutation again
        # np.save(
        #     "../outputs/permutation_results/subj{}/permutation_test_on_test_data_corr{}_{}.npy".format(
        #         subj, wb, model_name
        #     ),
        #     np.array(corr_dists),
        # )
        # pickle.dump(
        #     ps,
        #     open(
        #         "../outputs/permutation_results/subj{}/permutation_test_on_test_data_pvalue{}_{}.p".format(
        #             subj, wb, model_name
        #         ),
        #         "wb",
        #     ),
        # )
    else:
        label_idx = np.arange(X.shape[0])
        for _ in tqdm(range(repeat)):
            np.random.shuffle(label_idx)
            X_perm = X[label_idx, :]
            corrs_array, *cv_outputs = fit_encoding_model(
                X_perm,
                y,
                model_name=model_name,
                subj=subj,
                byROI=byROI,
                subset_idx=subset_idx,
                split_by_runs=split_by_runs,
                pca=pca,
                cv=False,
                saving=False,
                fix_testing=False,
            )
        corr_dists.append(corrs_array)
        # rsq_dists.append(rsqs_array)

        pickle.dump(
            corr_dists,
            open(
                "../outputs/permutation_results/subj{}/permutation_test_on_training_data_corr{}_{}.p".format(
                    subj, wb, model_name
                ),
                "wb",
            ),
        )
    # pickle.dump(
    #     rsq_dists,
    #     open(
    #         "../outputs/permutation_results/subj{}/permutation_test_rsq{}_{}_{}.p".format(
    #             subj, wb, model_name, permute_data
    #         ),
    #         "wb",
    #     ),
    # )
