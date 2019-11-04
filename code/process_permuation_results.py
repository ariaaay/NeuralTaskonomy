"""
This scripts loads the permutation results and computed per roi or per voxels p-values.
"""


import seaborn as sns
import argparse
from glob import glob
from statsmodels.stats.multitest import fdrcorrection

from util.model_config import *
from util.util import *

perm_dir = "../outputs/permutation_results"
acc_dir = "../outputs/encoding_results_v11"
nrep = 5000


def load_data(dir, subj, model, feature, roi=None, type="pvalue"):
    if roi is not None:
        # print(subj, model, feature, roi, type)
        output = pickle.load(
            open(
                "{}/subj{}/permutation_test_on_test_data_{}_{}_{}__TRavg_{}.p".format(
                    dir, subj, type, model, feature, roi
                ),
                "rb",
            )
        )
    else:
        fname = "{}/subj{}/permutation_test_on_test_data_{}_whole_brain_{}_{}__TRavg.p".format(
            dir, subj, type, model, feature
        )
        f = open(fname, "rb")
        output = pickle.load(f)
    return output


def get_per_voxel_p(feature, subj, roi):
    if roi:
        roi_index = ROI_labels.index(roi)
        try:
            p = load_data(
                perm_dir,
                subj=subj,
                model="taskrepr",
                feature=feature,
                roi=roi,
                type="pvalue",
            )
        except FileNotFoundError:  # Load correlations
            try:
                corr_dist = load_data(
                    perm_dir,
                    subj=subj,
                    model="taskrepr",
                    feature=feature,
                    roi=roi,
                    type="corr",
                )
                assert np.array(corr_dist).shape[0] == nrep

                acc_path = "{}/subj{}/corr_taskrepr_{}__TRavg.p".format(
                    acc_dir, subj, feature
                )
                tmp = pickle.load(open(acc_path, "rb"))
                assert len(tmp) == 10
                roi_acc = tmp[roi_index]
                acc = [item[0] for item in roi_acc]  # get corrs from (corrs, p-value)
                p = empirical_p(acc, np.array(corr_dist))
                pickle.dump(
                    p,
                    open(
                        "../outputs/permutation_results/subj{}/permutation_test_on_test_data_pvalue_taskrepr_{}.p".format(
                            subj, roi
                        ),
                        "wb",
                    ),
                )
            except FileNotFoundError:
                print("task " + feature + " resutls doesnt exists yet.")

    else:
        try:
            p = load_data(
                perm_dir, subj=subj, model="taskrepr", feature=feature, type="pvalue"
            )
        except FileNotFoundError:
            print("task " + feature + " results doesnt exists yet.")

    return p


def get_roi_p(feature, subj, roi, correction="fdr"):
    try:
        roi_index = [
            i for i, e in enumerate(ROI_labels) if roi in e
        ]  # get index for left and right brain
        assert len(roi_index) == 2
        corr_dist_L = load_data(
            perm_dir,
            subj=subj,
            model="taskrepr",
            feature=feature,
            roi=ROI_labels[roi_index[0]],
            type="corr",
        )
        corr_dist_L = np.array(corr_dist_L)
        assert corr_dist_L.shape[0] == nrep

        corr_dist_R = load_data(
            perm_dir,
            subj=subj,
            model="taskrepr",
            feature=feature,
            roi=ROI_labels[roi_index[1]],
            type="corr",
        )
        corr_dist_R = np.array(corr_dist_R)
        assert corr_dist_R.shape[0] == nrep

        corr_dist = np.hstack((corr_dist_L, corr_dist_R))
        corr_avg = np.mean(corr_dist, axis=1)

        acc_path = "{}/subj{}/corr_taskrepr_{}__TRavg.p".format(acc_dir, subj, feature)
        tmp = pickle.load(open(acc_path, "rb"))
        assert len(tmp) == 10
        roi_acc_L = tmp[roi_index[0]]
        roi_acc_R = tmp[roi_index[1]]
        roi_acc = roi_acc_L + roi_acc_R
        assert len(roi_acc) == corr_dist.shape[1]

        acc = np.mean([item[0] for item in roi_acc])  # get corrs from (corrs, p-value)

        p = empirical_p(acc, corr_avg, dim=1)
        if correction == "fdr":
            p = fdrcorrection(p)[1]
        return p

    except FileNotFoundError:
        print("task " + feature + " results doesnt exists yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please specific subjects/features to load"
    )
    parser.add_argument("--subj", type=int, default=1)
    # parser.add_argument("--comparison", action="store_true", default=False)
    # parser.add_argument("--reload", action="store_true", default=False)
    parser.add_argument("--roi", action="store_true", default=False)

    args = parser.parse_args()

    for feature in taskrepr_features:
        print("Loading results for task {}".format(feature))
        if args.roi:
            ps = list()
            for roi in ROIS:
                p = get_roi_p(
                    feature,
                    args.subj,
                    roi,
                )
                print("pvalue for " + roi + " is: " + str(p))
                ps.append(p)

            pickle.dump(
                ps,
                open(
                    "../outputs/baseline/empirical_p_values_avg_taskrepr_{}_subj{}_test_permute.p".format(
                        feature, args.subj
                    ),
                    "wb",
                ),
            )

        else:
            p = get_per_voxel_p(feature, args.subj, roi=False)
            pickle.dump(
                p,
                open(
                    "../outputs/baseline/empirical_p_values_taskrepr_{}_subj{}_whole_brain_test_permute.p".format(
                        feature, args.subj
                    ),
                    "wb",
                ),
            )
            fdr_p = fdrcorrection(p)[1]
            pickle.dump(
                fdr_p,
                open(
                    "../outputs/baseline/empirical_p_values_taskrepr_{}_subj{}_whole_brain_test_permute_fdr.p".format(
                        feature, args.subj
                    ),
                    "wb",
                ),
            )
            # print(p)
