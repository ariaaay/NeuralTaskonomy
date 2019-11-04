"""
This scripts runs FDR correction on given p-values.
"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
from glob import glob

from util.model_config import *


def compute_adjust_p(
    model,
    feature="",
    subj=None,
    version=11,
    TR="TRavg",
    whole_brain=False,
    correction="bonferroni",
):
    if whole_brain:
        out = pickle.load(
            open(
                glob(
                    "../outputs/encoding_results_v{}/subj{}/corr_whole_brain_{}_{}_*{}.p".format(
                        version, subj, model, feature, TR
                    )
                )[0],
                "rb",
            )
        )
        # print(len(out))
        pvalues = np.array(out)[:, 1]
        if correction == "fdr":
            adj_p = fdrcorrection(pvalues)[1]
        elif correction == "bonferroni":
            adj_p = pvalues * len(pvalues)

    else:
        adj_p = list()
        out = pickle.load(
            open(
                glob(
                    "../outputs/encoding_results_v{}/subj{}/corr_{}_{}_*{}.p".format(
                        version, subj, model, feature, TR
                    )
                )[0],
                "rb",
            )
        )
        for roi in out:
            pvalues = np.array(roi)[:, 1]
            if correction == "fdr":
                adj_p.append(fdrcorrection(pvalues)[1])
            elif correction == "bonferroni":
                adj_p.append(pvalues * len(pvalues))
    return pvalues, adj_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--whole_brain", action="store_true")
    parser.add_argument("--model", default="taskrepr")
    parser.add_argument("--correction", default="fdr")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--use_empirical_p", action="store_true")
    args = parser.parse_args()

    if args.whole_brain:
        wb = "_whole_brain"
    else:
        wb = ""

    all_ps = list()
    try:
        flist = model_features[args.model]
    except KeyError:
        flist = [""]
    for f in flist:
        if args.use_empirical_p:
            adj_p = pickle.load(
                open(
                    "../outputs/baseline/empirical_p_values_taskrepr_{}_subj{}{}_test_permute_fdr.p".format(
                        f, args.subj, wb
                    ),
                    "rb",
                )
            )
        else:
            p, adj_p = compute_adjust_p(
                args.model,
                f,
                subj=args.subj,
                whole_brain=args.whole_brain,
                correction=args.correction,
            )
        # print(len(adj_p))
        if args.whole_brain:
            print(f + ": " + str(np.sum(np.array(adj_p) < args.alpha)))

            mask_dir = "../outputs/voxels_masks/subj{}".format(args.subj)
            if not os.path.isdir(mask_dir):
                os.makedirs(mask_dir)
            if args.use_empirical_p:
                np.save(
                    "{}/{}_{}_emp_{}_{}{}".format(
                        mask_dir, args.model, f, args.correction, args.alpha, wb
                    ),
                    adj_p < args.alpha,
                )
            else:
                np.save(
                    "{}/{}_{}_{}_{}{}".format(
                        mask_dir, args.model, f, args.correction, args.alpha, wb
                    ),
                    adj_p < args.alpha,
                )

        else:
            for i in range(len(adj_p)):
                print(ROI_labels[i] + ": " + str(len(adj_p[i])))
                print(f + ": " + str(np.sum(np.array(adj_p[i]) < args.alpha)))

        all_ps.append(adj_p)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_xscale("log")
        # sns.distplot(adj_p, bins=100, hist=True, kde=True, rug=False, ax=ax)
        # plt.savefig(
        #     "../figures/pvalues/adjusted_p_values_{}_{}_subj{}{}_{}.png".format(
        #         args.model, f, args.subj, wb, args.correction
        #     )
        # )
        if not args.use_empirical_p:
            pickle.dump(
                all_ps,
                open(
                    "../outputs/baseline/adjusted_p_values_{}_{}_subj{}{}_{}.p".format(
                        args.model, f, args.subj, wb, args.correction
                    ),
                    "wb",
                ),
            )
