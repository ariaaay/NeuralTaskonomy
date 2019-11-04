"""
This scripts computes the dendrogram of tasks based on prediction
on all voxels of the whole brain.
"""


import argparse

# import plotly
# import plotly.figure_factory as ff

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from util.model_config import *

import numpy as np
from scipy.spatial.distance import pdist, squareform
import seaborn as sns


def sim(voxel_mat, metric=None):
    if metric is None:
        dist = squareform(pdist(voxel_mat))
    else:
        dist = squareform(pdist(voxel_mat, metric))
    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int)
    parser.add_argument("--method", type=str, default="masked_corr")
    args = parser.parse_args()

    matrix_path = "../outputs/task_matrix"
    if args.method == "sig":
        mat = np.load("{}/sig_mask_subj{}.npy".format(matrix_path, args.subj)).astype(
            int
        )
    elif args.method == "masked_corr":
        mat = np.load("../outputs/task_matrix/mask_corr_subj{}_emp_fdr.npy".format(args.subj))

    for linkage in ["average", "ward"]:
        Z = hierarchy.linkage(mat, linkage)
        # X = sim(mat,  metric=lambda u, v: u @ v)
        plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 1, 1)
        labs = list(task_label.values())[:21]
        dn = hierarchy.dendrogram(Z, ax=ax, labels=labs, leaf_font_size=15, color_threshold=0,
                                  above_threshold_color='gray')
        plt.xticks(rotation="vertical")

        #post hoc hand code node colors
        # if args.subj == 1:
        color_list = ["blue"]*9 + ["green"]*10 + ["purple"]*2


        [t.set_color(i) for (i, t) in zip(color_list, ax.xaxis.get_ticklabels())]

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_yaxis().set_visible(False)
        # plt.margins(0.2)
        plt.subplots_adjust(bottom=0.5)
        plt.savefig(
            "../figures/tasktree/dendrogram_subj{}_{}_{}.pdf".format(args.subj, args.method, linkage)
        )
