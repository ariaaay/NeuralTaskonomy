"""
This scripts computes the similarity or distance matrix of tasks based on prediction
on all voxels of the whole brain.
"""

import argparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster import AgglomerativeClustering

from util.model_config import *

sns.set(style="whitegrid", font_scale=1)

def cross_subject_corrs(mat_list):
    rs = list()
    for i in range(len(mat_list)-1):
        for j in range(len(mat_list)-1):
            if i != j+1:
                r = pearsonr(mat_list[i].flatten(), mat_list[j+1].flatten())
                rs.append(r[0])
    return rs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="please specify subject to show")
    # parser.add_argument(
    #     "--subj", type=str, default="1", help="specify which subject to build model on"
    # )

    parser.add_argument(
        "--use_prediction",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_significance",
        default=False,
        action="store_true",
        help="use the overlap of significant voxels",
    )
    parser.add_argument(
        "--use_mask_corr",
        default=False,
        action="store_true",
        help="use the masked correlation matrix"
    )
    parser.add_argument(
        "--ROI", default=False, action="store_true", help="use ROI data only"
    )
    parser.add_argument(
        "--method",
        default="cosine",
        help="define what metric should be used to generate task matrix"
    )
    parser.add_argument(
        "--empirical",
        default=False,
        action="store_true",
        help="use masked results with permutation p values"
    )

    parser.add_argument(
        "--compute_correlation_across_subject",
        action="store_true"
    )

    args = parser.parse_args()

    if args.empirical:
        p_method = "emp_fdr"
    else:
        p_method = "fdr"

    n_tasks = 21 #21 tasks
    all_sim = np.zeros((n_tasks, n_tasks))
    if args.compute_correlation_across_subject:
        all_mat = list()

    for i in range(3):
        voxel_mat = np.load("../outputs/task_matrix/mask_corr_subj{}_{}.npy".format(i + 1, p_method))
        # voxel_mat = np.vstack((voxel_mat[:14,:], voxel_mat[15:,:])) #remove curvature
        print(voxel_mat.shape)
        if args.method == "l2":
            sim = euclidean_distances(voxel_mat)
        elif args.method == "cosine":
            sim = cosine_similarity(voxel_mat, dense_output=False)

        np.save("../outputs/task_matrix/task_matrix_subj{}_{}.npy".format(i + 1, args.method), sim)
        all_sim = all_sim + sim
        all_mat.append(sim)

    if args.compute_correlation_across_subject:
        corrs = cross_subject_corrs(all_mat)
        print("correlations of similarity matrix across subject is: " + str(corrs))

    model = AgglomerativeClustering(n_clusters=3, linkage='single', affinity="cosine").fit(all_sim)

    # print(model.labels_)
    order = np.argsort(model.labels_)
    fit_data = all_sim[order]
    fit_data = fit_data[:, order]
    fit_data = fit_data / 3

    # all_sim = all_sim/3
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    all_task_label = list(task_label.values())
    # active_labels = all_task_label[:14] + all_task_label[15:]
    labs = np.array(all_task_label)

    ax = sns.heatmap(fit_data,
                     cmap=cmap,
                     square=True, linewidths=.5,
                     xticklabels=labs[order],
                     yticklabels=labs[order])

    ax.set_ylim(0,21.1)
    plt.subplots_adjust(bottom=0.3)

    plt.savefig("../figures/taskmatrix/task_matrix_all_{}_{}.pdf".format(args.method, p_method))
