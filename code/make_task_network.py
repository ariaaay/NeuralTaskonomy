"""
This scripts computes the graphs of tasks based on prediction
on all voxels of the whole brain.
"""


import argparse
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from util.model_config import *
from visualize_corr_in_pycortex import load_data

# orders are made to match taskonomy
# taskrepr_dict = {
#     "taskrepr": [
#         "class_1000",
#         "segment25d",
#         "room_layout",
#         "rgb2sfnorm",
#         "rgb2depth",
#         "rgb2mist",
#         "reshade",
#         "keypoint3d",
#         "keypoint2d",
#         "autoencoder",
#         "colorization",
#         "edge3d",
#         "edge2d",
#         "denoise",
#         "curvature",
#         "class_places",
#         "vanishing_point",
#         "segmentsemantic",
#         "segment2d",
#
#     ]
# }

# overwrite config to shorten model name
task_label = {
    "class_1000": "Object Cl.",
    "segment25d": "2.5D Segm.",
    "room_layout": "Layout",
    "rgb2sfnorm": "Normals",
    "rgb2depth": "Depth",
    "rgb2mist": "Distance",
    "reshade": "Reshading",
    "keypoint3d": "3D Keypoint",
    "keypoint2d": "2D Keypoint",
    "autoencoder": "Autoenc.",
    "colorization": "Color",
    "edge3d": "Occulusion Edges",
    "edge2d": "2D Edges",
    "denoise": "Denoise",
    "curvature": "Curvature",
    "class_places": "Scene Cl.",
    "vanishing_point": "Vanishing Pts.",
    "segmentsemantic": "Semantic Segm.",
    "segment2d": "2D Segm.",
    "jigsaw": "Jigsaw",
    "inpainting_whole": "Inpainting",
}

node_label = [task_label[v] for v in task_label.keys()]


def get_voxels(model_list, subj, TR="_TRavg", ROI=False):
    datamat = list()
    for l in model_list:
        if not ROI:
            data = load_data(
                'taskrepr', task=l, subj=subj, TR=TR, measure="corr",
            )
        else:
            print("Not Implemented")
        datamat.append(data)
    datamat = np.array(datamat)
    return datamat


def get_sig_mask(model_list, correction, alpha, subj):
    maskmat = list()
    for l in model_list:
        mask = np.load(
            "../outputs/voxels_masks/subj{}/{}_{}_{}_{}_whole_brain.npy".format(
                subj, "taskrepr", l, correction, alpha
            )
        )
        maskmat.append(mask)
    maskmat = np.array(maskmat)
    return maskmat


def load_prediction(model, task, subj, TR="_TRavg", ROI=False):
    import pickle

    if ROI:
        print("Not implemented")
        return
    else:
        preds = pickle.load(
            open(
                "../outputs/encoding_results/subj{}/pred_whole_brain_{}_{}_{}.p".format(
                    subj, model, task, TR
                ),
                "rb",
            )
        )
        return preds[0]


def correlate_pairwise_prediction(model, feature_list, subj, TR="_TRavg"):
    save_path = "../outputs/pred_corr/task_pred_correlation_subj{}.npy".format(subj)
    try:
        corrmat = np.load(save_path)
    except FileNotFoundError:
        corrmat = np.ones((len(feature_list), len(feature_list)))
        for i in tqdm(range(len(feature_list) - 1)):
            pred1 = load_prediction(
                model, task=feature_list[i], subj=subj, TR=TR
            )
            for j in tqdm(range(i + 1, len(feature_list))):
                pred2 = load_prediction(
                    model, task=feature_list[j], subj=subj, TR=TR
                )
                corr = columnwise_avg_corr(pred1, pred2)

                corrmat[i][j] = corr
                corrmat[j][i] = corr
        np.save(save_path, corrmat)
    return corrmat


def columnwise_avg_corr(x, y):
    from scipy.stats import pearsonr

    assert x.shape == y.shape
    corr_list = list()
    for i in range(x.shape[1]):
        corr = pearsonr(x[:, i], y[:, i])
        corr_list.append(corr)

    assert len(corr_list) == x.shape[1]
    return np.mean(corr_list)


def compute_graph(voxel_mat, metric=None):
    if metric is None:
        dist = squareform(pdist(voxel_mat))
    else:
        dist = squareform(pdist(voxel_mat, metric))
    G = nx.from_numpy_matrix(dist)
    return G


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors

    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def scale_width(arr, alpha, beta):
    vmin = np.min(arr)
    vmax = np.max(arr)
    new_arr = (arr - vmin) / (vmax - vmin)
    return new_arr * alpha + beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="please specify subject to show")
    parser.add_argument(
        "--subj", type=str, default="1", help="specify which subject to build model on"
    )
    parser.add_argument(
        "--compute_overlap",
        default=False,
        action="store_true",
        help="compute graph by quantifying overlapping areas of regions",
    )
    parser.add_argument(
        "--use_prediction",
        default=False,
        action="store_true",
        help="compute graph based on correlation of predictions",
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
        help="use the masked correlation matrix",
    )
    parser.add_argument(
        "--ROI", default=False, action="store_true", help="use ROI data only"
    )

    parser.add_argument(
        "--empirical",
        default=False,
        action="store_true",
        help="use significant threshold computed from permutation test"
    )
    args = parser.parse_args()

    if args.empirical:
        p_method = "emp_fdr"
    else:
        p_method = "fdr"

    if args.compute_overlap:
        graph_name = "overlap"

        voxel_mat = get_voxels(list(task_label.keys()), subj=args.subj)
        good_voxel_mat = voxel_mat > 0.1
        G = compute_graph(
            good_voxel_mat.astype(int), metric=lambda u, v: u @ v
        )  # here dist is similarity
        widths = [G[u][v]["weight"] for u, v in G.edges()]
        widths = scale_width(widths, alpha=10, beta=1)

    elif args.use_prediction:
        graph_name = "prediction"

        corr_mat = correlate_pairwise_prediction(
            "taskrepr", list(task_label.keys()), args.subj
        )
        G = nx.from_numpy_matrix(corr_mat)
        widths = [G[u][v]["weight"] for u, v in G.edges()]
        widths = scale_width(widths, alpha=10, beta=1)

    elif args.use_significance:
        graph_name = "sig"
        sig_mat = get_sig_mask(
            list(task_label.keys()), correction=p_method, alpha=0.05, subj=args.subj
        )
        np.save("../outputs/task_matrix/sig_mask_subj{}_{}.npy".format(args.subj, p_method), sig_mat)
        G = compute_graph(sig_mat.astype(int), metric=lambda u, v: u @ v)
        widths = [G[u][v]["weight"] for u, v in G.edges()]
        widths = scale_width(widths, alpha=10, beta=2)

    elif args.use_mask_corr:
        graph_name = "masked_corr"
        voxel_mat = get_voxels(list(task_label.keys()), subj=args.subj)
        sig_mat = get_sig_mask(
            list(task_label.keys()), correction=p_method, alpha=0.05, subj=args.subj
        )
        assert voxel_mat.shape == sig_mat.shape

        voxel_mat[~sig_mat] = 0
        print(voxel_mat.shape)
        np.save(
            "../outputs/task_matrix/mask_corr_subj{}_{}.npy".format(args.subj, p_method), voxel_mat
        )

        corr_mat = cosine_similarity(voxel_mat, dense_output=False)
        # G = compute_graph(voxel_mat, metric=lambda u, v: u @ v)
        G = nx.from_numpy_matrix(corr_mat)

        widths = [G[u][v]["weight"] for u, v in G.edges()]
        widths = scale_width(widths, alpha=10, beta=1)

    else:
        graph_name = "corr_of_accuracy"

        voxel_mat = get_voxels(list(task_label.keys()), subj=args.subj)

        G = compute_graph(voxel_mat)
        widths = [1 / G[u][v]["weight"] for u, v in G.edges()]
        wrange = max(widths) - min(widths)
        widths = [(w - min(widths)) / (wrange) * 10 for w in widths]

    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), node_label)))
    G = nx.drawing.nx_agraph.to_agraph(G)
    pos = nx.circular_layout(G)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nx.draw(
        G,
        pos,
        node_color="silver",
        node_size=1500,
        font_size=30,
        font_weight="bold",
        width=widths,
        # edge_color="#A0CBE2",
        edge_color=np.array(widths),
        edge_cmap=truncate_colormap(plt.cm.YlGnBu, 0.3, 0.8),
        with_labels=True,
        ax=ax,
    )

    # G.draw("../figures/task_graph.png", pos=pos, format="png", prog="neato")
    outpath = "../figures/taskmap/taskmap_subj{}_{}.pdf".format(args.subj, graph_name)
    fig.set_size_inches(21, 21)
    # plt.subplots_adjust(left=1)

    fig.tight_layout()
    fig.savefig(outpath)
