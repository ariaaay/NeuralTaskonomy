"This scripts visualize prediction performance with Pycortex."

import pickle
from glob import glob
import numpy as np
# from save_3d_views import *
import argparse

def load_data(
        model,
        layer="",
        task="",
        dataset="",
        subj=1,
        TR="_TRavg",
        measure="corr",
):
    if task is not "":
        model = model + "_" + task
    if measure == "corr":
        corrs = pickle.load(
            open(
                "../outputs/encoding_results/subj{}/corr_whole_brain_{}_{}{}{}.p".format(
                    subj, model, layer, dataset, TR
                ),
                "rb",
            )
        )
        r = np.array(corrs)
        r = r[:, 0]
        # print("{} maximum is: {}".format(layer, max(corrs)))
        # print("{} minimum is: {}".format(layer, min(corrs)))
    else:
        rsqs = pickle.load(
            open(
                "../outputs/encoding_results/subj{}/rsq_whole_brain_{}_{}{}{}.p".format(
                    subj, model, layer, dataset, TR
                ),
                "rb",
            )
        )
        rsqs = np.array(rsqs)
        # print("{} maximum is: {}".format(layer, max(rsqs)))
        # print("{} minimum is: {}".format(layer, min(rsqs)))
        rsqs[rsqs < 0] = 0
        r = np.sqrt(rsqs)
        # r = rsqs
    return r


def make_volume(subj,
                model, layer="", task="", dataset="", TR="_TRavg", mask_with_significance=False
                ):
    import cortex
    mask = cortex.utils.get_cortical_mask("sub-CSI{}".format(subj), "full")
    vals = load_data(
        model,
        layer=layer,
        task=task,
        dataset=dataset,
        subj=subj,
        TR=TR,
    )

    if mask_with_significance:
        correction = "emp_fdr"
        alpha = 0.05
        sig_mask = np.load(
            "../outputs/voxels_masks/subj{}/{}_{}{}_{}_{}_whole_brain.npy".format(subj, model, task, layer, correction,
                                                                                  alpha))
        print(sig_mask.shape)
        print(np.sum(sig_mask))
        if np.sum(sig_mask) > 0:
            mask[mask == True] = sig_mask
            vals = vals[sig_mask]

    vol_data = cortex.Volume(
        vals, "sub-CSI{}".format(subj), "full", mask=mask, cmap="hot", vmin=0.05, vmax=0.2
    )

    return vol_data


def model_selection(subj, model_dict, TR="_TRavg"):
    import cortex
    datamat = list()
    for m in model_dict.keys():
        if model_dict[m] is not None:
            for l in model_dict[m]:
                data = load_data(
                    m, task=l, subj=subj, TR=TR, measure="corr"
                )
                datamat.append(data)
    datamat = np.array(datamat)
    threshold_mask = np.max(datamat, axis=0) > 0.13
    best_model = np.argmax(datamat, axis=0)[threshold_mask]
    mask = cortex.utils.get_cortical_mask("sub-CSI{}".format(subj), "full")
    mask[mask == True] = threshold_mask

    vol_data = cortex.Volume(
        best_model, "sub-CSI{}".format(subj), "full", mask=mask, cmap="nipy_spectral"
    )
    return vol_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="please specific subject to show")
    parser.add_argument(
        "--subj", type=str, default="1", help="specify which subject to build model on"
    )
    parser.add_argument("--mask_sig", default=False, action="store_true")
    parser.add_argument("--make_viewer", default=False, action="store_true")
    args = parser.parse_args()

    subjport = int("1111{}".format(args.subj))

    taskrepr_dict = {
        "taskrepr": [
            "curvature",
            "edge2d",
            "edge3d",
            "keypoint2d",
            "keypoint3d",
            "rgb2depth",
            "reshade",
            "rgb2mist",
            "rgb2sfnorm",
            "class_1000",
            "class_places",
            "autoencoder",
            "denoise",
            "segment25d",
            "segment2d",
            "segmentsemantic",
            "vanishing_point",
            "room_layout",
            "colorization"
        ]
    }

    volumes = {
        "Curvature": make_volume(subj=args.subj, model="taskrepr", task="curvature",
                                 mask_with_significance=args.mask_sig),
        "2D Edges": make_volume(subj=args.subj, model="taskrepr", task="edge2d", mask_with_significance=args.mask_sig),
        "3D Edges": make_volume(subj=args.subj, model="taskrepr", task="edge3d", mask_with_significance=args.mask_sig),
        "2D Keypoint": make_volume(subj=args.subj, model="taskrepr", task="keypoint2d",
                                   mask_with_significance=args.mask_sig),
        "3D Keypoint": make_volume(subj=args.subj, model="taskrepr", task="keypoint3d",
                                   mask_with_significance=args.mask_sig),
        "Depth": make_volume(subj=args.subj, model="taskrepr", task="rgb2depth", mask_with_significance=args.mask_sig),
        "Reshade": make_volume(subj=args.subj, model="taskrepr", task="reshade", mask_with_significance=args.mask_sig),
        "Distance": make_volume(subj=args.subj, model="taskrepr", task="rgb2mist",
                                mask_with_significance=args.mask_sig),
        "Surface Normal": make_volume(subj=args.subj, model="taskrepr", task="rgb2sfnorm",
                                      mask_with_significance=args.mask_sig),
        "Object Class": make_volume(subj=args.subj, model="taskrepr", task="class_1000",
                                    mask_with_significance=args.mask_sig),
        "Scene Class": make_volume(subj=args.subj, model="taskrepr", task="class_places",
                                   mask_with_significance=args.mask_sig),
        "Autoencoder": make_volume(subj=args.subj, model="taskrepr", task="autoencoder",
                                   mask_with_significance=args.mask_sig),
        "Denoising": make_volume(subj=args.subj, model="taskrepr", task="denoise",
                                 mask_with_significance=args.mask_sig),
        "2.5D Segm.": make_volume(subj=args.subj, model="taskrepr", task="segment25d",
                                  mask_with_significance=args.mask_sig),
        "2D Segm.": make_volume(subj=args.subj, model="taskrepr", task="segment2d",
                                mask_with_significance=args.mask_sig),
        "Semantic Segm": make_volume(subj=args.subj, model="taskrepr", task="segmentsemantic",
                                     mask_with_significance=args.mask_sig),
        "Vanishing Point": make_volume(subj=args.subj, model="taskrepr", task="vanishing_point",
                                       mask_with_significance=args.mask_sig),
        "Room Layout": make_volume(subj=args.subj, model="taskrepr", task="room_layout",
                                   mask_with_significance=args.mask_sig),
        "Color": make_volume(subj=args.subj, model="taskrepr", task="colorization",
                             mask_with_significance=args.mask_sig),
        "Inpainting Whole": make_volume(subj=args.subj, model="taskrepr", task="inpainting_whole",
                                        mask_with_significance=args.mask_sig),
        "Jigsaw": make_volume(subj=args.subj, model="taskrepr", task="jigsaw", mask_with_significance=args.mask_sig),
        # "Taskrepr model comparison": model_selection(subj=args.subj, model_dict=taskrepr_dict),

    }
    import cortex

    if args.make_viewer:
        viewer_path = '../website/brain_viewer/src/subj{}'.format(args.subj)
        cortex.webgl.make_static(outpath=viewer_path, data=volumes, template = '../website/brain_viewer/template.html', recache=True)

    else:
        cortex.webgl.show(data=volumes, autoclose=False, port=subjport)

    import pdb

    pdb.set_trace()
