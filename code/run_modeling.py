"""
This scripts takes an features space and runs encoding models (ridge regression) to
predict brain data.
"""

from scipy.io import loadmat
import argparse
import torch
import pickle
from glob import glob
from encodingmodel.encoding_model import fit_encoding_model, permutation_test
from featureprep.feature_prep import get_features
# from data_consistency import *

from util.util import *

def run(
        fm,
        br,
        model_name,
        test,
        notest,
        whole_brain,
        br_subset_idx,
        stacking,
        split_by_runs,
        pca,
        fix_testing,
        cv,
        model_list,
):
    print("Features are {}. Using whole brain data: {}".format(model_name, whole_brain))

    if not test:
        print("Fitting Encoding Models")
        fit_encoding_model(
            fm,
            br,
            model_name=model_name,
            byROI=not whole_brain,
            subset_idx=br_subset_idx,
            subj=args.subj,
            split_by_runs=split_by_runs,
            pca=pca,
            stacking=stacking,
            fix_testing=fix_testing,
            cv=cv,
            saving=True,
            model_list=model_list
        )
    if not notest:
        print("Running Permutation Test")
        permutation_test(
            fm,
            br,
            model_name=model_name,
            byROI=not whole_brain,
            subset_idx=br_subset_idx,
            subj=args.subj,
            split_by_runs=split_by_runs,
            permute_y=args.permute_y,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="please specify features to model from"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        help="input the name of the features."
             "Options are: convnet, scenenet, response,surface_normal_latent, surface_normal_subsample, RT, etc",
    )
    parser.add_argument(
        "--layer", type=str, default="", help="input name of the convolutional layer"
    )
    parser.add_argument(
        "--notest", action="store_true", help="run encoding model with permutation test"
    )
    parser.add_argument(
        "--test", action="store_true", help="running permutation testing only"
    )
    parser.add_argument(
        "--whole_brain", action="store_true", help="use whole brain data for modeling"
    )
    parser.add_argument(
        "--subj", type=str, default="1", help="specify which subject to build model on"
    )
    parser.add_argument(
        "--TR", type=str, default="avg", help="specify which TR to build model on"
    )
    parser.add_argument(
        "--dim", type=str, default="", help="specify the dimension of the pic2vec model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="specify which brain response you to build model on.",
    )
    parser.add_argument(
        "--stacking",
        type=str,
        default=None,
        help="run stacking net to select features in the joint model.",
    )
    parser.add_argument(
        "--split_by_runs",
        action="store_true",
        help="split the training and testing samples by runs.",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="run pca on features; has to applied on per datasets training.",
    )
    parser.add_argument(
        "--indoor_only", action="store_true", help="run models only on indoor images."
    )
    parser.add_argument(
        "--fix_testing",
        action="store_true",
        help="used fixed sampling for training and testing",
    )
    parser.add_argument(
        "--cv", action="store_true", default=False, help="run cross-validation"
    )
    parser.add_argument(
        "--permute_y",
        action="store_true",
        default=False,
        help="permute test label but not training label",
    )
    args = parser.parse_args()


    # Load brain data
    if args.whole_brain:
        # Load whole brain data (the one that is zscored across runs)
        if args.TR == "avg":
            br_data_tr3 = np.load(
                "{}/CSI{}_TR{}_zscore.npy".format(cortical_dir, args.subj, 3)
            )
            br_data_tr4 = np.load(
                "{}/CSI{}_TR{}_zscore.npy".format(cortical_dir, args.subj, 4)
            )
            br_data = (br_data_tr3 + br_data_tr4) / 2
        else:
            br_data = np.load(
                "{}/CSI{}_TR{}_zscore.npy".format(cortical_dir, args.subj, args.TR)
            )

        print("Brain response size is: " + str(br_data.shape))

    else:
        # Load ROI data
        if args.TR == "avg":
            br_data = [
                loadmat("../BOLD5000/CSI{}_ROIs_TR{}.mat".format(args.subj, tr))
                for tr in [3, 4]
            ]

        elif args.TR == "avg234":
            br_data = [
                loadmat("../BOLD5000/CSI{}_ROIs_TR{}.mat".format(args.subj, tr))
                for tr in [2, 3, 4]
            ]

        else:
            br_data = loadmat(
                "../BOLD5000/CSI{}_ROIs_TR{}.mat".format(args.subj, args.TR)
            )

    # Load feature spaces
    if args.stacking is not None:
        model_list = args.stacking.split(" ")
        # model_list = [
        #     ["convnet", "fc7"],
        #     ["convnet", "conv4"],
        #     ["scenenet", "fc"],
        #     ["surface_normal_subsample"],
        # ]  # no support for different brain response to datasets yet; all br_subset_idx should be the non-repetitive trials
        feature_mat = []
        model_name_to_save = "stacking " + args.stacking + "_TR" + args.TR
        print("Running stacking on: " )
        print(model_list)
        for model in model_list:
            print(model)
            if ('_' in model) and ('taskrepr' not in model):
                m = model.split("_")
                
                ftm, br_subset_idx = get_features(
                    args.subj, m[0], layer=m[1],
                )
            else:
                ftm, br_subset_idx = get_features(
                    args.subj, model,
                )
            feature_mat.append(ftm)
            stacking = True
            # TODO: assert all subset idx are the same for all features
    else:
        stacking = False
        model_list = None
        print("Running ridge on " + args.model)
        feature_mat, br_subset_idx = get_features(
            args.subj,
            args.model,
            layer=args.layer,
            dim=args.dim,
            dataset=args.dataset,
        )

        # no need to subset trials if it is for RT, or valence response, i.e. non image related features

        if args.split_by_runs:
            split_in_name = "_by_runs"
        else:
            split_in_name = ""

        if args.pca:
            pca_in_name = "_pca"
        else:
            pca_in_name = ""

        if args.indoor_only:
            indoor_in_name = "indoor"
        else:
            indoor_in_name = ""

        # if args.fix_testing:
        #     fix_in_name = "_fix"
        # else:
        #     fix_in_name = ""

        model_name_to_save = (
                args.model
                + "_"
                + args.layer
                + args.dim
                + args.dataset
                + "_TR"
                + args.TR
                + split_in_name
                + pca_in_name
                + indoor_in_name
        )

    run(
        feature_mat,
        br_data,
        model_name=model_name_to_save,
        test=args.test,
        notest=args.notest,
        whole_brain=args.whole_brain,
        br_subset_idx=br_subset_idx,
        stacking=stacking,
        split_by_runs=args.split_by_runs,
        pca=args.pca,
        fix_testing=args.fix_testing,
        cv=args.cv,
        model_list=model_list,
    )
