"""
This scripts estimates the consistency of predictions by computing the
correlations of prediction performances across subjects.
"""

from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import argparse

from util.model_config import  *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    args = parser.parse_args()
    tasks = [task_label[t] for t in taskrepr_features]
    df = pd.read_csv("../outputs/encoding_results/all/encoding_df.csv")
    subj_corr = list()
    for i in np.arange(2)+1:
        subi_df = df[df['Subject'] == i]
        for j in np.arange(i,3)+1:
            corri = list()
            corrj = list()
            subj_df = df[df['Subject'] == j]
            for roi in ROIS:
                for task in tasks:
                    di = subi_df[(subi_df.ROI == roi) & (subi_df.features == task)]['correlation']
                    dj = subj_df[(subj_df.ROI == roi) & (subj_df.features == task)]['correlation']
                    # import pdb; pdb.set_trace()

                    corri.append(np.mean(di))
                    corrj.append(np.mean(dj))
            subj_corr = pearsonr(corri, corrj)
            print("Computing correlation between subject {} and {}: {}".format(i, j, subj_corr[0]))
            # print(corri)
            # print(corrj)