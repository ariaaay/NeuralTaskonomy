"""
This scripts make plots of the results from encoding models on ROI data.
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from util import *
import os
from glob import glob
from util.model_config import *

# from encoding_model import *
import seaborn as sns
import pandas as pd
import argparse
from glob import glob

sns.set(style="whitegrid", font_scale=2.2)
# sns.set_style("white")


def load_df(dataframe, inpath, model, layers=None, subset=None, saving=True):
    if layers is None:
        layers = ["_"]
    if subset is None:
        subset = ""
    for l in layers:
        corrs = pickle.load(
            open(
                glob("{}/corr_{}_{}*TRavg{}.p".format(inpath, model, l, subset))[0],
                "rb",
            )
        )
        rsqs = pickle.load(
            open(
                glob("{}/rsq_{}_{}*TRavg{}.p".format(inpath, model, l, subset))[0], "rb"
            )
        )

        for i, corr in enumerate(corrs):
            for j, r in enumerate(corr):
                # print(r.shape)
                vd = dict()
                vd["correlation"] = r[0]
                vd["rsquare"] = rsqs[i][j]
                vd["ROI"] = ROI_labels[i][2:]
                vd["hemisphere"] = ROI_labels[i][0:2]
                vd["features"] = task_label[l]
                vd["model"] = model
                if subset is not "":
                    vd["subset"] = subset
                dataframe = dataframe.append(vd, ignore_index=True)
    if saving:
        dataframe.to_csv("{}/encoding_df.csv".format(inpath))
    return dataframe


def plot_point_and_violin(
    dataframe, outpath, model, y, xlab, ylab, subset="", saving=True
):
    horder = ["EarlyVis", "OPA", "LOC", "PPA", "RSC"]
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 1, 1)
    sns.pointplot(
        x="features",
        y=y,
        hue="ROI",
        data=dataframe[dataframe.model == model],
        scale=0.8,
        hue_order=horder,
        ax=ax,
        dodge=True,
    )
    ax1 = fig.add_subplot(2, 1, 2)
    sns.violinplot(
        x="features",
        y=y,
        hue="ROI",
        data=dataframe[dataframe.model == model],
        scale="count",
        hue_order=horder,
        linewidth=1.2,
        saturation=0.8,
        ax=ax1,
    )
    sns.lineplot(
        x=(-0.5, 6.5),
        y=0,
        data=dataframe[dataframe.model == model],
        ax=ax1,
        color="black",
        linewidth=0.6,
    )  # zero line
    ax.set(xlabel=xlab, ylabel=ylab)
    ax1.set(xlabel=xlab, ylabel=ylab)
    ax1.set_ylim(-0.2, 0.5)
    sns.despine(left=True)
    if saving:
        fig.savefig(outpath + "{}_encoding_{}{}.png".format(model, y, subset), dpi=600)


def plot_point(dataframe, outpath, model, y, xlab, ylab, subset="", saving=True):
    horder = ["EarlyVis", "OPA", "LOC", "PPA", "RSC"]
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)
    sns.pointplot(
        x="features",
        y=y,
        hue="ROI",
        data=dataframe[dataframe.model == model],
        scale=0.8,
        hue_order=horder,
        ax=ax,
        dodge=True,
    )
    ax.set(xlabel=xlab, ylabel=ylab)


    sns.despine(left=True)
    if saving:
        fig.savefig(
            outpath + "{}_encoding_{}{}_point_plot.png".format(model, y, subset),
            dpi=600,
        )


def plot_bar(dataframe, outpath, model, horders, subset="", saving=True):
    fig = plt.figure(figsize=(40, 9))
    # import pdb; pdb.set_trace()
    ax = fig.add_subplot(1, 1, 1)
    sns.barplot(
        x="ROI",
        y="correlation",
        hue="features",
        data=dataframe[dataframe.model == model],
        hue_order=horders,
        ax=ax,
        palette=sns.color_palette("colorblind"),
    )
    plt.legend(bbox_to_anchor=(0.5, 1), loc="upper center", borderaxespad=0.0, ncol=7, frameon=False)
    sns.despine(left=True, fig=fig, ax=ax)
    # ax.legend(bbox_to_anchor=(1.01, 1.04), loc=2, borderaxespad=0.0, title="Tasks")
    ax.set(xlabel="", ylabel="Correlation (r)")

    # import pdb; pdb.set_trace()

    for i, p in enumerate(ax.lines):
        if i in {15, 16, 17, 18, 83}:
            continue
        anx = p.get_xdata()[0]
        any = p.get_ydata()[1] + 0.001
        ax.annotate("*", xy=(anx, any), xycoords="data", ha="center")

    # ax1 = fig.add_subplot(2, 1, 2)
    # sns.barplot(
    #     x="ROI",
    #     y="rsquare",
    #     hue="features",
    #     data=dataframe[dataframe.model == model],
    #     hue_order=horders,
    #     ax=ax1,
    # )
    # ax1.get_legend().set_visible(False)
    # ax1.set(xlabel="ROIs", ylabel="R-square")

    plt.tight_layout()
    if saving:
        fig.savefig(
            outpath + "{}_encoding_bar_corr{}.png".format(model, subset), dpi=500
        )


def plot_single_layer(dataframe, outpath, model, subset="", saving=True):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 1, 1)
    sns.lineplot(
        x="ROI", y="correlation", data=dataframe[dataframe.model == model], ax=ax
    )
    ax1 = fig.add_subplot(2, 1, 2)
    sns.despine(left=True)
    sns.lineplot(x="ROI", y="rsquare", data=dataframe[dataframe.model == model], ax=ax1)
    if saving:
        fig.savefig(outpath + "{}_encoding_bar{}.png".format(model, subset), dpi=500)


# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please specific subjects/features to plot from"
    )
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--comparison", action="store_true", default=False)
    parser.add_argument("--reload", action="store_true", default=False)
    # parser.add_argument('--models', type=str, default='convnet')
    parser.add_argument("--plot_only", action="store_true", default=False)
    parser.add_argument("--point_plot", action="store_true", default=False)
    parser.add_argument("--bar_plot", action="store_true", default=False)
    parser.add_argument("--point_and_violin_plot", action="store_true", default=False)
    parser.add_argument("--no_saving", action="store_true", default=False)
    parser.add_argument("--plot_all_subjects", action="store_true", default=False)

    args = parser.parse_args()

    inpath = "../outputs/encoding_results/subj{}".format(args.subj)
    outpath = "../figures/encoding_models/subj{}/".format(args.subj)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    if not True in [args.reload, args.plot_only]:
        try:
            df = pd.read_csv(inpath + "/encoding_df.csv")
        except FileNotFoundError:
            print("No dataframe found, computing dataframe...")
            df = pd.DataFrame(
                columns=[
                    "model",
                    "correlation",
                    "rsqaure",
                    "ROI",
                    "hemisphere",
                    "features",
                ]
            )
    else:
        print("Making new dataframe...")
        df = pd.DataFrame(
            columns=["model", "correlation", "rsqaure", "ROI", "hemisphere", "features"]
        )

    ## load dataframes
    if args.no_saving:
        save = False
    else:
        save = True

    # if "convnet" not in df.model.values:
    #     print("Loading encoding model for ImageNet...")
    #     df = load_df(df, inpath, model="convnet", layers=conv_layers, saving=save)

    # if "scenenet" not in df.model.values:
    #     print("Loading encoding model for SceneNet...")
    #     df = load_df(df, inpath, model="scenenet", layers=scenenet_layers, saving=save)
    #
    # if "surface_normal" not in df.model.values:
    #     print("Loading encoding model for surface normal...")
    #     load_df(df, inpath, model="surface_normal", layers=sf_methods, saving=save)

    if "taskrepr" not in df.model.values:
        print("Loading encoding model for task representation...")
        df = load_df(
            df, inpath, model="taskrepr", layers=taskrepr_features, saving=save
        )
    else:
        for feature in taskrepr_features:
            if task_label[feature] not in df.features.values:
                print("Loading encoding model for task {}".format(feature))
                df = load_df(
                    df, inpath, model="taskrepr", layers=[feature], saving=save
                )

    # if "response" not in df.model.values:
    #     print("Loading encoding model with valence ratings...")
    #     df = load_df(df, inpath, model="response", saving=save)
    #
    # if "RT" not in df.model.values:
    #     print("Loading encoding model with reaction time...")
    #     df = load_df(df, inpath, model="RT", saving=save)

    ## =============================================
    ## Point and volin plot
    ## =============================================
    if args.point_and_violin_plot:
        plot_point_and_violin(
            df,
            outpath,
            model="convnet",
            y="correlation",
            xlab="ImageNet Layers",
            ylab="Correlations (r)",
            saving=save,
        )
        plot_point_and_violin(
            df,
            outpath,
            model="convnet",
            y="rsquare",
            xlab="ImageNet Layers",
            ylab="R-square",
        )
        plot_point_and_violin(
            df,
            outpath,
            model="scenenet",
            y="correlation",
            xlab="SceneNet Layers",
            ylab="Correlations (r)",
        )
        plot_point_and_violin(
            df,
            outpath,
            model="scenenet",
            y="rsquare",
            xlab="SceneNet Layers",
            ylab="R-square",
        )
    ## =============================================
    ## bar plot
    ## =============================================
    if args.bar_plot:
        torder = [task_label[f] for f in taskrepr_features]
        # plot_bar(df, outpath, model='surface_normal', horders=methods)
        plot_bar(df, outpath, model="taskrepr", horders=torder)
        # plot_single_layer(df, outpath, model="response")
        # plot_single_layer(df, outpath, model="RT")

    ## =============================================
    ## point plot
    ## =============================================
    if args.point_plot:
        plot_point(
            df,
            outpath,
            model="convnet",
            y="correlation",
            xlab="ImageNet Layers",
            ylab="Correlations (r)",
        )

        plot_point(
            df,
            outpath,
            model="convnet",
            y="rsquare",
            xlab="ImageNet Layers",
            ylab="R-square",
        )

        plot_point(
            df,
            outpath,
            model="scenenet",
            y="correlation",
            xlab="SceneNet Layers",
            ylab="Correlations (r)",
        )

        plot_point(
            df,
            outpath,
            model="scenenet",
            y="rsquare",
            xlab="SceneNet Layers",
            ylab="R-square",
        )

    #
    # df_byrun = load_df(
    #     df, "convnet", conv_layers[:5], "_by_runs", saving=not args.plot_only
    # )
    # plot_point_and_violin(
    #     df_byrun[df_byrun.subset == "_by_run"],
    #     model="convnet",
    #     y="correlation",
    #     xlab="Convnet Layers",
    #     ylab="Correlations",
    #     subset="_by_run",
    # )

    # plt.show()

    # average of TR 2,3,4
    # if "convnet" not in df.model.values:
    #     print("Loading encoding model for ImageNet...")
    #     df = load_df(df, "convnet", conv_layers, saving=save)
    # plot_point_and_violin(
    #     df,
    #     model="convnet",
    #     y="correlation",
    #     xlab="ImageNet Layers",
    #     ylab="Correlations (r)",
    #     saving=save,
    # )
    if args.no_saving:
        plt.show()

    # ================== comparison ========================
    if args.comparison:
        palette = sns.color_palette("colorblind", 11)
        palette += sns.color_palette("husl", 7)

        # fig = plt.figure(figsize=(20, 10))
        fig = plt.figure(figsize=(20, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        sns.barplot(
            x="ROI",
            y="correlation",
            hue="features",
            data=df[df.model.isin(["convnet", "taskrepr"])],
            palette=palette,
            hue_order=taskrepr_features + conv_layers,
            ax=ax1,
        )
        sns.despine(left=True)
        ax1.set(xlabel="ROIs", ylabel="Correlations (r)")
        plt.legend(bbox_to_anchor=(1.01, 1.1), loc=2, borderaxespad=0.0, title="Tasks")

        # plt.tight_layout()
        fig.savefig(outpath + "tasks_comparison_bar.png", dpi=500)

    if args.plot_all_subjects:
        # assume all df are created
        outpath = "../figures/encoding_models/all/"
        if not os.path.isdir(outpath):
            os.makedirs(outpath)

        df_list = []
        for i in range(3):
            try:
                inpath = "../outputs/encoding_results/subj{}".format(i + 1)
                df = pd.read_csv(inpath + "/encoding_df.csv")
            except FileNotFoundError:
                "Subject {} not found. Please load dataframe first".format(i + i)

            df["Subject"] = str(i + 1)
            df_list.append(df)
        df_all = pd.concat(df_list)
        df_all.to_csv(
            "../outputs/encoding_results/all/encoding_df.csv"
        )

        grid = sns.FacetGrid(
            df_all[df_all.model == "taskrepr"],
            row="Subject",
            row_order=["1", "2", "3"],
            # legend_out=True,
            despine=True,
            height=3,
            aspect=5,
            dropna=False,
        )

        def _bar(x, y, **kwargs):
            data = kwargs.pop("data")
            ax = plt.gca()

            sns.barplot(x, y, data=data, ax=ax, **kwargs)

        torder = [task_label[f] for f in taskrepr_features]

        grid.map_dataframe(
            _bar,
            "ROI",
            "correlation",
            hue="features",
            hue_order=torder,
            palette=sns.color_palette("colorblind", len(torder)),
        )
        # grid.add_legend(title="Tasks", ncol=10, loc='upper center')

        grid.set_axis_labels("ROI", "Correlations (r)")

        handles = grid._legend_data.values()
        labels = grid._legend_data.keys()
        grid.fig.legend(
            title="Tasks",
            handles=handles,
            labels=labels,
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.49, 0.95),
            frameon=False,
        )
        ax = grid.axes
        sns.despine(fig=grid.fig, ax=ax, left=True, bottom=True)

        grid.savefig(outpath + "tasks_corr.pdf")
