# -*- coding: utf-8 -*-
'''
This is a series of custom functions for the inferring of GRN from single cell RNA-seq data.

Codes were written by Kenji Kamimoto.


'''

###########################
### 0. Import libralies ###
###########################


# 0.1. libraries for fundamental data science and data processing

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm.auto import tqdm



settings = {"save_figure_as": "png"}

def plot_scores_as_rank(links, cluster, n_gene=50, save=None):
    """
    Pick up top n-th genes wich high-network scores and make plots.

    Args:
        links (Links object): See network_analisis.Links class for detail.
        cluster (str): Cluster nome to analyze.
        n_gene (int): Number of genes to plot. Default is 50.
        save (str): Folder path to save plots. If the folde does not exist in the path, the function create the folder.
            If None plots will not be saved. Default is None.
    """
    values = ['degree_centrality_all',
                  'degree_centrality_in', 'degree_centrality_out',
                  'betweenness_centrality',  'eigenvector_centrality']
    for value in values:

        res = links.merged_score[links.merged_score.cluster == cluster]
        res = res[value].sort_values(ascending=False)
        res = res[:n_gene]

        fig = plt.figure()

        plt.scatter(res.values, range(len(res)))
        plt.yticks(range(len(res)), res.index.values)#, rotation=90)
        plt.xlabel(value)
        plt.title(f" {value} \n top {n_gene} in {cluster}")
        plt.gca().invert_yaxis()
        plt.subplots_adjust(left=0.5, right=0.85)

        if not save is None:
            os.makedirs(save, exist_ok=True)
            path = os.path.join(save, f"ranked_values_in_{links.name}_{value}_{links.threshold_number}_in_{cluster}.{settings['save_figure_as']}")
            fig.savefig(path, transparent=True)
        plt.show()


def _plot_goi(x, y, goi, args_annot, scatter=False, x_shift=0.1, y_shift=0.1):
    """
    Plot an annoation to highlight one point in scatter plot.

    Args:
        x (float): Cordinate-x.
        y (float): Cordinate-y.
        args_annot (dictionary): arguments for matplotlib.pyplot.annotate().
        scatter (bool): Whether to plot dot for the point of interest.
        x_shift (float): distance between the annotation and the point of interest in the x-axis.
        y_shift (float): distance between the annotation and the point of interest in the y-axis.
    """

    default = {"size": 10}
    default.update(args_annot)
    args_annot = default.copy()

    arrow_dict = {"width": 0.5, "headwidth": 0.5, "headlength": 1, "color": "black"}
    if scatter:
        plt.scatter(x, y, c="none", edgecolor="black")
    plt.annotate(f"{goi}", xy=(x, y), xytext=(x+x_shift, y+y_shift),
                 color="black", arrowprops=arrow_dict, **args_annot)



def plot_score_comparison_2D(links, value, cluster1, cluster2, percentile=99, annot_shifts=None, save=None, fillna_with_zero=True, plt_show=True):
    """
    Make a scatter plot that shows the relationship of a specific network score in two groups.

    Args:
        links (Links object): See network_analisis.Links class for detail.
        value (srt): The network score to be shown.
        cluster1 (str): Cluster nome to analyze. Network scores in the cluste1 are shown as x-axis.
        cluster2 (str): Cluster nome to analyze. Network scores in the cluste2 are shown as y-axis.
        percentile (float): Genes with a network score above the percentile will be shown with annotation. Default is 99.
        annot_shifts ((float, float)): Shift x and y cordinate for annotations.
        save (str): Folder path to save plots. If the folde does not exist in the path, the function create the folder.
            If None plots will not be saved. Default is None.
    """
    res = links.merged_score[links.merged_score.cluster.isin([cluster1, cluster2])][[value, "cluster"]]
    res = res.reset_index(drop=False)
    piv = pd.pivot_table(res, values=value, columns="cluster", index="index")
    if fillna_with_zero:
        piv = piv.fillna(0)
    else:
        piv = piv.fillna(piv.mean(axis=0))

    goi1 = piv[piv[cluster1] > np.percentile(piv[cluster1].values, percentile)].index
    goi2 = piv[piv[cluster2] > np.percentile(piv[cluster2].values, percentile)].index

    gois = np.union1d(goi1, goi2)

    x, y = piv[cluster1], piv[cluster2]
    plt.scatter(x, y, c="none", edgecolor="black")

    if annot_shifts is None:
        x_shift, y_shift = (x.max() - x.min())*0.03, (y.max() - y.min())*0.03
    else:
        x_shift, y_shift = annot_shifts
    for goi in gois:
        x, y = piv.loc[goi, cluster1], piv.loc[goi, cluster2]
        _plot_goi(x, y, goi, {}, scatter=False, x_shift=x_shift, y_shift=y_shift)

    plt.xlabel(cluster1)
    plt.ylabel(cluster2)
    plt.title(f"{value}")
    if not save is None:
        os.makedirs(save, exist_ok=True)
        path = os.path.join(save, f"values_comparison_in_{links.name}_{value}_{links.threshold_number}_{cluster1}_vs_{cluster2}.{settings['save_figure_as']}")
        plt.savefig(path, transparent=True)
    if plt_show:
        plt.show()






def plot_score_comparison_2D_with_plotly(links, value, cluster1, cluster2, fillna_with_zero=True):
    """
    Make a scatter plot that shows the relationship of a specific network score in two groups.

    Args:
        links (Links object): See network_analisis.Links class for detail.
        value (srt): The network score to be shown.
        cluster1 (str): Cluster nome to analyze. Network scores in the cluste1 are shown as x-axis.
        cluster2 (str): Cluster nome to analyze. Network scores in the cluste2 are shown as y-axis.

    """

    try:
        import plotly.express as px
        res = links.merged_score[links.merged_score.cluster.isin([cluster1, cluster2])][[value, "cluster"]]
        res = res.reset_index(drop=False)
        piv = pd.pivot_table(res, values=value, columns="cluster", index="index")
        piv = piv.reset_index(drop=False)

        if fillna_with_zero:
            piv = piv.fillna(0)
        else:
            piv = piv.fillna(piv.mean(axis=0))


        x, y = piv[cluster1], piv[cluster2]

        fig = px.scatter(piv, x=cluster1, y=cluster2,
                         hover_data=['index'], template="plotly_white")

        return fig
    except:
        print("Interactive mode requires plotly. Please install plotly before use.: pip install plotly")




######################
### score dynamics ###
######################
def plot_score_per_cluster(links, goi, save=None, plt_show=True):
    """
    Plot network score for a specific gene.
    This function can be used to compare network score of a specific gene between clusters
    and get insight about the dynamics of the gene.

    Args:
        links (Links object): See network_analisis.Links class for detail.
        goi (srt): Gene name.
        save (str): Folder path to save plots. If the folde does not exist in the path, the function create the folder.
            If None plots will not be saved. Default is None.
    """
    print(goi)
    res = links.merged_score[links.merged_score.index==goi]
    res = res.rename(
        columns={"degree_centrality_all": "degree\ncentrality",
                 "betweenness_centrality": "betweenness\ncentrality",
                 "eigenvector_centrality": "eigenvector\ncentrality"})
    # make plots
    values = [ "degree\ncentrality",  "betweenness\ncentrality",
              "eigenvector\ncentrality"]
    for i, value in zip([1, 2, 3], values):
        plt.subplot(1, 3,  i)
        ax = sns.stripplot(data=res, y="cluster", x=value,
                      size=10, orient="h",linewidth=1, edgecolor="w",
                      order=links.palette.index.values,
                      palette=dict(links.palette.palette))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False,
                        left=False,
                        right=False,
                        top=False)
        if i > 1:
            plt.ylabel(None)
            ax.tick_params(labelleft=False)

    if not save is None:
        os.makedirs(save, exist_ok=True)
        path = os.path.join(save,
                           f"score_dynamics_in_{links.name}_{links.threshold_number}_{goi}.{settings['save_figure_as']}")
        plt.savefig(path, transparent=True)
    if plt_show:
        plt.show()


