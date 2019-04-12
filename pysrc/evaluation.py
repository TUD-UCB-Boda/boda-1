"""Winograd transformation evaluation.

Usage:
    evaluation.py benchmark [--disable-cache]
    evaluation.py graph

Options:
    -h --help   Show this screen.
    --version   Show version.
"""

import os
import subprocess
import re as regex
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc, rcParams
import matplotlib.patches as patches
from docopt import docopt
import numpy as np
from sympy import *

from main import F

# colors_hot = ['#ff5733', '#ffc300', '#c70039']
colors_hot = ['#00204a', '#005792', '#fd5f00']
colors_cold = ['#9d8f8f', '#bcbab8', '#5c8d89']
cmap_name = 'my_hot_color'
cm_hot = LinearSegmentedColormap.from_list(
        'hot_cm', colors_hot, N=3)
cm_cold = LinearSegmentedColormap.from_list(
        'cold_cm', colors_cold, N=3)

color_schemes = [cm_cold, cm_hot]

positions = [.93, -.13]

background_color = "#ffffff"
grid_color = 'black'
rc('axes', facecolor = background_color)
rc('axes', edgecolor = grid_color)
#rc('axes', linewidth = 1.2)
rc('axes', linewidth = 0.6)
rc('axes', grid = True )
rc('axes', axisbelow = True)
rc('grid',color = grid_color)
rc('grid',linestyle='-' )
#rc('grid',linewidth=0.7 )
rc('grid',linewidth=0.3 )
rc('xtick', labelsize=8) 
rc('ytick', labelsize=8)


def plot_clustered_stacked(dfall, ratios, max_alphas, print_legend, file_name, **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""

    n_clusters = len(dfall)

    fig, axes = plt.subplots(nrows=1, ncols=n_clusters)
    for ax_idx, ax in enumerate(axes):
        n_df = len(dfall[ax_idx])
        n_col = len(dfall[ax_idx][1].columns) 
        n_ind = len(dfall[ax_idx][0].index)
        for df_idx, df in enumerate(dfall[ax_idx]): # for each data frame
            axe = df.plot(kind="bar",
                          linewidth=0.05,
                          edgecolor='#233142',
                          stacked=True,
                          legend=False,
                          grid=True,
                          position=positions[df_idx],
                          ax=axes[ax_idx],
                          width=0.4,
                          align = 'center',
                          cmap=color_schemes[df_idx],
                          **kwargs)  # make bar plots

        axe2 = ratios[ax_idx].plot(kind="line",
                      secondary_y=True,
                      linewidth=0.5,
                      antialiased=True,
                      marker='x', 
                      color='#e84545',
                      markersize=3,
                      legend=False,
                      grid=False,
                      ax=axes[ax_idx],
                      **kwargs)  # make bar plots

        axe2.annotate(r'$\alpha={0}$'.format(max_alphas[ax_idx]), xy=(ratios[ax_idx].idxmax(), ratios[ax_idx][ratios[ax_idx].idxmax()]), xycoords='data', textcoords='offset pixels', xytext=(3, -3), fontsize='x-small')

        axe2.yaxis.grid(color='#bcbcbc', linestyle='-',zorder=0)

        ax.set_xlim(-0.5, n_ind)
        ax.xaxis.grid(0)
        ax.set_xlabel('')
        ax.yaxis.grid(color='#444f5a', linestyle='--')
        ax.set_xticks((np.arange(-.3, n_ind*n_df, n_df) + 1 / float(n_df + 1)) / 2.)
        ax.set_xticklabels(df.index, rotation = 45)
        # axe.set_title(title)
        if ax_idx == 0:
            axe.set_ylabel('# of Arithmetic operations', fontsize=7)
            handles, labels = axe.get_legend_handles_labels()
            handles2, labels2 = axe2.get_legend_handles_labels()
            handles = handles + handles2
            labels = labels + labels2
        if ax_idx == len(axes)-1:
            axe2.set_ylabel('Arithmetic reduction ratio', fontsize=7)

    # before_labels = ['Baseline:']
    # before_handles = [plt.Line2D((0,0),(0,0), color='w', linestyle='')]
    # after_labels = ['Optimized:']
    # after_handles = [plt.Line2D((0,0),(0,0), color='w', linestyle='')]
    # reduction_handles = []
    # reduction_labels = []
    # for idx_lbl, lbl  in enumerate(labels):
    #     if 'before' in lbl and not 'FMA' in lbl:
    #         before_labels.append(lbl.replace('_before',''))
    #         before_handles.append(handles[idx_lbl])
    #     elif 'after' in lbl:
    #         after_labels.append(lbl.replace('_after',''))
    #         after_handles.append(handles[idx_lbl])
    #     elif 'reduction' in lbl:
    #         reduction_labels.append(lbl)
    #         reduction_handles.append(handles[idx_lbl])

    # fig.legend(before_handles, before_labels, loc='lower left', bbox_to_anchor=(-.015, 1.25), ncol=3, fontsize=7, frameon=True, framealpha=0)
    # fig.legend(after_handles, after_labels, loc='lower center', bbox_to_anchor=(0.55, 1.25), ncol=4, fontsize=7, frameon=True, framealpha=0)
    # fig.legend(reduction_handles, reduction_labels, loc='lower right', bbox_to_anchor=(1, 1.25), fontsize=7, frameon=True, framealpha=0)

    if print_legend:
        draw_labels = ['Baseline:']
        draw_handles = [plt.Line2D((0,0),(0,0), color='w', linestyle='')]
        for idx_lbl, lbl  in enumerate(labels):
            if 'before' in lbl and not 'FMA' in lbl:
                draw_labels.append(lbl.replace('_before',''))
                draw_handles.append(handles[idx_lbl])
            elif 'after' in lbl:
                if not 'Optimized:' in draw_labels:
                    draw_labels.append('Optimized:')
                    draw_handles.append(plt.Line2D((0,0),(0,0), color='w', linestyle=''))
                draw_labels.append(lbl.replace('_after',''))
                draw_handles.append(handles[idx_lbl])
            elif 'reduction' in lbl:
                draw_labels.append(lbl)
                draw_handles.append(handles[idx_lbl])

        leg = fig.legend(draw_handles, draw_labels, loc='upper center', ncol=12, fontsize=7, frameon=False, edgecolor='black', borderpad=0.25, handletextpad=0.5, borderaxespad=0)
        leg.get_frame().set_linewidth(0.1)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.set_size_inches(6.5, 6.5/5.0)
    fig.savefig(file_name+".pdf", bbox_inches='tight', pad_inches=0.1)

def plot_overall_reduction_ratio(dfall, max_alphas):
    n_clusters = len(dfall)
    fig, axes = plt.subplots(nrows=1, ncols=n_clusters)

    for ax_idx, ax in enumerate(axes):
        n_ind = len(dfall[ax_idx][1])
        # print dfall[ax_idx]
        axe = dfall[ax_idx][1].plot(kind="line",
                      linewidth=0.5,
                      antialiased=True,
                      marker='x', 
                      color='#e84545',
                      markersize=3,
                      legend=False,
                      grid=False,
                      ax=axes[ax_idx])

        axe.annotate(r'$\alpha={0}$'.format(max_alphas[ax_idx]), xy=(dfall[ax_idx][1].idxmax(), dfall[ax_idx][1][dfall[ax_idx][1].idxmax()]), xycoords='data', textcoords='offset pixels', xytext=(-6, 2), fontsize='x-small')

        axe.set_ymargin
        axe.set_xlim(-0.5, n_ind)
        axe.xaxis.grid(0)
        axe.set_xlabel('')
        axe.yaxis.grid(color='#444f5a', linestyle='--')
        axe.set_xticks(np.arange(0, n_ind, 1))
        axe.set_yticks(np.arange(0,0.65, 0.2))
        axe.set_xticklabels(dfall[ax_idx][0], rotation = 45)
        if ax_idx == 0:
            axe.set_ylabel('Arithmetic \nreduction ratio', fontsize=7)

    plt.subplots_adjust(top=0.99, right=0.99)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.set_size_inches(6.5, 3.5/5.0)
    fig.savefig("overall_ratio.pdf", bbox_inches='tight', pad_inches=0)


def run_benchmarks(disable_cache):
    m_set = [2,3,4,5,6,7,8,9,10]
    r_set = [3,5,7]

    if os.path.exists('results.csv'):
        os.remove("results.csv")
        print 'Previous results removed successfully!'
    for r in r_set:
        for m in m_set:
            F(m,r, 1, disable_cache)


def draw_graphs():
    data = pd.read_csv("results.csv", names=['F', 'm', 'r', 'alpha', 'type', 'mult_after', 'mult_before', 'add_after', 'add_before', 'fma'])
    data.loc[data.type.str.startswith('G'), 'type'] = 'G'
    data.loc[data.type.str.startswith('B'), 'type'] = 'B'
    data.loc[data.type.str.startswith('A'), 'type'] = 'A'

    data_group = data.groupby(['F','type','alpha', 'm', 'r'], as_index=False, sort=False).sum()


    G_data = data_group[data_group.type == 'G']
    B_data = data_group[data_group.type == 'B']
    A_data = data_group[data_group.type == 'A']

    for name, i in zip(['G_data', 'B_data', 'A_data'], [G_data, B_data, A_data]):
        data_frame_cluster = []
        ratio_df_cluster = []
        max_alpha_cluster = []
        for r in data.r.unique():
            data_view = i[i.r == r]

            df_filts_before = pd.DataFrame(np.stack((data_view.add_before, data_view.mult_before, [0]*data_view.shape[0]), axis=1),
                               index=data_view.F,
                               columns=["Add_before", "Mul_before", "FMA_before"])
            df_filts_after = pd.DataFrame(np.stack((data_view.add_after, data_view.mult_after, data_view.fma), axis=1),
                               index=data_view.F,
                               columns=["Add_after", "Mul_after", "FMA_after"])

            data_frame_cluster.append([df_filts_before, df_filts_after])

            after_sum = data_view.mult_after + data_view.add_after + data_view.fma*2
            before_sum = data_view.mult_before + data_view.add_before
            ratio = (before_sum-after_sum)/before_sum
            ratio.name='reduction_ratio'
            max_point = data_view.iloc[ratio.reset_index().reduction_ratio.idxmax()]
            max_alpha_cluster.append(max_point.m+max_point.r-1)

            ratio_df_cluster.append(ratio.reset_index().reduction_ratio)
        plot_clustered_stacked(data_frame_cluster, ratio_df_cluster, max_alpha_cluster, True if name=='G_data' else False, name)

    overall_ratio_df_cluster = []
    max_alpha_cluster = []
    for r in data_group.r.unique():
        data_view = data_group[data_group.r == r]
        overall_summary_r_df = data_view.groupby(['F','alpha', 'm', 'r'], as_index=False, sort=False).sum()
        overall_before = overall_summary_r_df.add_before+overall_summary_r_df.mult_before
        overall_after = overall_summary_r_df.add_after+overall_summary_r_df.mult_after+overall_summary_r_df.fma*2
        overall_ratio = (overall_before-overall_after)/overall_before
        overall_ratio.name = 'reduction_ratio'

        max_point = overall_summary_r_df.iloc[overall_ratio.reset_index().reduction_ratio.idxmax()]
        max_alpha_cluster.append(max_point.m+max_point.r-1)

        
        overall_ratio_df_cluster.append((overall_summary_r_df.F, overall_ratio.reset_index().reduction_ratio))

    plot_overall_reduction_ratio(overall_ratio_df_cluster, max_alpha_cluster)

def draw_accuracy_graphs():
    fig, axes = plt.subplots(nrows=1, ncols=1)

    data = pd.read_csv("accuracy_log.csv")
    errors = []
    selected_rows = []
    for x in data['errors']:
        errors.append(eval(x))
    errors = np.array(errors).T
    df = pd.DataFrame(errors, columns=data.alpha.values)
    alpha_set = set(data.alpha.values)
    info = pd.DataFrame()
    for alpha in alpha_set:
        if len(df[alpha].shape)>1:
            idx = df[alpha].mean().tolist().index(min(df[alpha].mean()))
            selected_rows.append(idx)
            info[alpha] = df[alpha].iloc[:,idx]
        else:
            info[alpha] = df[alpha]

    ax = info.boxplot(showfliers=False)

    medians = info.median()
    ratios = [np.NaN, 1]
    for pair in zip(medians, medians[1:]):
        ratios.append((pair[1]-pair[0])/pair[0])

    ratios = pd.Series(ratios)
    ax2 = ratios.plot(kind="line", color='#ff5959', linewidth=0.7, marker='o', markersize=2,
        secondary_y=True)
    ax2.set_ylabel('Error increase rate', fontsize=7)

    # polynomials = []
    # p_count = 0
    # for alpha in alpha_set:
    #     poly = None
    #     items = data.loc[data.alpha==alpha].reset_index()
    #     if len(items)>1:
    #         poly = items.loc[selected_rows[p_count]].polynomials
    #         p_count += 1
    #     else:
    #         poly = items.loc[0].polynomials

    #     polynomials.append(str(eval(poly)))

    # polynomials = pd.Series(polynomials)
    # polynomials = pd.concat([pd.Series(info.columns), polynomials], axis=1)
    # polynomials.columns = ['alpha', 'polynomials']
    # print polynomials
    # print data.loc[selected_rows]
    # table(ax, polynomials, loc='right', colWidths=[0.1, 0.8])


    ax.set_yscale('log')
    ax.set_ylabel('L1-Norm Error', fontsize=7)
    ax.set_xlabel(r'$\alpha$', fontsize=7)
    ax.yaxis.grid(color='#444f5a', linestyle='--')
    ax.xaxis.grid(color='#a8b5c1', linestyle='--')

    plt.subplots_adjust(top=0.99, right=0.99)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.set_size_inches(3.5, 6.5/5.0)
    fig.savefig("accuracy.pdf", bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    arguments = docopt(__doc__, version='Winograd transformation evaluation v0.1')

    if arguments['benchmark']:
        run_benchmarks(arguments['--disable-cache'])
    elif arguments['graph']:
        draw_graphs()
        draw_accuracy_graphs()
    else:
        print 'Incorrect mode to operate!'
        exit()


