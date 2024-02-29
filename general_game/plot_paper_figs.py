import os
import sys
import argparse
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
sns.set_theme(context="talk", 
              style="ticks", 
              palette="deep", 
              font="sans-serif",
              color_codes=True, 
              font_scale=1.1,
              rc={
                  'figure.facecolor': 'white', 
                  "font.family": "sans-serif", 
                  'axes.labelpad': 8, 
                  'legend.fontsize':17,
                  'lines.linewidth': 0.8,
                  'lines.linestyle': '--', 
                  'lines.marker': 'o', 
                  'lines.markersize': 6
                  }
                )


if __name__ == '__main__':  

    parser = argparse.ArgumentParser(
        description='Plot the results for the general game'
    )
    parser.add_argument(
        '--filename', 
        type=str, 
        default=os.path.join('res', 'fast_vs_slow_closed.csv'),
        help='filename to load'
    )

    args = parser.parse_args()

    path_to_file = args.filename
    if not os.path.isfile(path_to_file):
        sys.exit(f'File {path_to_file} does not exist')

    print('INFO: only to plot the Figs in the paper following README (reading sequence hardcoded).')

    df = pd.read_csv(path_to_file, sep='\t')
    df['z_id'] = pd.factorize(df['rho'])[0] # id for each distribution

    print('Dataframe Info::\n', df[['alpha', 'beta', 'z_id', 'ne']])


    fig1, ax1 = plt.subplots()

    n_tot = 100000 # default parameter in the scripts (hardcoded dependency)

    ax1.plot(
        np.linspace(0,1,101), 
        [e/n_tot for e in ast.literal_eval(df['x_0'].iloc[0])], 
        color='tab:gray', 
        label='initial distribution',
        markersize=2
    )
    ax1.plot(
        np.linspace(0,1,101), 
        [e/n_tot for e in ast.literal_eval(df['x_N'].iloc[0])[0]],
        color='tab:blue', 
        label='slow-dynamics', 
        marker='s', 
        markerfacecolor='none', 
        markeredgecolor='tab:blue', 
        markeredgewidth=1
    )
    ax1.plot(
        np.linspace(0,1,101), 
        [e/n_tot for e in ast.literal_eval(df['x_N'].iloc[1])[0]], 
        color='tab:red', 
        label='fast-dynamics', 
        marker='^', 
        markerfacecolor='none', 
        markeredgecolor='tab:red', 
        markeredgewidth=1
    )
    ax1.legend()
    ax1.set_xlabel(r'$x^{(u)}$')
    ax1.set_ylabel(r'$\rho^{(u)}$')
    ax1.set_ylim([0,0.125])



    fig2, ax2 = plt.subplots()

    ax2.plot(
        np.linspace(0,1,101), 
        [e/n_tot for e in ast.literal_eval(df['x_0'].iloc[2])],
        color='tab:gray', 
        label='initial distribution',
        markersize=2
    )
    ax2.plot(
        np.linspace(0,1,101), 
        [e/n_tot for e in ast.literal_eval(df['x_N'].iloc[2])[0]],
        color='tab:blue', 
        label='slow-dynamics',
        marker='s', 
        markerfacecolor='none', 
        markeredgecolor='tab:blue', 
        markeredgewidth=1
    )
    ax2.plot(
        np.linspace(0,1,101), 
        [e/n_tot for e in ast.literal_eval(df['x_N'].iloc[3])[0]], 
        color='tab:red', label='fast-dynamics', 
        marker='^', 
        markerfacecolor='none', 
        markeredgecolor='tab:red', 
        markeredgewidth=1
    )
    ax2.set_xlabel(r'$x^{(u)}$')
    ax2.set_ylabel(r'$\rho^{(u)}$')
    ax2.set_ylim([0,0.125])


    # zoomed-in data inside the figure
    x1, x2, y1, y2 = 0.15, 0.9, 0.0, 0.035 # bounds for the zoomed-in plot

    axins = inset_axes(ax2, width='80%', height='70%', loc='upper right')

    axins.plot([], [], markersize=2, color='tab:gray', label='initial distribution') # here just to have it in the legend
    axins.plot(
        np.linspace(0,1,101), 
        [e/n_tot for e in ast.literal_eval(df['x_N'].iloc[2])[0]], 
        color='tab:blue', 
        label='slow-dynamics', 
        marker='s', markersize=4, 
        markerfacecolor='none', 
        markeredgecolor='tab:blue', 
        markeredgewidth=0.75
    )
    axins.plot(
        np.linspace(0,1,101), 
        [e/n_tot for e in ast.literal_eval(df['x_N'].iloc[3])[0]], 
        color='tab:red', 
        label='fast-dynamics', 
        marker='^', 
        markersize=4, 
        markerfacecolor='none', 
        markeredgecolor='tab:red', 
        markeredgewidth=0.75
    )
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])  # Hide x-axis ticks
    axins.set_yticks([])  # Hide y-axis ticks

    axins.legend()

    ax2.indicate_inset_zoom(axins, edgecolor="black") # mark zoom



    # ************** Make the Figures for the Best responses **************

    sns.set_theme(context="talk", 
                style="ticks", 
                palette="deep", 
                font="sans-serif",
                color_codes=True, 
                font_scale=1.2,
                rc={
                    'figure.facecolor': 'white', 
                    "font.family": "sans-serif", 
                    'axes.labelpad': 8, 
                    'legend.fontsize':15,
                    'lines.linewidth': 0.8,
                    'lines.linestyle': '--', 
                    'lines.marker': 'o', 
                    'lines.markersize': 6
                    }
                )


    fig3, axs1 = plt.subplots(1,2, figsize=(10,3.5))

    col0_slow = ast.literal_eval(
        df.iloc[0]['col0'].replace("array", "").replace("(", "").replace(")", "")
    ) # benefit of 0 if 1 was stubborn
    row0_slow = ast.literal_eval(
        df.iloc[0]['row0'].replace("array", "").replace("(", "").replace(")", "")
    )
    col0_fast = ast.literal_eval(
        df.iloc[1]['col0'].replace("array", "").replace("(", "").replace(")", "")
    ) # benefit of 0 if 1 was stubborn
    row0_fast = ast.literal_eval(
        df.iloc[1]['row0'].replace("array", "").replace("(", "").replace(")", "")
    )

    axs1[0].plot(
        [v[0] for v in col0_slow[0]], 
        color='tab:blue', 
        label='slow-dynamics', 
        marker='s', markerfacecolor='none', 
        markeredgecolor='tab:blue', 
        markeredgewidth=1
    ) # if player 1 was stubborn
    axs1[0].plot([
        v[0] for v in col0_fast[0]], 
        color='tab:red', 
        label='fast-dynamics', 
        marker='^', 
        markerfacecolor='none', 
        markeredgecolor='tab:red', 
        markeredgewidth=1
    ) # if player 1 was stubborn
    axs1[0].set_xlabel(r'$a^{(0)}$')
    axs1[0].set_ylabel('Player 0 Payoff')
    axs1[0].set_xticks(np.arange(0, len(col0_slow[0]) + 1, step=2))

    axs1[1].plot(
        [v[1] for v in row0_slow[0]], 
        color='tab:blue', 
        label='slow-dynamics', 
        marker='s', 
        markerfacecolor='none', 
        markeredgecolor='tab:blue', 
        markeredgewidth=1
    ) # if player 0 was stubborn
    axs1[1].plot(
        [v[1] for v in row0_fast[0]], 
        color='tab:red', 
        label='fast-dynamics', 
        marker='^', 
        markerfacecolor='none',
          markeredgecolor='tab:red', 
          markeredgewidth=1
    ) # if player 0 was stubborn
    axs1[1].set_xlabel(r'$a^{(1)}$')
    axs1[1].set_ylabel('Player 1 Payoff')
    axs1[1].set_xticks(np.arange(0, len(col0_slow[0]) + 1, step=2))

    axs1[0].set_ylim([0.68, 0.88])
    axs1[1].set_ylim([0.14, 0.32])
    num_ticks = 4
    axs1[0].yaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    axs1[1].yaxis.set_major_locator(plt.MaxNLocator(num_ticks))

    axs1[0].legend(loc='lower left')
    axs1[1].legend(loc='center right')



    fig4, axs2 = plt.subplots(1,2, figsize=(10,3.5))

    col0_slow = ast.literal_eval(
        df.iloc[2]['col0'].replace("array", "").replace("(", "").replace(")", "")
    ) # benefit of 0 if 1 was stubborn
    row0_slow = ast.literal_eval(
        df.iloc[2]['row0'].replace("array", "").replace("(", "").replace(")", "")
    )
    col0_fast = ast.literal_eval(
        df.iloc[3]['col0'].replace("array", "").replace("(", "").replace(")", "")
    ) # benefit of 0 if 1 was stubborn
    row0_fast = ast.literal_eval(
        df.iloc[3]['row0'].replace("array", "").replace("(", "").replace(")", "")
    )

    axs2[0].plot(
        [v[0] for v in col0_slow[0]], 
        color='tab:blue', 
        label='slow-dynamics',
          marker='s',
          markerfacecolor='none', 
          markeredgecolor='tab:blue', 
          markeredgewidth=1
    ) # if player 1 was stubborn
    axs2[0].plot(
        [v[0] for v in col0_fast[0]], 
        color='tab:red', 
        label='fast-dynamics', 
        marker='^',
        markerfacecolor='none', 
        markeredgecolor='tab:red', 
        markeredgewidth=1
    ) # if player 1 was stubborn
    axs2[0].set_xlabel(r'$a^{(0)}$')
    axs2[0].set_ylabel('Player 0 Payoff')
    axs2[0].set_xticks(np.arange(0, len(col0_slow[0]) + 1, step=2))

    axs2[1].plot(
        [v[1] for v in row0_slow[0]],
        color='tab:blue', 
        label='slow-dynamics', 
        marker='s', 
        markerfacecolor='none', 
        markeredgecolor='tab:blue', 
        markeredgewidth=1
    ) # if player 0 was stubborn
    axs2[1].plot(
        [v[1] for v in row0_fast[0]], 
        color='tab:red', 
        label='fast-dynamics',
        marker='^', 
        markerfacecolor='none', 
        markeredgecolor='tab:red', 
        markeredgewidth=1
    ) # if player 0 was stubborn
    axs2[1].set_xlabel(r'$a^{(1)}$')
    axs2[1].set_ylabel('Player 1 Payoff')
    axs2[1].set_xticks(np.arange(0, len(col0_slow[0]) + 1, step=2))

    axs2[0].set_ylim([0.67, 0.76])
    axs2[1].set_ylim([0.23, 0.32])
    num_ticks = 4
    axs2[0].yaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    axs2[1].yaxis.set_major_locator(plt.MaxNLocator(num_ticks))


    axs2[0].legend(loc='lower left')
    axs2[1].legend(loc='center right')


    save_dir = 'res'
    if not os.path.isdir(save_dir): # create folder if not already present
        os.mkdir(save_dir)

    fig1.savefig(os.path.join(save_dir, 'fig8a.pdf'), bbox_inches='tight')
    fig2.savefig(os.path.join(save_dir, 'fig8b.pdf'), bbox_inches='tight')

    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, 'fig7a.pdf'), bbox_inches='tight')

    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, 'fig7b.pdf'), bbox_inches='tight')
