import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import sys
import os
import seaborn as sns
sns.set_theme(context="talk",
              style="ticks",
              palette="deep",
              font="sans-serif",
              color_codes=True,
              font_scale=1.4,
              rc={
                  'figure.facecolor': 'white',
                  'font.family': 'sans-serif', 
                  'axes.labelpad': 8,
                  'legend.fontsize': 21,
                  'lines.markersize': 7,
                  'lines.linewidth': 0.8,
                  'lines.linestyle': '--',
                  'lines.marker': 'o'
                }
            )
sns.set_palette('tab10')



# defines a filename accoridng to the parameters of the experiment
def find_filename(prefix: str, row: pd.core.frame.DataFrame):
    recall_flag = ''
    if row['recall'].iloc[0] > 0:
        recall_flag = 'recall'
    filename = prefix + '_w_' + str(row['w'].iloc[0]) + '_a_' +\
                str(row['a'].iloc[0]) + '_b' + str(round(row['b'].iloc[0], 3)) +\
                '_z0_' + str(row['z1'].iloc[0]) + '_z1_' + str(row['z2'].iloc[0]) +\
                '_x0_' + str(row['x1'].iloc[0]) + '_x1_' + str(row['x2'].iloc[0]) +\
                recall_flag + '_xi_' + str(row['xi_target'].iloc[0])
    return filename



# check if the two rows (of a dataframe) refer to the same set of parameters
def check_rows(row1: pd.core.frame.DataFrame, row2: pd.core.frame.DataFrame):

    f1 = row1['w'].iloc[0] == row2['w'].iloc[0]
    f2 = row1['a'].iloc[0] == row2['a'].iloc[0]
    f3 = row1['b'].iloc[0] == row2['b'].iloc[0]
    f4 = row1['z1'].iloc[0] == row2['z1'].iloc[0]
    f5 = row1['z2'].iloc[0] == row2['z2'].iloc[0]
    f6 = row1['x1'].iloc[0] == row2['x1'].iloc[0]
    f7 = row1['x2'].iloc[0] == row2['x2'].iloc[0]
    f8 = row1['recall'].iloc[0] == row2['recall'].iloc[0]
    f9 = row1['xi_target'].iloc[0] == row2['xi_target'].iloc[0]

    return f1 and f2 and f3 and f4 and f5 and f6 and f7 and f8 and f9



def find_title(row: pd.core.frame.DataFrame):

    tit = r'$\alpha=$' + str(row['a'].iloc[0]) + ', ' + \
          r'$\beta=$' + str(row['b'].iloc[0]) + ' and ' + \
          r'$\psi(w=$' + str(row['w'].iloc[0]) + '$)$' + \
          r'$x^{(i;T)}=$' + str(row['xi_target'].iloc[0])
    
    if row['z1'].iloc[0] == row['x1'].iloc[0] and row['z2'].iloc[0] == row['x2'].iloc[0]:
        tit += '\nusers have ' + \
               r'$z^{(1)}=x^{(1)}_0=$' + str(row['z1'].iloc[0]) + ' and ' + \
               r'$z^{(2)}=x^{(2)}_0=$' + str(row['z2'].iloc[0])
    
    elif row['z1'].iloc[0] == row['z2'].iloc[0]:
        tit += '\nusers have ' + \
               r'$z^{(1)}=z^{(2)}_0=$' + str(row['z1'].iloc[0]) + ' and ' + \
               r'$x^{(1)}=$' + str(row['x1'].iloc[0]) + ', ' + \
               r'$x^{(2)}=$' + str(row['x2'].iloc[0])
        
    else: # general case we write all the info we have
        tit += '\n1st delta: ' + r'$z^{(1)}=$' + str(row['z1'].iloc[0]) + ', ' +\
                    r'$x^{(1)}=$' + str(row['x1'].iloc[0]) + \
               '\n2nd delta: ' + r'$z^{(2)}=$' + str(row['z2'].iloc[0]) + ', ' +\
                    r'$x^{(2)}=$' + str(row['x2'].iloc[0]) 
               
    if row['recall'].iloc[0]:
        tit = 'self-thinking ' + tit

    return tit



# plots the average opinion of the user population over time for the greedy and 
# for the optimal strategy, considering the same values of the parameters
def greedy_vs_opt(opt_row: pd.core.frame.DataFrame,
                  greedy_row: pd.core.frame.DataFrame, args_p):
    
    if check_rows(opt_row, greedy_row):

        ex_opt = ast.literal_eval(opt_row['ex_t'].iloc[0])
        ex_gre = ast.literal_eval(greedy_row['ex_t'].iloc[0])

        # subsampled markers with dots
        fig, ax = plt.subplots(figsize=(7.5, 6))

        sub_every = 3

        # optimal strategy
        ax.plot(ex_opt, color='tab:cyan', linestyle='solid', marker='')

        ax.plot([], [], linestyle='solid', color='tab:cyan',
                marker='o', markerfacecolor='none', markeredgecolor='tab:cyan', markeredgewidth=1.5,
                label='optimal') # for legend
        for i in range(0, len(ex_opt), sub_every):
            ax.plot(i, ex_opt[i], marker='o', markerfacecolor='none', markeredgecolor='tab:cyan', markeredgewidth=1.5)
        for i in range(0, len(ex_opt)):
            if i % sub_every != 0:
                ax.plot(i, ex_opt[i], marker='o', markerfacecolor='none', markeredgecolor='tab:cyan', markersize=2.5)
        
        # greedy strategy
        ax.plot(ex_gre, color='tab:red', linestyle='solid', marker='')

        ax.plot([], [], linestyle='solid', color='tab:red',
                marker='^', markerfacecolor='none', markeredgecolor='tab:red', markeredgewidth=1.5,
                label='greedy') # for legend
        for i in range(0, len(ex_gre), sub_every):
            ax.plot(i, ex_gre[i], marker='^', markerfacecolor='none', markeredgecolor='tab:red', markeredgewidth=1.5)
        for i in range(0, len(ex_gre)):
            if i % sub_every != 0:
                ax.plot(i, ex_gre[i], marker='o', markerfacecolor='none', markeredgecolor='tab:red', markersize=2.5)

        ax.legend(loc='lower right')

        ax.set_xlabel(r'$n$')
        ax.set_ylabel(r'$\mathbb{E}_x[x_n]$')

        ax.set_ylim(0.5, 1.0)

        if not args_p.hide_title:
            ax.set_title(find_title(opt_row))

        if args_p.save:
            # the two rows refer to the same set of parameters
            filename = find_filename('gre_vs_opt', opt_row) + '.pdf'
            fig.savefig(os.path.join('res', filename), bbox_inches='tight')
        else:
            fig.tight_layout()
            plt.show()
    else:
        sys.exit('ERROR: the provided rows do not refer to the same experiment')



def plot_d1_d2_xi(row: pd.core.frame.DataFrame, args_p):

    x_t = ast.literal_eval(row['x_t'].iloc[0]) # list of pairs (opinions)
    xi  = ast.literal_eval(row['x_i'].iloc[0])

    if len(x_t) != (row['N'].iloc[0] + 1): # N+1 elements
        sys.exit('ERROR: opinion sequence is of {} elements instead of {}'.format(
                    len(x_t), row['N'].iloc[0] + 1
                ))
        
    elif len(xi) != row['N'].iloc[0]:
        sys.exit('ERROR: xi optimal sequence is of {} elements instead of {}'.format(
                    len(xi), row['N'].iloc[0]
                ))
    
    else:
        xi = [np.nan] + xi

        # subsampled markers with dots
        fig, ax = plt.subplots(figsize=(7.5, 6))

        sub_every = 3

        ax.plot([i for i in range(len(xi))], xi, color='tab:orange', linestyle='solid', marker='')

        # plot influencer strategy
        ax.plot([], [], linestyle='solid', color='tab:orange',
                marker='o', markerfacecolor='none', markeredgecolor='tab:orange', markeredgewidth=1.5,
                label=r'$x^{(i)}$') # for legend
        for i in range(0, len(xi), sub_every):
            ax.plot(i, xi[i], marker='o', markerfacecolor='none', markeredgecolor='tab:orange', markeredgewidth=1.5)
        for i in range(0, len(xi)):
            if i % sub_every != 0:
                ax.plot(i, xi[i], marker='o', markerfacecolor='none', markeredgecolor='tab:orange', markersize=2.5)

        # plot group 1 opinion distribution value
        d1_t = [pair[0] for pair in x_t]
        ax.plot([i for i in range(len(d1_t))], d1_t, color='tab:blue', linestyle='solid', marker='')

        ax.plot([], [], linestyle='solid', color='tab:blue',
                 marker='^', markerfacecolor='none', markeredgecolor='tab:blue', markeredgewidth=1.5,
                 label=r'$x^{(u=0)}$') # for legend
        for i in range(0, len(d1_t), sub_every):
            ax.plot(i, d1_t[i], marker='^', markerfacecolor='none', markeredgecolor='tab:blue', markeredgewidth=1.5)
        for i in range(0, len(d1_t)):
            if i % sub_every != 0:
                ax.plot(i, d1_t[i], marker='o', markerfacecolor='none', markeredgecolor='tab:blue', markersize=2.5)
        
        # plot group 2 opinion distribution value
        d2_t = [pair[1] for pair in x_t]
        ax.plot([i for i in range(len(d2_t))], d2_t, color='tab:green', linestyle='solid', marker='')

        ax.plot([], [], linestyle='solid', color='tab:green',
                marker='s', markerfacecolor='none', markeredgecolor='tab:green', markeredgewidth=1.5,
                label=r'$x^{(u=1)}$') # for legend
        for i in range(0, len(d2_t), sub_every):
            ax.plot(i, d2_t[i], marker='s', markerfacecolor='none', markeredgecolor='tab:green', markeredgewidth=1.5)
        for i in range(0, len(d2_t)):
            if i % sub_every != 0:
                ax.plot(i, d2_t[i], marker='o', markerfacecolor='none', markeredgecolor='tab:green', markersize=2.5)

        if not args_p.hide_labels:
            ax.set_xlabel(r'$n$')
            ax.set_ylabel('opinion')

        if not args_p.hide_title:
            ax.set_title( row['strategy'].iloc[0] + ' strategy, ' +find_title(row))

        ax.legend(loc='lower right')

        if args_p.save:
            filename = find_filename('gre_vs_opt', row)  + row['strategy'].iloc[0] + '.pdf'
            fig.savefig(os.path.join('res', filename), bbox_inches='tight')
        else:
            fig.tight_layout()
            plt.show()
