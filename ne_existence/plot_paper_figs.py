import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties # for bold legend title
import seaborn as sns
sns.set_theme(
    context="talk", 
    style="ticks", 
    palette="deep", 
    font="sans-serif",
    color_codes=True, 
    font_scale=1.2,
    rc={
        'figure.facecolor': 'white', 
        'font.family': 'sans-serif', 
        'axes.labelpad': 8, 
        'legend.fontsize':20,
        'lines.markersize': 8, 
        'lines.linewidth': 0.8,
         'lines.linestyle': '--', 
         'lines.marker': 'o', 
         'lines.markersize': 2
        }
    )


def uniqe_vals(matrix_list):
    unique_values = []
    for r in matrix_list:
        for e in r:
            if not e in unique_values:
                unique_values.append(e)
    return unique_values


# plots the nash equilibria in a 2D grid each unique value has a different color
def ne_color_plot_custom(ne_matrix, x_vals_p, y_vals_p, el_color_mapping_p, unique_vals, add_info_p):

    print(f'\t the delta values are ({add_info_p[0]}, {add_info_p[1]})')

    fig, ax = plt.subplots(figsize=(9, 7))

    # for legend appearence
    legend_dict = {}
    for uv in unique_vals:
        uv = str(uv)
        col = el_color_mapping_p[uv]
        legend_dict[tuple(col)] = ax.scatter([], [], label=uv, color=col)

    num_rows, num_cols = len(ne_matrix), len(ne_matrix[0])
    
    for i in range(num_rows):
        for j in range(num_cols):
            value = ne_matrix[i][j]
            ax.plot(x_vals_p[j], 
                    y_vals_p[i],
                    marker='o', 
                    markersize=6, 
                    color=el_color_mapping_p[str(value)], 
                    label=value
                )

    ax.set_ylabel(r'$\beta$')
    ax.set_ylim(0,1)
    ax.set_xlabel(r'$\delta$')
    ax.set_xlim(0,1)
    legend = ax.legend(bbox_to_anchor=(1.01, 1), 
                       loc='upper left', 
                       handles=list(legend_dict.values()), 
                       scatterpoints=1,
                       title_fontsize=13, 
                       title='Nash Equilibria', 
                       fontsize=18
                    )
    title_font = FontProperties(weight='bold', size='small')
    legend.get_title().set_fontproperties(title_font)
    for hl in legend.legendHandles:
        hl._sizes = [160] # make the dots in the legend bigger
    ax.grid(True)
    fig.tight_layout()

    if save_flag:
        save_dir = 'res'
        if not os.path.isdir(save_dir): # create folder if not already present
            os.mkdir(save_dir)
        fig.savefig(
            os.path.join(save_dir, f'ne_color_{add_info_p[2]}_{add_info_p[0]}_{add_info_p[1]}.pdf'), 
            bbox_inches='tight'
        )
    else:
        plt.show()



if __name__ == '__main__':

    save_flag = True # always save the plots in 'res'

    df = pd.read_csv(os.path.join('res', 'ne_existence_data.csv'), sep='\t')

    # simplification n_x, n_y in this case are the same for all (not necessarily true always)
    nx = ast.literal_eval(df['x_vals'].iloc[0])
    ny = ast.literal_eval(df['y_vals'].iloc[0])


    all_fig6a = ast.literal_eval(df['nash_equilibria'].iloc[0])
    unique1 = uniqe_vals(all_fig6a)
    add_info1 = (df['delta0'].iloc[0], df['delta1'].iloc[0], df['z_vec'].iloc[0])

    all_fig6b = ast.literal_eval(df['nash_equilibria'].iloc[1])
    unique2 = uniqe_vals(all_fig6b)
    add_info2 = (df['delta0'].iloc[1], df['delta1'].iloc[1], df['z_vec'].iloc[1])

    all_fig6c = ast.literal_eval(df['nash_equilibria'].iloc[2])
    unique3 = uniqe_vals(all_fig6c)
    add_info3 = (df['delta0'].iloc[2], df['delta1'].iloc[2], df['z_vec'].iloc[2])

    all_fig5 = ast.literal_eval(df['nash_equilibria'].iloc[3])
    unique4 = uniqe_vals(all_fig5)
    add_info4 = (df['delta0'].iloc[3], df['delta1'].iloc[3], df['z_vec'].iloc[3])

    all_sublists = unique1 + unique2 + unique3 + unique4
    unique_values = [] # I do them as a string to use them as keys in dictionary later
    for e in all_sublists:
        e = str(e)
        if not e in unique_values:
            unique_values.append(e)

    # Create color palette
    colors_list = list(sns.color_palette("twilight", n_colors=20))\
                    + list(sns.color_palette("tab20c", n_colors=20))

    # Create a FIXED mapping between each unique element and a consistent color
    element_color_mapping = {}

    element_color_mapping['[(1, 1)]'] = colors_list[3]
    element_color_mapping['[(1, 5)]'] = colors_list[-22]
    element_color_mapping['[]'] = colors_list[-26]
    element_color_mapping['[(5, 5)]'] = colors_list[-12]
    element_color_mapping['[(1, 2)]'] = colors_list[23]
    element_color_mapping['[(1, 3)]'] = colors_list[12]
    element_color_mapping['[(1, 4)]'] = colors_list[1]
    element_color_mapping['[(2, 5)]'] = colors_list[-10]
    element_color_mapping['[(2, 4)]'] = colors_list[10]
    element_color_mapping['[(4, 5)]'] = colors_list[25]
    element_color_mapping['[(3, 5)]'] = colors_list[27]
    element_color_mapping['[(2, 2)]'] = colors_list[6]
    element_color_mapping['[(3, 3)]'] = colors_list[11]
    element_color_mapping['[(4, 4)]'] = colors_list[15]

    res_unique = set(unique_values) - {'[]', '[(5, 5)]', '[(1, 1)]', '[(2, 5)]',
                                    '[(1, 5)]', '[(2, 2)]', '[(3, 3)]', '[(4, 4)]',
                                    '[(1, 4)]', '[(1, 3)]', '[(1, 2)]', '[(2, 3)]',
                                    '[(4, 5)]', '[(3, 5)]'}
    res_colors = set(colors_list) - {colors_list[-26], colors_list[-12], colors_list[3], 
                                    colors_list[-10], colors_list[-22], colors_list[6], 
                                    colors_list[12], colors_list[15], colors_list[1],
                                    colors_list[25], colors_list[23], colors_list[10],
                                    colors_list[11], colors_list[27]}

    for element, color in zip(res_unique, res_colors):
        element_color_mapping.setdefault(element, color)


    # plot all the figures included in the paper
    ne_color_plot_custom(all_fig6a, nx, ny, element_color_mapping, unique1, add_info1)
    ne_color_plot_custom(all_fig6b, nx, ny, element_color_mapping, unique2, add_info2)
    ne_color_plot_custom(all_fig6c, nx, ny, element_color_mapping, unique3, add_info3)
    ne_color_plot_custom(all_fig5, nx, ny, element_color_mapping, unique4, add_info4)
