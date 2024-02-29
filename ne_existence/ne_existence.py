# Script that plots for which choices of parameters \delta and \beta the game
# has at least one nash equilibrium (if the NE is not unique the scripts writes
# it to the console). All the parameters that are not \delta or \beta can be 
# passed from command line interface. Plots also the values of the different NE 

# NOTE: to uniform the "language" of the 'rect' script and the general script we
# save the characteristic parameter of the \psi function as p1 and p2. 
# (in the rect case p1=1.0, p2=w and in the general it depends on \psi choice)

import os
import sys
import numpy as np
import pandas as pd
import utils.arguments as arguments
import subprocess
import ast
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
import colorcet as cc
import seaborn as sns
sns.set_theme(context="talk",
              style="ticks", 
              palette="deep", 
              font="sans-serif",
              color_codes=True, 
              font_scale=1.2,
              rc={
                  'figure.facecolor': 'white', 
                  'font.family': "sans-serif", 
                  'axes.labelpad': 8, 
                  'legend.fontsize':15,
                  'lines.markersize': 8, 
                  'lines.linewidth': 0.8,
                  'lines.linestyle': '--', 
                  'lines.marker': 'o'
                }
            )
sns.set_palette('tab10')



# call the script where the Nash equilibria for the two players game is computed
# 'other_param' is a dict, the keys are the args name and the values are the args values
def run_simulator(a_p, b_p, other_param, no_recall_p, strict_p):

    add_args = ' '.join(['{} {}'.format(k, v) for k,v in other_param.items()])
    
    # add the boolean flags if specified
    if no_recall_p:
        add_args += ' --no-recall'
    if strict_p:
        add_args += ' --strict' # Note: used only with the 'xxx_rect.csv' version
    
    script_filename = 'two_players.py' if args.general else 'two_players_rect.py'
    script_path = os.path.join('utils', script_filename)

    command = f'python {script_path} -a {a_p} -b {b_p} {add_args} --existence'
    
    # run the subprocess and collect its output from std out
    result = subprocess.run(command, capture_output=True, text=True, shell=True)

    # check for errors in the subprocess
    err = result.stderr.strip()
    if err != "":
        print('\nSUBPROCESS ERROR: an error occured in the called script')
        sys.exit(err)

    # parse the result and return the list of Nash equilibria
    ne_list_str = result.stdout.strip()
    output_list = ast.literal_eval(ne_list_str) # convert string to list   
    if not isinstance(output_list, list):
        sys.exit('FATAL ERROR: the simulator output is not a list')
    
    return output_list



# the script that finds NE accepts alpha instead of delta, but the two are in 1-1
def from_delta_to_alpha(delta_p, beta_p):
    return delta_p * (1-beta_p)



def interactive_overwrite():
    answer = input("\t\tDo you want to overwrite the experiment? (yes/no): ")
    if answer.lower() == "yes" or answer.lower() == "y":
        print()
    elif answer.lower() == "no" or answer.lower() == "n":
        print()
        sys.exit(1)
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
        interactive_overwrite()



def bool_existence_plot(nash_equilibria_p, nx_p, ny_p, x_vals_p, y_vals_p, title_p):

    plt.figure(figsize=(8, 8))

    #translate the data into boolean data: NE exists/not
    nash_existence = np.zeros((ny_p, nx_p), dtype=bool)
    for i, row in enumerate(nash_equilibria_p): # i: idx row, j: idx column
        for j, element in enumerate(row):
            nash_existence[i, j] = bool(len(element))
    

    custom_colors = ['tab:red', 'tab:green']
    cmap = colors.ListedColormap(custom_colors)
    extent = [
        min(x_vals_p) - (1/(2*(nx_p-1))), 
        max(x_vals_p) + (1/(2*(nx_p-1))), 
        min(y_vals_p) - (1/(2*(ny_p-1))), 
        max(y_vals_p) + (1/(2*(ny_p-1)))
    ]
    plt.imshow(nash_existence, origin='lower', extent=extent, cmap=cmap, vmin=0, vmax=1, interpolation='none')
    
    x_grid_p, y_grid_p = np.meshgrid(x_vals_p, y_vals_p) # reconstruct the grid of points
    plt.scatter(x_grid_p.flatten(), y_grid_p.flatten(), color='k', edgecolors='k', marker='o', s=4)
    plt.xlabel(r'$\delta = \frac{\alpha}{1-\beta}$')
    plt.xlim([0, 1])
    plt.ylabel(r'$\beta$')
    plt.ylim([0, 1])
    if title_p == '':
        plt.title('Existence of Nash Equilibrium')
    else:
        plt.title(title_p, y=1.03, fontsize=10)
    plt.colorbar().remove() # remove the colorbar (rather useless)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# plot the NE over the (\delta, \beta) space, each NE with a different color
def ne_color_plot(ne_matrix, x_vals_p, y_vals_p):

    unique_values = []
    for r in ne_matrix:
        for e in r:
            if not e in unique_values:
                unique_values.append(e)

    num_colors = len(unique_values)
    palette = "glasbey"
    colors = cc.palette[palette][:num_colors]

    plt.figure(figsize=(9, 7))

    legend_dict = {}
    for val in unique_values:
        if tuple(val) not in legend_dict:
            legend_dict[tuple(val)] = plt.scatter([], [], label=val, color=colors[list(unique_values).index(val)])

    num_rows, num_cols = len(ne_matrix), len(ne_matrix[0]) # valid because it is squared and there is at least one element
    
    for i in range(num_rows):
        for j in range(num_cols):
            value = ne_matrix[i][j]
            value_indices = list(unique_values).index(value)
            plt.plot(x_vals_p[j], y_vals_p[i], marker='o', markersize=8, color=colors[value_indices], label=value)

    plt.ylabel(r'$\beta$')
    plt.ylim(0,1)
    plt.xlabel(r'$\delta$')
    plt.xlim(0,1)
    plt.grid(True)
    legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', handles=list(legend_dict.values()), scatterpoints=1, title='Nash Equilibria', fontsize=15)
    for hl in legend.legendHandles:
        hl._sizes = [160]
    plt.tight_layout()
    plt.show()

   

if __name__ == '__main__':

    args = arguments.parse_args()

    delta_range = (0, 1) # degree of stubborness \delta = \alpha = (1 - \beta)
    beta_range  = (0, 1)

    delta_pts = nx = args.n_points # points along x-axis
    beta_pts  = ny = args.n_points # points along y-axis

    x_vals = []
    y_vals = []
    nash_equilibria = []

    if args.only_plot: # retrieve and organize all the experiments

        filename = 'ne_existence_data_general.csv' if args.general else 'ne_existence_data.csv'
        filename_path = os.path.join('res', filename)

        if not os.path.exists(filename_path): # check csv file existence
            sys.exit(f'FATAL ERROR: no csv file with NE exists: {filename_path}')
        df_out = pd.read_csv(filename_path, sep='\t')

        if args.all: # the title is defined from the experiment setup

            nx_ny  = [] # nx=ny always holds
            z_info, r_info, n_info, psi_info, p1_info, p2_info, d0_info, d1_info = [], [], [], [], [], [], [], []

            for index, row in df_out.iterrows():

                nx_ny.append(row['n_x'])
                z_info.append(ast.literal_eval(row['z_vec']))
                r_info.append(ast.literal_eval(row['rho_vec']))
                n_info.append(row['n_max'])
                psi_info.append(row['psi'])
                p1_info.append(row['p1'])
                p2_info.append(row['p2'])
                d0_info.append(row['delta0'])
                d1_info.append(row['delta1'])
                nash_equilibria.append(ast.literal_eval(row['nash_equilibria']))
                x_vals.append(np.array(ast.literal_eval(row['x_vals'])))
                y_vals.append(np.array(ast.literal_eval(row['y_vals'])))
        
        else: # check if the specified experiment is in the dataset of results
            
            record_exists = (
                (df_out['n_x'] == nx) & 
                (df_out['n_y'] == ny) &
                (df_out['strict'] == args.strict) &
                (df_out['no_recall'] == args.no_recall) &
                (df_out['z_vec'].apply(lambda arr: np.array_equal(ast.literal_eval(arr), args.prejudice))) &
                (df_out['x0_vec'].apply(lambda arr: np.array_equal(ast.literal_eval(arr), args.init_x))) &
                (df_out['rho_vec'].apply(lambda arr: np.array_equal(ast.literal_eval(arr), args.rho))) & 
                (df_out['n_max'] == args.nmax) &
                (df_out['psi'] == args.psi) &
                (df_out['p1'] == args.param[0]) &
                (df_out['p2'] == args.param[1]) &
                (df_out['delta0'] == args.d0) &
                (df_out['delta1'] == args.d1)
            )
            
            if record_exists.any() == False:
                sys.exit('FATAL ERROR: the experiment cannot be plotted since not in the dataset')
            
            else:
                nash_equilibria = ast.literal_eval(df_out.loc[record_exists, 'nash_equilibria'].iloc[0])
                x_vals = np.array(ast.literal_eval(df_out.loc[record_exists, 'x_vals'].iloc[0]), dtype=object)
                y_vals = np.array(ast.literal_eval(df_out.loc[record_exists, 'y_vals'].iloc[0]), dtype=object)

    else: # run the experiments, save the results (csv) and plot them

        other_param = {
            '-z': ' '.join(str(e) for e in  args.prejudice),
            '-x': ' '.join(str(e) for e in  args.init_x), 
            '-r': ' '.join(str(e) for e in  args.rho),
            '-d0': args.d0,
            '-d1': args.d1,
             '-n': args.nmax
        }
        
        if args.general:
            other_param.update({
                '--psi': args.psi,
                '--param': ' '.join(str(e) for e in  args.param)
            })
        else: 
            other_param.update({'-w': args.param[1]}) # args.param[1] = w


        # (delta, beta) grid values are the middle point values of the 'n_points' # of bins
        half_int_x = (delta_range[1]-delta_range[0])/(2*(delta_pts))
        half_int_y = (beta_range[1]-beta_range[0])/(2*(beta_pts))

        x_vals = np.linspace(delta_range[0]+half_int_x, delta_range[1]-half_int_x, delta_pts)
        y_vals = np.linspace(beta_range[0]+half_int_y, beta_range[1]-half_int_y, beta_pts)

        x_grid, y_grid = np.meshgrid(x_vals, y_vals) # grid of points


        filename = 'ne_existence_data_general.csv' if args.general else 'ne_existence_data.csv'
        save_dir = 'res'
        if not os.path.isdir(save_dir): # create folder if not already present
            os.mkdir(save_dir)

        # check if the file exists and load it, otherwise create an empty DataFrame
        filename_path = os.path.join('res', filename)
        if os.path.exists(filename_path):
            df_out = pd.read_csv(filename_path, sep='\t')
        else:
            df_out = pd.DataFrame(columns=[
                'n_x', 
                'n_y', 
                'strict', 
                'no_recall', 
                'z_vec', 
                'x0_vec', 
                'rho_vec', 
                'n_max', 
                'psi', 
                'p1', 
                'p2', 
                'delta0', 
                'delta1', 
                'x_vals', 
                'y_vals', 
                'nash_equilibria'
            ])

        # check if the experiment has been already conducted (lists need to be converted from string)
        record_exists = (
            (df_out['n_x'] == nx) & 
            (df_out['n_y'] == ny) &
            (df_out['strict'] == args.strict) &
            (df_out['no_recall'] == args.no_recall) &
            (df_out['z_vec'].apply(lambda arr: np.array_equal(ast.literal_eval(arr), args.prejudice))) &
            (df_out['x0_vec'].apply(lambda arr: np.array_equal(ast.literal_eval(arr), args.init_x))) &
            (df_out['rho_vec'].apply(lambda arr: np.array_equal(ast.literal_eval(arr), args.rho))) & 
            (df_out['n_max'] == args.nmax) &
            (df_out['psi'] == args.psi) &
            (df_out['p1'] == args.param[0]) &
            (df_out['p2'] == args.param[1]) &
            (df_out['delta0'] == args.d0) &
            (df_out['delta1'] == args.d1)
        )
            
        save_flag = True
        if record_exists.any(): # one True in the above -> record exists
            save_flag = False   # avoid to overwrite the experiment, but execute it anyways
            interactive_overwrite()


        nash_equilibria = np.empty((ny, nx), dtype=object)
        for i in tqdm(range(ny)):
            for j in range(nx):
                x = x_grid[i, j]
                y = y_grid[i, j] # this is beta

                # translate x (degree of stubborness to alpha)
                x_alpha = from_delta_to_alpha(x, y)

                # we get the list of Nash equilibria for the game
                ne = run_simulator(x_alpha, y, other_param, args.no_recall, args.strict)
                if args.verbose and (len(ne) > 1):
                    print('Multiple ({}) NE for a={} and b={}'.format(len(ne), x_alpha, y))

                nash_equilibria[i, j] = ne # list of NE
        
        nash_equilibria = nash_equilibria.tolist() # I use it as list of lists for plotting
        
        # Note: wrap the lists around square brakets to save them as a single element
        row_to_add = {
            'n_x': nx,
            'n_y': ny,
            'strict': args.strict,
            'no_recall': args.no_recall,
            'z_vec': [args.prejudice],
            'x0_vec': [args.init_x],
            'rho_vec': [args.rho],
            'n_max': args.nmax,
            'psi': args.psi,
            'p1': args.param[0],
            'p2': args.param[1],
            'delta0': args.d0,
            'delta1': args.d1,
            'x_vals': [x_vals.tolist()],
            'y_vals': [y_vals.tolist()],
            'nash_equilibria': [nash_equilibria]
        } # save as a regular list of list
        
        df_out = pd.concat([df_out, pd.DataFrame(row_to_add)])
        df_out.to_csv(filename_path, index=False, sep='\t')


    # *******************************  plotting  *******************************

    if not args.no_plot:

        if args.all: # NE are in a list, plot all

            for idx_exp, ne in enumerate(nash_equilibria):

                # construct title starting from the experiment parameters
                str_title = (
                    r'$N=$'+str(n_info[idx_exp]) + 
                    r',$\psi=$'+str(psi_info[idx_exp]) + 
                    r' (param: {}, {})'.format(p1_info[idx_exp], p2_info[idx_exp]) +
                    r' Users: '
                )
                for idx_ze, ze in enumerate(z_info[idx_exp]):
                    str_title += (r'$z^{('+str(idx_ze)+r')}=$' + str(ze) + 
                                r',$\rho^{('+str(idx_ze)+r')}=$'+str(r_info[idx_exp][idx_ze])+' ')
                    
                str_title += ' Players: '+r'$\delta{(0)}=$'+str(d0_info[idx_exp])+r',$\delta{(1)}=$'+str(d1_info[idx_exp])
                
                # plot both the existence and the 'color' of equilibrium plots
                bool_existence_plot(ne, nx_ny[idx_exp], nx_ny[idx_exp], x_vals[idx_exp], y_vals[idx_exp], str_title)
                ne_color_plot(ne, x_vals[idx_exp], y_vals[idx_exp])


        else:
            bool_existence_plot(nash_equilibria, nx, ny, x_vals, y_vals, '')
            ne_color_plot(nash_equilibria, x_vals, y_vals)
