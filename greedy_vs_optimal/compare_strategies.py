import pandas as pd
import sys
import os
import argparse
import ast
from tqdm import tqdm
import utils.plotting as plo
import utils.helpers as hlp



def parse_args():

    parser = argparse.ArgumentParser(
        description="Define scenarios and call greedy and optimal scripts, then plots."
    )
    
    parser.add_argument(
        '-s', 
        '--save', 
        default=False, 
        action='store_true',
        help='Save all the produced plots in pdf format'
    )
    parser.add_argument(
        '-u', 
        '--users', 
        metavar='v', 
        type=int, 
        default=10000,
        help='Number of users'
    )
    parser.add_argument(
        '-i', 
        '--intervals', 
        metavar='v', 
        type=int, 
        default=100,
        help='Number of discrete intervals in the opinion space'
    )
    parser.add_argument(
        '-n', 
        '--nmax', 
        metavar='v', 
        type=int, 
        default=50,
        help='Maximum number of posts'
    )
    parser.add_argument(
        '--z-list',
        metavar='''"[(a1,a2),..,(z1,z2)]"''', 
        type=str, 
        default="[]", # take as a string
        help='List of z-pairs to be passed as a string in double quotes'
    )
    parser.add_argument(
        '--x0-list', 
        metavar='''"[(a1,a2),..,(z1,z2)]"''', 
        type=str, 
        default="[]",
        help='List of x0-pairs to be passed as a string in double quotes'
    )
    parser.add_argument(
        '--a-list', 
        metavar='e1 .. eN', 
        type=float, 
        nargs='*', 
        default=[],
        help='List of 1st weights in opinion update'
    )
    parser.add_argument(
        '--b-list', 
        metavar='e1 .. eN', 
        type=float, 
        nargs='*', 
        default=[],
        help='List of 2nd weight in opinion update'
    )
    parser.add_argument(
        '--w-list', 
        metavar='e1 .. eN', 
        type=float, 
        nargs='*', 
        default=[],
        help='List of widths of psi rect function'
    )
    parser.add_argument(
        '--xi-target', 
        type=float, 
        default=1.0, 
        help='The target opinion of the influencer'
    )
    parser.add_argument(
        '--recall', 
        default=False, 
        action='store_true',
        help='Prejudice exterts an influence also when users are not reached by a post'
    )
    parser.add_argument(
        '--only-plot', 
        default=False, 
        action='store_true',
        help='Does not call the scripts and plots the data in the csv (if present)'
    )
    parser.add_argument(
        '--only-state',
        default=False,
        action='store_true',
        help='Plots only the average opinion'
    )
    parser.add_argument(
        '--hide-title', 
        default=False, 
        action='store_true',
        help='Hides the title in the plots'
    )
    parser.add_argument(
        '--hide-labels', 
        default=False, 
        action='store_true',
        help='Hides the labels in the plots for the deltas'
    )

    args = parser.parse_args()

    # convert the list of list red as string to a list of lists
    args.z_list  = ast.literal_eval(args.z_list)  
    args.x0_list = ast.literal_eval(args.x0_list)

    left_most  = 1.0 / (2.0*args.intervals)
    right_most = 1 - left_most
    if args.z_list == []: # assign all the default values
        args.z_list.extend([(left_most, right_most), (0.5, 0.5)])

    else: # check if the provided parameters make sense
        for z_pair in args.z_list:
            if any([v<0.0 or v>1.0 for v in z_pair]):
                sys.exit('ERROR: all values of z must be in [0,1]')
    
    if args.x0_list == []: # default
        args.x0_list.extend([(left_most, right_most)])
    else:
        for x0_pair in args.x0_list:
            if any([v<0.0 or v>1.0 for v in x0_pair]):
                sys.exit('ERROR: all values of x0 must be in [0,1]')

    if args.a_list == []: # default
        args.a_list.extend([0.05])
    else:
        if any([v<0.0 or v>1.0 for v in args.a_list]):
            sys.exit('ERROR: all values of alpha must be in [0,1]')

    b_vals = {} # dict for the beta values for each alpha value
    keys = list(range(len(args.a_list)))

    if args.b_list == []: # default
        beta_default = [0.3]
        for key, v in zip(keys, args.a_list):
            b_vals[key] = beta_default
    else:
        if any([v<0.0 or v>1.0 for v in args.b_list]):
            sys.exit('ERROR: all values of c coeff must be in [0,1]')

        for key, v in zip(keys, args.a_list):
            b_vals[key] = args.b_list # list of beta for each alpha
        
    args.b_list = b_vals

    if args.w_list == []: # default
        args.w_list.extend([0.1])
    else:
        if any([v<0.0 or v>1.0 for v in args.w_list]):
            sys.exit('ERROR: all values of w must be in [0,1]')

    return args



def create_opt_list(w_vals_p: list, a_vals_p: list, b_vals_p: dict, recall: int,
                    z_pair_p: list, x_pair_p: list, B_p: int, N_p: int, xi_target: float):
    
    ret_list = [] # I want a list of dictionary options
    for w in w_vals_p:
        for idx, a in enumerate(a_vals_p):
            for b in b_vals_p[idx]:
                for z_p in z_pair_p:
                    for x_p in x_pair_p:
                        ret_list.append(
                            {
                                'w': w, 
                                'a': a, 
                                'b': b, 
                                'B': B_p, 
                                'N': N_p,
                                'z1': z_p[0], 
                                'z2': z_p[1], 
                                'x1': x_p[0],
                                'x2': x_p[1], 
                                'recall': recall, 
                                'xi_target': xi_target
                            }
                        )
    return ret_list



if __name__ == "__main__":

    args = parse_args()

    save_dir = 'res'

    if not args.only_plot:

        list_opt = create_opt_list(args.w_list, args.a_list, args.b_list, args.recall,
                                   args.z_list, args.x0_list, args.intervals, args.nmax,
                                   args.xi_target)

        res_df = pd.DataFrame(columns=['strategy', 'w', 'a', 'b', 
                                       'z1', 'z2', 'x1', 'x2', 'recall', 
                                       'N', 'B', 'xi_target', 
                                       'x_t', 'ex_t', 'x_i'
                                    ])

        # determining the optimal strategy
        print('\t...computing the optimal strategy...')

        path_to_cpp = os.path.join('utils', 'dynsolve3.cpp')
        hlp.compile_cpp(path_to_cpp)
        exe_cpp = os.path.splitext(path_to_cpp)[0]

        for opt_v in tqdm(list_opt):

            xi_t, ex_t, x_pair_t = hlp.call_cpp(opt_v, exe_cpp)

            dict_row = {
                'strategy': 'optimal', 
                'w': opt_v['w'], 
                'a': opt_v['a'], 
                'b': opt_v['b'],
                'z1': opt_v['z1'], 
                'z2': opt_v['z2'], 
                'x1': opt_v['x1'], 
                'x2': opt_v['x2'],
                'recall': opt_v['recall'], 
                'N': opt_v['N'], 
                'B': opt_v['B'], 
                'xi_target': opt_v['xi_target'],
                'x_t': [x_pair_t], # between [] to save in 1 cell
                'ex_t': [ex_t], 
                'x_i': [xi_t]
            } 

            new_row_df = pd.DataFrame(dict_row)
            res_df = pd.concat([res_df, new_row_df])


        # determining  the greedy strategy
        print('\t...computing greedy strategy...')

        for opt_v in tqdm(list_opt):

            xi_t, ex_t, x_pair_t = hlp.call_py(opt_v)

            dict_row = {
                'strategy': 'greedy', 
                'w': opt_v['w'], 
                'a': opt_v['a'], 
                'b': opt_v['b'],
                'z1': opt_v['z1'], 
                'z2': opt_v['z2'], 
                'x1': opt_v['x1'], 
                'x2': opt_v['x2'],
                'recall': opt_v['recall'], 
                'N': opt_v['N'], 
                'B': opt_v['B'], 
                'xi_target': opt_v['xi_target'],
                'x_t': [x_pair_t], 
                'ex_t': [ex_t], 
                'x_i': [xi_t]
            }

            new_row_df = pd.DataFrame(dict_row)
            res_df = pd.concat([res_df, new_row_df])
            
        
        if not os.path.isdir(save_dir): # create folder if not already present
            os.mkdir(save_dir)

        filename = os.path.join(save_dir, 'one_player_compare.csv')
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename, sep='\t')
            if existing_df.columns.tolist() == res_df.columns.tolist():
                res_df = pd.concat([existing_df, res_df])
            else:
                sys.exit("ERROR: Headers of csv files do not match.")

        res_df.to_csv(filename, sep='\t', index=False) # save to file



    # Load and plot the results (all of them)
    res_file_path = os.path.join(save_dir, 'one_player_compare.csv')
    if not os.path.exists(res_file_path):
        sys.exit('FATAL ERROR: no results file found')
    df = pd.read_csv(res_file_path, sep='\t')

    # group for same values which characterize the experiment
    grouped = df.groupby(['w', 'a', 'b', 'z1', 'z2', 'x1', 'x2', 'recall', 'N', 'B', 'xi_target'])

    # check that each group (-> experiment) has two records ('optimal' and 'strategy')
    if any([v!=2 for v in grouped.size().reset_index(name='counts')['counts'].values]):
        sys.exit('FATAL ERROR: in one cas either optimal or greedy solution is missing')
    
    for id, group in grouped:
        # in each group there are exactly two rows, one is 'optimal', the other 'greedy'
        opt_row = group[group['strategy'] == 'optimal']
        gre_row = group[group['strategy'] == 'greedy']

        plo.greedy_vs_opt(opt_row, gre_row, args)
        
        if not args.only_state:
            plo.plot_d1_d2_xi( gre_row, args)
            plo.plot_d1_d2_xi(opt_row, args)
