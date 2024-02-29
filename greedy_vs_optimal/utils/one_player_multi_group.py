# Computes strategies of a SINGLE influencer who wants to make a population of
# users move to a certain opinion (x^T=1) in the opinion space on limited time N
#
# MAIN HYPOTHESIS:
# 1a. initial distribution of users is a (multiple) delta around z
# 1b. arbitrary initial opinion x_u that by default is equal to the prejudice z
# 2. we restrict ourself to the one-dimensional case, where x \in [0,1]
# 3. each discrete event is a 'post generation', we index them with n (time)
# 4. only one influencer in the population (->\tilde{p}_i=1)
# 5. we assume that at each post the number of users which move is deterministic
#    and coincides with the mean value of users who change opinion in the stoch-
#    astic model, i.e., \omega\theta*N
#
# AVAILABLE STRATEGIES:
# - greedy : at each instant the influencer choses their opinion maximizing the
#            average \Delta s (opinion shift) of the user population
# - extreme: the influencer holds the extreme opinion x^E \in {0,1} for all n


import math
import os
import sys
from scipy import optimize
import argparse
import pandas as pd
import copy


def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Influencer's strategies to pull users towars x=1."
    )
    parser.add_argument(
        '-v', 
        '--verbose', 
        default=False, 
        action='store_true', 
        help='Detailed program output'
    )
    parser.add_argument(
        '-s',
        '--save',
        default=False,
        action='store_true',
        help='''Save temporal sequence of x averages as 'default'.csv'''
    )
    parser.add_argument(
        '-u', 
        '--users', 
        type=int, 
        default=10000, 
        help='Number of users'
    )
    parser.add_argument(
        '-i',
        '--intervals',
        type=int,
        default=100,
        help='Number of discrete intervals in the opinion space'
    )
    parser.add_argument(
        '-z', 
        '--prejudice', 
        type=float, 
        nargs='*',
        default=[], 
        help='Prejudice of the user groups (delta)'
    ) 
    parser.add_argument(
        '-x', 
        '--init-x', 
        type=float, 
        nargs='*', 
        default=[], 
        help='Initial opinion of the user groups (delta)'
    )
    parser.add_argument(
        '-c', 
        '--count', 
        type=float, 
        nargs='*', 
        default=[], 
        help='Proportion of users in each group (delta)'
    )
    parser.add_argument(
        '-a',
        '--alpha', 
        type=float, 
        default=0.2, 
        help='1st weight in opinion update'
    )
    parser.add_argument(
        '-b', 
        '--beta', 
        type=float, 
        default=0.7, 
        help='2nd weight in opinion update'
    )
    parser.add_argument(
        '-n', 
        '--nmax', 
        type=int, 
        default=50, 
        help='Maximum number of posts'
    )
    parser.add_argument(
        '--xi-target', 
        type=float, 
        default=1.0, 
        help='The target opinion of the influencer'
    )
    parser.add_argument(
        '-t', 
        '--type', 
        type=str, 
        choices=['greedy', 'extreme'], 
        default='greedy', 
        help='Set simulation type'
    )
    parser.add_argument(
        '--param', 
        type=float, 
        nargs=2, 
        default=[1.0, 0.5],
        help='Values of the two parameters controlling the shape of psi'
    )
    parser.add_argument(
        '--call', 
        default=False, 
        action='store_true',
          help='Inhibits output messages to use std:out to pass the results'
    )
    parser.add_argument(
        '--recall', 
        default=False, 
        action='store_true', 
        help='If set users go back to their prejudice when not reached by a post'
    )

    args = parser.parse_args()

    if args.prejudice == []: # set default according to bins (as in the c++ program)
        args.prejudice.extend([1.0/(2*args.intervals), 1.0 - (1.0/(2*args.intervals))])
        args.count.extend([0.5, 0.5])

    elif len(args.count) != len(args.prejudice):
        sys.exit('ERROR: group mismatch: {} z values and {} counts'.format(
                 (len(args.prejudice), len(args.count))))
    
    if args.init_x == []: # defualt, x(0)=z
        args.init_x.extend(args.prejudice)

    elif len(args.init_x) != len(args.prejudice):
        sys.exit('ERROR: group mismatch: {} x init and {} z values'.format(
                 (len(args.init_x), len(args.prejudice))))

    if abs(1-sum([v for v in args.count])) > 1e-5:
        sys.exit(f'ERROR: count values need to sum to 1, got {sum([v for v in args.count])}')

    if any([v<0.0 or v>1.0 for v in args.prejudice]):
        sys.exit('ERROR: all values of prejudice must lay in [0,1]')

    if (args.alpha + args.beta) > 1.0:
        sys.exit('ERROR: alpha+beta is > 1')

    if len(args.param) != 2:
        sys.exit('ERROR: psi parameters must be a list of 2 elements')

    if args.call == True:
        args.save = False # inhibt all the save to files when call is set

    return args


# maximum theoretical point in which a population starting in z can reach when
# only 1 influencer placed in x=1 and there are no personalization an feedback
def compute_x_max(a_p, b_p, c_p, z_p):
    xi_p = 1
    return (1/(1-b_p))*(a_p*z_p + c_p*xi_p)


# vector with the middle value of ech bin (corresponds to user opinion)
def idx_to_val(ni_p):
    return [(1+2*i)/(2*len(ni_p)) for i in range(len(ni_p))]


# ni_p : list of users in each bin (distribution of the population)
# psi_p: pointer to the psi function
# Provides a lambda function with xi as parameter, each bin is a delta with ni
# users (mid bin opinion), and it is a addend of this objective function
def obj_func(ni_p, psi_p, params_p, recall_p):
    addends = []          # obj is the sum of the contribution of each bin
    xb = idx_to_val(ni_p) # central interval values (as a list)

    for i, g_c in ni_p.items():
        for j in range(len(g_c)): # loop over group counts

            addends.append(lambda xi, n=i, m=j, k=g_c: 
                k[m]*psi_p(abs(xi-xb[n]),*params_p)*abs(xi_target-(a*args.prejudice[m]+b*xb[n]+c*xi)) if not recall_p
                else
                    k[m]*psi_p(abs(xi-xb[n]),*params_p)*abs(xi_target-(a*args.prejudice[m]+b*xb[n]+c*xi)) + 
                    (k[m]-k[m]*psi_p(abs(xi-xb[n]),*params_p))*abs(xi_target-((a/(a+b))*args.prejudice[m]+(b/(a+b))*xb[n]))
            )

    return lambda xi: sum(f(xi) for f in addends)


def find_x_opt(ni_p, psi_p, params_p, recall_p):
    grid = slice(1/(2*args.intervals), 1, 1/args.intervals) # finish=None to restric the search on the grid
    return optimize.brute(obj_func(ni_p, psi_p, params_p, recall_p), (grid, ), finish=None) # global min


def save_data(avg_time_p, xi_time_p, x_user_time_p, filename):
    save_dir = 'res'
    if not os.path.isdir(save_dir): # create folder if not already present
        os.mkdir(save_dir)

    df = pd.DataFrame({'avg_x':avg_time_p, 'xi':xi_time_p})
    df.to_csv(os.path.join(save_dir, filename + '.csv'))
    if not args.call:
        print(f'influencer data SAVED in csv format in {save_dir}')

    n_list = [i for i in range(len(x_user_time_p))] # we can assign instants list
    df_x = pd.DataFrame({'n': n_list, 'bin_count': x_user_time_p})
    df_x.to_csv(os.path.join(save_dir, filename + '_distrib' + '.csv'))
    if not args.call:
        print('users data SAVED in csv format in {save_dir}')


def strategic_xi(type_p, ni_p, psi_p, params_p):
    if type_p == 'greedy':
        # determine the strategy with highest payoff during the current iteration
        xi_v = find_x_opt(ni_p, psi_p, params_p, args.recall)
        if not args.call:
            print('step-OPTIMAL x is {:.5f}'.format(xi_v))
        return xi_v
    
    elif type_p == 'extreme':
        return (1+2*(n_bin-1))/(2*n_bin) # defined as the max middle value

    else:
        sys.exit(f'FATAL ERROR: unsupported simulation type {args.type}')


# return a pointer to the psi function
# 'd' represents the distance in opinion d=abs(x^u-x^i)
# 'a', 'b' are the characteristic parameters of funcs, two are sufficient fo all
def psi_function(B_p):
    epsilon = (1.0/(2.0*B_p))*0.001
    return lambda d, p1, p2: p1 if d < p2+epsilon else 0


def check_params(params_p):
    if params_p[0] < 0 or params_p[0] > 1:
        sys.exit(f'ERROR: a passed {params_p[0]}, must be in [0,1]')
    if params_p[1] < 0 or params_p[1] > 1:
        sys.exit(f'ERROR: b passed {params_p[0]}, must be in [0,1] for rect')


# the total number of users in each bin (irregardless of prejudice class)
def from_dict_to_hist(dict_p): # dict format: {k1(int):[n_z1, n_z2, ..]}
    return [sum(v) for _, v in dict_p.items()]


# returns the state as in the cpp simulator from the structure 'ni'
def from_ni_to_state(ni_p, p1_p):

    # find the index in the ni structure where the element is > 0
    d1, d2 = -1, -1
    B_p = len(ni_p)

    # ni: dict, keys=index of the opinion bin, value=user count per z group
    if p1_p == 1.0:
        if not any([len(counts) != 2 for _, counts in ni_p.items()]): 

            # |\psi| == 1 hence we always have only 2 groups (no splits)
            for bin_idx, two_groups_counts in ni_p.items(): # loop over opinion bins
                if two_groups_counts[0] > 0:
                    d1 = bin_idx
                    break # no other non-zero elements in the first group
                          # as \psi=1, the group does never split
            for bin_idx, two_groups_counts in ni_p.items():
                if two_groups_counts[1] > 0:
                    d2 = bin_idx
                    break # no other non-zero elements in the second group
            
            return d1*B_p + d2 # states have integers identifiers in [0, B^2-1]
        else:
            sys.exit('ERROR: the state can be retrieved only for 2 z-groups')
    else:
        sys.exit('ERROR: the state can be retrieved only if |\psi|=1')



# global parameters (also used in the functions)
args = parse_args()

N_MAX = args.nmax
U_TOT = args.users
n_bin = args.intervals

a = args.alpha
b = args.beta
c = 1-a-b

xi_target = args.xi_target

if not args.call and args.verbose:
        print('INFO: influencer pulls to the right of [0,1]')

params = args.param # (psi) function parameters
check_params(params)
psi = psi_function(n_bin)
if args.verbose and not args.call:
    print(f'INFO: {args.type} psi function selected')

x_max = 0 # theoretical limit the population can reach subject to xi=1
for i, z_v in enumerate(args.prejudice):
    x_max += args.count[i]*compute_x_max(a,b,c,z_v) # count is normalized here


# user population initialization (dict), keys are the index of the opinion bin,
# the values are lists with user counts for each prejudice group (if the same
# value of prejudice is passed twice, those will be considered as two z-groups)
bin_indices = range(n_bin)
ni = {idx: [0]*len(args.prejudice) for idx in bin_indices}
for group in range(len(args.prejudice)): # loop over the prejudice groups 
    # allow for the initial opinion (x0) to be different from the prejudice (z)
    initial_bin_index = math.floor(n_bin*args.init_x[group])
    ni[initial_bin_index][group] = int(args.count[group]*U_TOT)

tot_user_count, max_group_count, idx_max_group = 0, 0, 0
for bin_index, counts in ni.items():
    tot_user_count += sum(counts)
    for e in [count for count in counts if count > max_group_count]:
        max_group_count, idx_max_group = e, bin_index
# add or remove excess users (due to roundings) to the group with highest count
user_count_difference = tot_user_count - U_TOT
if user_count_difference:
    idx_max = ni[idx_max_group].index(max(ni[idx_max_group]))
    ni[idx_max_group][idx_max] += \
        user_count_difference if user_count_difference > 0 else -user_count_difference

if not args.call and args.verbose:
    for bin_idx, counts in ni.items():
        for i, e in enumerate(counts):
            if e > 0: # plot only non-zero entries
                print('\bin {:5d}, z-group {:2d} had count {:6d}'.format(bin_idx, i, e))


x_avg = sum([(1/U_TOT)*z_c*((1+2*j)/(2*n_bin)) for j, g_counts in ni.items() for z_c in g_counts])
if not args.call:
    print('Initial average opinion {:.5f}'.format(x_avg))


avg_time, xi_time, state_time, x_user_time = [], [], [], []

for iter in range(N_MAX):

    xi = strategic_xi(args.type, ni, psi, params) # influencer (strategic) position
    xi_time.append(xi)                            # save the strategy

    avg_time.append(x_avg) # save the value of the average op value (starting point)
    state_time.append(from_ni_to_state(ni, params[0])) # save (user) state
    x_user_time.append(from_dict_to_hist(ni))          # save user distribution

    if not args.call:
        print('ITER {:6d} average opinion is {:.5f}'.format(iter, x_avg))

    ni_curr = copy.deepcopy(ni) # because we will be updating future values of it
                                # as we procede with the iterations of the cycle
    
    for bin_idx, g_counts in ni_curr.items():
        
        x = (1+2*bin_idx)/(2*n_bin) # bin middle opinion value

        for idx_z, z_group_count in enumerate(g_counts):

            z = args.prejudice[idx_z]
            ni_z_move = math.ceil(z_group_count*psi(abs(xi-x), *params))

            if args.verbose and z_group_count > 0 and not args.call:
                print('bin {:5d}, z-group {:2d} had count {:6d} of which {:6d} are moving'.format(
                    bin_idx, idx_z, z_group_count, ni_z_move))
                print('z:', z, 'cz', z_group_count, 'x:', x, 'xi:', xi, 'psi', psi(abs(xi-x), *params))

            x_t = a*z + b*x + c*xi
            bin_idx_x_t = math.floor(n_bin*x_t)
        
            # update the ni list with the users who just moved
            ni[bin_idx][idx_z]     = ni[bin_idx][idx_z]     - ni_z_move
            ni[bin_idx_x_t][idx_z] = ni[bin_idx_x_t][idx_z] + ni_z_move


            if args.recall: # those not reached go back to their prejudice z

                ni_recall = z_group_count - ni_z_move # all users not reached by a post

                a_norm = a / (a+b)
                b_norm = b / (a+b)
                x_b = a_norm*z + b_norm*x # new opinion towards the prejudice

                bin_idx_x_b = math.floor(n_bin*x_b)

                ni[bin_idx][idx_z]     = ni[bin_idx][idx_z]     - ni_recall # empty the bin
                ni[bin_idx_x_b][idx_z] = ni[bin_idx_x_b][idx_z] + ni_recall

    
    # average opinion of the overall population (all groups)
    x_avg = sum([(1/U_TOT)*z_c*((1+2*j)/(2*n_bin)) for j, g_counts in ni.items() for z_c in g_counts])

# We have N+1 states an N actions, add the state to which the last action led
avg_time.append(x_avg)
state_time.append(from_ni_to_state(ni, params[0]))


if not args.call:
    print('average final opinion {:.5f} and max possible {:.5f}\n'.format(x_avg, x_max))
    if abs(x_avg-x_max) > 10*(1/n_bin): # arbitrary threshold
        print('WARNING: the final average opinion is far from the theoretical limit')


# save the temporal sequence of the mean population opinion in csv file
if args.save:
    defualt_name = args.type + '_rect_' + str(params[0]) + '_' + str(params[1])
    save_data(avg_time, xi_time, x_user_time, defualt_name)

print(f"{xi_time}\t{avg_time}\t{state_time}") # communicate with calling script
