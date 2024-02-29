import utils.cmdargs as cmd
import utils.helpers as hlp
import utils.iofile as iof
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm


if __name__ == "__main__":

    args = cmd.parse_args()

    N, B, U = args.nmax, args.intervals, args.users

    alpha, beta = args.alpha, args.beta

    # each value in these lists characterizes one group of users (z, x0, rho)
    z_group_vals, rho_group_vals = args.prejudice, args.rho # z, rho vals per group
    x0_group_vals = args.init_x                             # corresponding x0
    num_groups = len(z_group_vals)

    delta_0, delta_1 = args.d0, args.d1

    if args.verbose:
        cmd.print_info(args)

    # User population data structure (discretized):
    # (dict) key  : integer index of the opinion bin, 
    #        value: (list) NUMBER of users that are in bin_key for each z-group;
    #               element l[i] is the count of users with z=z_vals[i] that are
    #               in bin bin_key at the given time instant
    bin_z_groups_count = {key: [0]*len(z_group_vals) for key in range(B)}

    for z_group in range(num_groups): # loop over each group
       
        bin_x0 = hlp.x_to_bin(x0_group_vals[z_group], B) # bin of x0
        
        # the index (z_group) in the list identifies the corresponding z_val
        bin_z_groups_count[bin_x0][z_group] = math.floor(rho_group_vals[z_group]*U)

    # add or remove excess users (due to roundings, up to # group, i.e. B for
    # defaults) start from middle bin and fill the one to the left and also that
    # to the right, otherwise put the last user in middle bin - preserve simmetry
    u_partial = sum([sum(g_count) for k, g_count in bin_z_groups_count.items()])
    remaining = U - u_partial

    if remaining > 0:
        middle_bin = int((B-1)/2)
        first_non_zero_idx = bin_z_groups_count[middle_bin].index(next(
            x for x in bin_z_groups_count[middle_bin] if x != 0
        ))
        bin_z_groups_count[middle_bin][first_non_zero_idx] += 1
        remaining -= 1
        idx_left, idx_right = middle_bin, middle_bin
        while (remaining > 1):
            l_left  = bin_z_groups_count[idx_left-1]
            l_right = bin_z_groups_count[idx_right+1]
            first_left  = l_left.index(next(x for x in l_left if x != 0))
            first_right = l_right.index(next(x for x in l_right if x != 0))
            l_left[first_left] += 1
            l_right[first_right] += 1
            idx_left, idx_right = idx_left-1, idx_right+1
            remaining -= 2
        if remaining == 1:
            bin_z_groups_count[middle_bin][first_non_zero_idx] += 1

    if args.verbose:
        fig, ax = plt.subplots()
        ax.plot([sum(v) for k,v in bin_z_groups_count.items()], marker='o')
        ax.set_xlabel(r'x^{(u)}')
        ax.set_ylabel('Count')
        plt.show()

    init_distrib = [sum(gl) for _, gl in bin_z_groups_count.items()]


    #***** Compute the payoff matrix for the game and additional measures ******

    game_matrix = np.empty((N+1, N+1), dtype=list)
    pop_matrix  = np.empty((N+1, N+1), dtype=list)
    x_N_matrix  = np.empty((N+1, N+1), dtype=list)

    n_repeat = 50 # number of posts one has to commit to when chosing the strategy

    for strategy_pair in tqdm(hlp.find_all_strategies(N)): # loop over all strategies
        row_idx, col_idx = strategy_pair # strategies coincide with matrix idx

        # determine the player who first ends the exploratory phase
        m = strategy_pair.index(min(strategy_pair))
        M = (0 if m else 1)

        pop0, pop1 = 100, 100 # init popularity, cumulative over the time
        
        s_bin_z_groups = copy.deepcopy(bin_z_groups_count) # init distrib

        for n in range(0, N): # unfold the dynamics over time
            # copy to avoid use updated (t+1) vals instead of non-updated (t)
            old_groups = copy.deepcopy(s_bin_z_groups) 

            theta_tot_0, theta_tot_1 = 0, 0 # total feedback at each instant

            # FIRST PHASE: both players are in exploration phase
            if n < strategy_pair[m]:
                xi_0, xi_1 = delta_0, 1 - delta_1

                for ii in range(n_repeat):

                    for bin_k, bin_info in old_groups.items(): # loop over opinion bins
                        for z_idx, z_count in enumerate(bin_info):
                            if z_count > 0: # if there are z-group users in 'bin_k'
                                # pass the original structure (to be updated)
                                feedback_0, feedback_1 = hlp.update_opinion(
                                    s_bin_z_groups, bin_k, z_idx, xi_0, xi_1, pop0, pop1, args
                                )
                                # update the aggregated FEEDBACK
                                theta_tot_0 += feedback_0
                                theta_tot_1 += feedback_1
                    
                    # update the popularity with the total feedback
                    if args.norm:
                        norm_c = U**2
                        pop0, pop1 = pop0 + theta_tot_0/norm_c, pop1 + theta_tot_1/norm_c
                    else:
                        pop0, pop1 = pop0 + theta_tot_0, pop1 + theta_tot_1 # no normalize

            # SECOND PHASE: m finishes exploration and starts tageting, M continues
            elif strategy_pair[m] <= n < strategy_pair[M]:

                xi_0, xi_1 = delta_0, 1 # if M is player 0, 0 continues exploration
                if M:                   # if M is player 1, 1 continues exploration
                    xi_0, xi_1 = 0, 1 - delta_1

                for ii in range(n_repeat):

                    for bin_k, bin_info in old_groups.items(): # loop over opinion bins
                        for z_idx, z_count in enumerate(bin_info):

                            if z_count > 0: # if there are z-group users in 'bin_k'
                                # pass the original structure (to be updated)
                                feedback_0, feedback_1 = hlp.update_opinion(
                                    s_bin_z_groups, bin_k, z_idx, xi_0, xi_1, pop0, pop1, args
                                )
                                # update the aggregated FEEDBACK
                                theta_tot_0 += feedback_0
                                theta_tot_1 += feedback_1

                    # update the popularity with the total feedback
                    if args.norm:
                        norm_c = U**2
                        pop0, pop1 = pop0 + theta_tot_0/norm_c, pop1 + theta_tot_1/norm_c
                    else:
                        pop0, pop1 = pop0 + theta_tot_0, pop1 + theta_tot_1 # no normalize

            # LAST PHASE: both players are in targeting phase
            else:
                xi_0, xi_1 = 0, 1

                for ii in range(n_repeat):

                    for bin_k, bin_info in old_groups.items():
                        for z_idx, z_count in enumerate(bin_info):
                            if z_count > 0: # if there are z-group users in 'bin_k'
                                # pass the original structure (to be updated)
                                feedback_0, feedback_1 = hlp.update_opinion(
                                    s_bin_z_groups, bin_k, z_idx, xi_0, xi_1, pop0, pop1, args
                                )
                                # update the aggregated FEEDBACK
                                theta_tot_0 += feedback_0
                                theta_tot_1 += feedback_1

                    # update the popularity with the total feedback
                    if args.norm:
                        norm_c = U**2
                        pop0, pop1 = pop0 + theta_tot_0/norm_c, pop1 + theta_tot_1/norm_c
                    else:
                        pop0, pop1 = pop0 + theta_tot_0, pop1 + theta_tot_1 # no normalize


            tot_users = sum([sum(g_counts) for kb, g_counts in s_bin_z_groups.items()])
            if tot_users != args.users: # sanity check (TODO: comment it)
                sys.exit(f'FATAL ERROR: user number not conserved, {tot_users} instead of {args.users}')

        ex = hlp.population_mean(s_bin_z_groups, B, U) # average opinion value

        # the payoff is the mean x for player 1 and 1-(mean x) for player 0
        game_matrix[row_idx, col_idx] = np.array([1-ex, ex])
        pop_matrix[row_idx, col_idx]  = [pop0, pop1]          # final popularity
        x_N_matrix[row_idx, col_idx]  = s_bin_z_groups.copy() # final distribution


if args.verbose:
    print('The payoff matrix is: ')
    for row in game_matrix:
        print(' '.join('({:.6f},{:.6f})'.format(el[0], el[1]) for el in row))
    print()


# *************** Compute Nash Equilibria with the payoff matrix ***************

ne_set = hlp.nash_equilibria_with_ties(game_matrix, args)

if args.existence:
    print(ne_set) # list of Nash equilibria (may be empty)
else:
    if ne_set != []:
        x_N_ne_list, pop_ne_list, payoff_mat_list = [], [], []
        payoff_row0_list, payoff_col0_list = [], []
        for ne in ne_set:
            print('\tNE: {} with payoff0:{:.4f} and u1:{:.4f}\n'.format(
                ne, game_matrix[ne][0], game_matrix[ne][1]
            ))
            x_N_ne_list.append(x_N_matrix[ne])
            pop_ne_list.append(pop_matrix[ne])
            payoff_mat_list.append(game_matrix[ne].tolist())

            # first row and column of the entire payoff matrix
            payoff_row0_list.append(game_matrix[0,:].tolist())
            payoff_col0_list.append(game_matrix[:,0].tolist())

        # save all info (results as lists)
        iof.create_or_append_csv(args, ne_set, x_N_ne_list, init_distrib, pop_ne_list,
                                 payoff_mat_list, payoff_row0_list, payoff_col0_list)
    
    else: # signal that there exist no Nash equilibria
        iof.create_or_append_csv(args, ne_set, [], init_distrib, [], [],
                                 [game_matrix[0,:]], [game_matrix[:,0]])
        
        print('\nINFO: there exist NO Nash Equilibria for the game')