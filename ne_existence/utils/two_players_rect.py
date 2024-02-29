# Two players' game when psi (the influence function) is a 0-1 function, in this
# case the prejudice-groups of users never split and thus to retain the state it
# is not necessary to discretize the opi nion space.

# ASSUMPTION: the players have "target" opinion \in \{0,1\}, without loss of
#             generality we set x^{(i;0)} = 0 and x^{(i;1)} = 1

# \psi is forced to be a rectangular function and needs to have amplitude 1.0
# to avoid that the delta groups of users split (exponential growth of splits)


import sys
import argparse
import numpy as np
import copy


def parse_args():

    parser = argparse.ArgumentParser(description=
        "Find Nash Equilibria in pure strategy of the two-players game."
    )
    parser.add_argument(
        '-z', 
        '--prejudice', 
        metavar="e1 .. eN", 
        type=float,
        nargs='*', 
        default=[], 
        help='Prejudice of the user groups (delta)'
    ) 
    parser.add_argument(
        '-x', 
        '--init-x', 
        metavar="e1 .. eN", 
        type=float, 
        nargs='*', 
        default=[], 
        help='Initial opinion of the user groups (delta)'
    )
    parser.add_argument(
        '-r', 
        '--rho', 
        metavar="e1 .. eN", 
        type=float, 
        nargs='*', 
        default=[], 
        help='Proportion of users in each group'
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
        default=10, 
        help='Maximum number of posts'
    )
    parser.add_argument(
        '-d0', 
        type=float, 
        default=0.1, 
        help='Opinion delta for player 1 (x^T=0)'
    )
    parser.add_argument(
        '-d1', 
        type=float, 
        default=0.1, 
        help='Opinion delta for player 2 (x^T=1)'
    )
    parser.add_argument(
        '-w', 
        '--width', 
        type=float, 
        default=0.7, 
        help='Value of the double-sided width of rectangular psi'
    )
    # flags
    parser.add_argument(
        '-v',
        '--verbose', 
        default=False, 
        action='store_true', 
        help='Detailed program output'
        )
    parser.add_argument(
        '--no-recall', 
        default=False, 
        action='store_true', 
        help='If set users DO NOT self-think when not reached'
    )
    parser.add_argument(
        '--strict', 
        default=False, 
        action='store_true', 
        help='Enforces to satisfy the strict version of assumption 3'
    )
    parser.add_argument(
        '--existence', 
        default=False, 
        action='store_true', 
        help='Reduces the output to i (int) = # of NE'
    )

    args = parser.parse_args()


    # checks on the user initialization parameters and setting defaults values

    if args.prejudice == []: # default z args at the extreme of [0,1]; 2 groups
        args.prejudice = [0.0, 1.0]

    if args.rho == []:       # defualt: balanced two groups
        args.rho = [0.5, 0.5]
    
    if len(args.rho) != len(args.prejudice): # must have same # of groups
        sys.exit('ERROR: group specification mismatch: %d z and %d rhos'
                    %(len(args.prejudice), len(args.rho)))
    
    if args.init_x == []: # default, x(0) = z
        args.init_x = copy.deepcopy(args.prejudice)
    
    if len(args.init_x) != len(args.prejudice): # check group number
        sys.exit('ERROR: group specification mismatch: %d x init and %d z'
                    %(len(args.init_x), len(args.prejudice)))

    if abs(1-sum([v for v in args.rho])) > 1e-8:
        sys.exit('ERROR: rho values need to sum to 1, got %f'
                    %sum([v for v in args.rho]))
    
    if any([v<0.0 or v>1.0 for v in args.prejudice]):
        sys.exit('ERROR: all values of prejudice must be in [0,1]')

    if args.d0<0.0 or args.d0>1.0 or args.d1<0.0 or args.d1>1.0:
        sys.exit('ERROR: all deltas must be in [0,1]')

    if args.existence:       # print just the Nash Eq. (red by another script)
        args.verbose = False # force no verbose output

    return args


def print_info(args_p): # print some info of the simulator for debug purposes
    print('users group specification:')
    for i_z, z_v in enumerate(args_p.prejudice):
        print('z-group with z={:.3f}, x0={:.3} and r={:.3}'.format(
            z_v, args_p.init_x[i_z], args_p.rho[i_z])
        )


# convex combination, to be used when a user is subject to one or both players
def update_x(alpha_p, beta_p, z_p, xn_p, xi_p):
    return alpha_p*z_p + beta_p*xn_p + (1-alpha_p-beta_p)*xi_p


# users go back to their prejudice when no reached by any post
def self_think(alpha_p, beta_p, z_p, xn_p):
    return (alpha_p / (alpha_p+beta_p))*z_p + (beta_p / (alpha_p+beta_p))*xn_p
    

def find_all_strategies(N_p): # list of all possible strategies (solutions space)
    pairs = []
    for i in range(N_p+1):
        for j in range(N_p+1):
            pairs.append((i,j))
    return pairs


# receives a np array (bidimensional) which represents the matrix of the pay-
# offs for both players (i.e., [payoff_player_0, payoff_player_1]) for each of 
# the strategies combinations. The strategies space is S^2, where S={ 0,1,..,N }
def find_nash_equilibria(payoffs_matrix):

    # we extract the payoffs of the single players as np matrices
    payoffs_0 = np.array([[x[0] for x in row] for row in payoffs_matrix])
    payoffs_1 = np.array([[x[1] for x in row] for row in payoffs_matrix])

    # for player 0 the best responses are the argmax of the values in each column
    # of the payoff matrix (since each column corresponds to an action of player 1)
    best_responses_0 = np.argmax(payoffs_0, axis=0)
    if args.verbose: print('best responses player 0', best_responses_0)
    candidate_pair_0 = [(BR_0, i) for i, BR_0 in enumerate(best_responses_0)]
    if not args.existence:
        print(candidate_pair_0)

    # similarly for player 1 we do the argmax on the rows
    best_responses_1 = np.argmax(payoffs_1, axis=1)
    if args.verbose: print('best responses player 1', best_responses_1)
    candidate_pair_1 = [(i, BR_1) for i, BR_1 in enumerate(best_responses_1)]
    if not args.existence:    
        print(candidate_pair_1)

    # I have to take the INTERSECTION, if not empty, of the pairs of indices
    # specified by 'candidate_pair_0' and 'candidate_pair_0'
    # if there are some MUTUAL best responses we have NASH EQUILIBRIA
    intersection = [p for p in candidate_pair_0 if p in candidate_pair_1]

    return intersection


# same method as above which considers also possible ties in terms of payoffs in the
# payoff matrices, thus the best responses may have multiple actions for the same
# action of the other player
def find_nash_equilibria_with_ties(payoffs_matrix):

    # we extract the payoffs of the single players as np matrices
    payoffs_0 = np.array([[x[0] for x in row] for row in payoffs_matrix])
    payoffs_1 = np.array([[x[1] for x in row] for row in payoffs_matrix])

    # for player 0 the best responses are the argmax of the values in each column of
    # the payoff matrix (since each column corresponds to an action of player 1) to
    # consider ties I apply the max function and then find all corresponding indices

    best_responses_0 = [np.where(col == np.max(col))[0].tolist() for col in payoffs_0.T]
    if args.verbose: print('best responses player 0', best_responses_0)
    candidate_pair_0 = []
    for i, BR_0 in enumerate(best_responses_0):
        for br_0 in BR_0:
            candidate_pair_0.append((br_0, i))
    if not args.existence:
        print(candidate_pair_0)

    # similarly for player 1 we do the argmax on the rows
    best_responses_1 = [np.where(row == np.max(row))[0].tolist() for row in payoffs_1]
    if args.verbose: print('best responses player 1', best_responses_1)
    candidate_pair_1 = []
    for i, BR_1 in enumerate(best_responses_1):
        for br_1 in BR_1:
            candidate_pair_1.append((i, br_1))
    if not args.existence:    
        print(candidate_pair_1)

    # I have to take the INTERSECTION, if not empty, of the pairs of indices
    # specified by 'candidate_pair_0' and 'candidate_pair_0'
    # if there are some MUTUAL best responses we have NASH EQUILIBRIA
    intersection = [p for p in candidate_pair_0 if p in candidate_pair_1]

    return intersection


# rectangular function, takes the absolute value of the opinion dist as param
def psi_rect(d_p, w_p):
    return 1.0 if d_p <= w_p else 0.0


# checks the (weaker) version of assumption 3: "when posting the targeting opi-
# nion each player DOES NOT reach ALL GROUPS. When posting the exploring opinion
# the player should reache AT LEAST one group".
# or strict version if 'strict_p' is set (i.e., reach both groups in exploration)
# NOTE: the check is done with respect to the init x (normally we have init_x=z)
def check_on_w(groups_p, delta_0_p, delta_1_p, w_p, strict_p):
    # first check: both players should not reach the furthest group when posting
    # their target opinion -> they don't reach all the user groups (trivial case)
    g_furthest_from_0 = max([v[0] for k, v in groups_p.items()])
    g_furthest_from_1 = min([v[0] for k, v in groups_p.items()])

    if w_p > g_furthest_from_0 or w_p > 1 - g_furthest_from_1:
        sys.exit(
            'ERROR: a player reaches all population at n=0 from the target opinion'
        )
    
    if strict_p:
        # stricter check: both groups need to be reached in exploration
        g_exp_furthest_from_0 = max([abs(v[0]-(delta_0_p)) for k, v in groups_p.items()])
        g_exp_furthest_from_1 = max([abs(v[0]-(1-delta_1_p)) for k, v in groups_p.items()])
        if w_p < g_exp_furthest_from_0 or w_p < g_exp_furthest_from_1:
            sys.exit(
                'ERROR: a player does not reach both groups in exploration'
            )
    else:
        # second check: the players should reach at least one group in exploration
        g_exp_closest_from_0 = min([abs(v[0]-(delta_0_p)) for k, v in groups_p.items()])
        g_closest_exp_from_1 = min([abs(v[0]-(1-delta_1_p)) for k, v in groups_p.items()])
        if w_p < g_exp_closest_from_0 or w_p < g_closest_exp_from_1:
            sys.exit(
                'ERROR: a player does not reach any group in exploration'
            )
    return True


def update_group(x_g, z_g, xi_0_p, xi_1_p, w_p, alpha_p, beta_p):

    d_group_from_0 = abs(x_g - xi_0_p) # distance of the group from x^{(i;0)}
    d_group_from_1 = abs(x_g - xi_1_p) # from x^{(i;1)}

    if psi_rect(d_group_from_0, w_p) and psi_rect(d_group_from_1, w_p): # both players reach group
        x_eff = (xi_0_p+xi_1_p) / 2.0
        return update_x(alpha_p, beta_p, z_g, x_g, x_eff)

    elif psi_rect(d_group_from_0, w_p): # only player 0 can influence
        return update_x(alpha_p, beta_p, z_g, x_g, xi_0_p)
    
    elif psi_rect(d_group_from_1, w_p): # only player 1 can influence
        return update_x(alpha_p, beta_p, z_g, x_g, xi_1_p)

    else: # none of the players reaches the group
        # NOTE 1: to be in this situation in the two players setting is rare because in a 
        # timeslot the groups are either subject to the influence of one influencer or both,
        # so they actually do NOT perform self-thinking
        return self_think(alpha_p, beta_p, z_g, x_g)




if __name__ == '__main__':

    args = parse_args()

    alpha = args.alpha
    beta  = args.beta

    z_groups  = args.prejudice
    r_groups  = args.rho       # the relative weight of each prejudice group
                               # this is relevant only when we will compute the mean
    x0_groups = args.init_x    # initial opinion value

    N = args.nmax

    w = args.width

    delta_0 = args.d0
    delta_1 = args.d1

    if args.no_recall:
        sys.exit('FATAL ERROR: the no-recall behaviour is not implemented yet (update line 308)')

    # data format: key->group id: value->(current_group_op, group_prejudice, rho)
    groups = {id_g: [x0_groups[id_g], z, r_groups[id_g]] for id_g, z in enumerate(z_groups)}
    check_on_w(groups, delta_0, delta_1, w, args.strict)

    if args.verbose:
        for g, g_x0 in groups.items():
            print('z-group {:2d} has x: {:.3f} and z: {:.3f} with {:.3f} proportion'.format(
                g, g_x0[0], g_x0[1], g_x0[2]))

    all_strategies = find_all_strategies(N)

    game_matrix = np.empty((N+1, N+1), dtype=list)

    for s_pair in all_strategies: # loop over all the possible strategies
        
        row_idx, col_idx = s_pair     # the strategies coincide with matrix's indices

        m = s_pair.index(min(s_pair)) # player that first ends the exploratory phase
        M = (0 if m else 1)           # the other
        
        tmp_groups = copy.deepcopy(groups) # non-updated version for this cycle

        for n in range(0, N):
            xi_0 = -1000 # init
            xi_1 = -1000

            # FIRST PHASE: both players in exploration
            if n < s_pair[m]:
                xi_0 = delta_0
                xi_1 = 1 - delta_1
                for id_g, g in tmp_groups.items(): # elements are (curr_opinion, prejudice)
                    g[0] = update_group(g[0], g[1], xi_0, xi_1, w, alpha, beta)

            # SECOND PHASE: m finishes exploration and M continues
            elif s_pair[m] <= n < s_pair[M]:
                if M: # player 1 continues the exploratory phase
                    xi_0 = 0
                    xi_1 = 1 - delta_1
                else:
                    xi_0 = delta_0
                    xi_1 = 1
                for id_g, g in tmp_groups.items():
                    g[0] = update_group(g[0], g[1], xi_0, xi_1, w, alpha, beta)

            # LAST PHASE: both players in targeting
            else:
                xi_0 = 0
                xi_1 = 1
                for id_g, g in tmp_groups.items():
                    g[0] = update_group(g[0], g[1], xi_0, xi_1, w, alpha, beta)
            
        # I need to weight the groups by their proportion (rho)
        ex = np.sum([v[0]*v[2] for _, v in tmp_groups.items()]) # average opinion value

        game_matrix[row_idx, col_idx] = np.array([1-ex, ex])

    if args.verbose:
        print('The payoff matrix is: ')
        for row in game_matrix:
            for el in row:
                print('({:.6f},{:.6f})'.format(el[0], el[1]), end='')
            print()
        print()

    ne_set = find_nash_equilibria_with_ties(game_matrix)
    if args.existence:
        print(ne_set) # list of Nash equilibria (may be empty)
    else:
        if ne_set != []:
            print()
            for ne in ne_set:
                print('\tNE: {} whose payoffs are u0:{:.3f} and u1:{:.3f}\n'.format(
                    ne, game_matrix[ne][0], game_matrix[ne][1])
                )
            for k, v in tmp_groups.items():
                print(v)
        else:
            print('\nthere exist no Nash Equilibria for the game')
