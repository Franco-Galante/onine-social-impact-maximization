import sys
import math
import warnings
import numpy as np


def psi_function(str_psi, loop_type, param_p):
    """
    Select the lambda function for psi.

    Defines the lambda function to be used as the psi function, according to the
    specified command line parameter. The psi function is made up by two factors
    the theta (feedback) and omega (proposition) function. The returned function
    can have 3 or 4 parameters depending if the script is for the open or closed
    loop. Check for the fixed user defined parameters (pw and pt).

    Parameters:
    - str_psi  : Shape of psi, to choose between 'rect', 'exp_lin', 'exp_exp'.
    - loop_type: If opinion update is open (False) or closed loop (True).

    Returns:
    Lambda function for the psi function.
    """

    if param_p[0] < 0 or param_p[0] < 0:
        pw, pt = param_p[0], param_p[1]
        sys.exit(f'ERROR: pw and pt must be positive, got pw={pw} pt={pt}')

    if str_psi == 'rect':
        if param_p[1] < 0 or param_p[1] > 1:
            pt = param_p[1]
            sys.exit(f'ERROR: for {str_psi} pt needs to be in [0,1], got {pt}')

    elif str_psi == 'exp_lin':
        if param_p[1] > 10:
            warnings.warn('Exponent parameter pw very high', UserWarning)

    elif str_psi == 'exp_exp':
        if param_p[0] > 10 or param_p[1] > 10:
            warnings.warn('Exponent parameter pw or pt very high', UserWarning)

    # define the lambda function for psi according tot the specifications
    if loop_type: # closed loop, functions of opinion distance (d), parameter
                  # for omega function (pw), for theta function(pt) and normali-
                  # zed popularity of the posting influencer i

        if str_psi == 'rect':
            return lambda d, pw, pt, pi_i: (
                pt if d <= (pw*2*pi_i) else 0
            )
        
        elif str_psi == 'exp_lin':
            return lambda d, pw, pt, pi_i: (
                math.exp(-pw*((d**2) / pi_i)) * max((1.0 - pt*abs(d)), 0)
            )
        
        elif str_psi == 'exp_exp':
            return lambda d, pw, pt, pi_i: (
                math.exp(-pw*((d**2) / pi_i)) * math.exp(-pt*(d**2))
            )
        
        elif str_psi == 'exp_exp_pow':
            return lambda d, pw, pt, pi_i: (
                math.exp(-pw*((d**2) / (pi_i**4))) * math.exp(-pt*(d**2))
            )
        
        else:
            sys.exit(f'FATAL ERROR: {str_psi} psi not supported')
        
    else: # open loop, same a above but whitout popularity

        if str_psi == 'rect':
            return lambda d, pw, pt: pt if d <= pw else 0

        elif str_psi == 'exp_lin':
            return lambda d, pw, pt: (
                math.exp(-pw*(d**2)) * max((1.0 - pt*abs(d)), 0)
            )
        
        elif str_psi == 'exp_exp':
            return lambda d, pw, pt: (
                math.exp(-pw*(d**2)) * math.exp(-pt*(d**2))
            )

        elif str_psi == 'exp_exp_pow': # actually same as above (chenge only in pop)
            return lambda d, pw, pt: (
                math.exp(-pw*(d**2)) * math.exp(-pt*(d**2))
            )
        
        else:
            sys.exit(f'FATAL ERROR: {str_psi} psi not supported')


def find_all_strategies(N_p): # list of all possible strategies (solutions space)
    """
    Compute the list of all strategies given N_p.

    Parameters:
    - N_p: Time horizon in terms of posts considered by the players.

    Returns:
    List of all possible combinations of strategies of the two players.
    """
    return [(i, j) for i in range(N_p + 1) for j in range(N_p + 1)]


def x_to_bin(x_p, B_p):
    """
    Compute the id of the bin x_p is in.

    Parameters:
    - x_p: (real) opinion value.
    - B_p: Number of discrete bins.

    Returns:
    Integer id (starting from 0) of the bin x_p is in out of the B_p possible.
    """
    if x_p >= 1.0: # avoid the id to overflow to B+1 when x_p=1
        x_p = x_p - 10e-10

    return math.floor(B_p * x_p)


# returns the midpoint of the opinion bin indexed by 'bin_p'
def bin_to_x(bin_p, B_p):
    """
    Compute the middle point of bin 'bin_p'.

    Parameters:
    - bin_p: Integer id of the bin.
    - B_p  : Number of discrete bins.

    Returns:
    Float representing the middle value of the 'bin_p' bin.
    """
    return (1+2*bin_p)/(2*B_p)


# convex combination, to be used when a user is subject to one or both players
# also retruns the index of the bin the updated opinion is in
def compute_x(z_val, xn_val, xi_val, args_p):
    """
    Eq. (1), opinion update when psi=1.

    Parameters:
    - z_val : The prejudice of the group.
    - xn_val: The current value of  opinion at time n.
    - xi_val: The (possibly mean) influence from the influencer(s).
    - args_p: Command line parameters with the specifications of the experiment.

    Returns:
    Opinion bin index of the updated opinion value.
    """
    alpha, beta = args_p.alpha, args_p.beta
    B = args_p.intervals

    updated_x = alpha*z_val + beta*xn_val + (1-alpha-beta)*xi_val
    bin_updated_x = x_to_bin(updated_x, B)

    return bin_updated_x


# users go back to their prejudice when no reached by any post
def self_think(z_val, xn_val, args_p):
    """
    Second part of Eq. (1), when psi=0.

    Parameters:
    - z_val : The prejudice of the group.
    - xn_val: The current value of  opinion at time n.
    - args_p: Command line parameters with the specifications of the experiment.

    Returns:
    Opinion bin index of the updated opinion value through 'self-thinking'.
    """
    alpha, beta = args_p.alpha, args_p.beta
    B = args_p.intervals

    norm_c = alpha / (alpha+beta)
    updated_x = norm_c*z_val + (1-norm_c)*xn_val
    bin_updated_x = x_to_bin(updated_x, B)

    return bin_updated_x


def update_opinion(bin_z_group_p, bin_k_p, z_idx_p, xi_0_p, xi_1_p, pop0_p, pop1_p, args_p):
    """
    Updates the opinions of the population, Eq. (1).

    Parameters:
    - bin_z_group_p: The datastructure for the population by reference (dict).
    - bin_k_p      : The key (ID) of the bin where the z-group is in at n.
    - z_idx_p      : The index of the z-group in the structure.
    - xi_0_p       : Opinion of player 0.
    - xi_1_p       : Opinion of player 1.
    - pop0_p       : Popularity of player 0 at time n.
    - pop1_p       : Popularity of player 1 at time n.
    - args_p       : Command line param with specifications of the experiment.

    Returns:
    A pair with the feedback provided to each player by the users who moved.
    """
    B = args_p.intervals

    pw, pt = args_p.param
    psi = psi_function(args_p.psi, args_p.closed, args_p.param)

    x_val_bin = bin_to_x(bin_k_p, B)
    z_val_bin = args_p.prejudice[z_idx_p]
    z_count   = bin_z_group_p[bin_k_p][z_idx_p] # number of users in the group
    
    # distance of z-group users in bin_k (x=x_val_bin) from the two players
    d_from_0 = abs(xi_0_p - x_val_bin) # from player 0
    d_from_1 = abs(xi_1_p - x_val_bin) # from player 1

    psi_wrt_0, psi_wrt_1 = 0, 0
    if args_p.closed: # psi_function() ensures to pick the right shape
        psi_wrt_0 = psi(d_from_0, pw, pt, pop0_p / (pop0_p+pop1_p))
        psi_wrt_1 = psi(d_from_1, pw, pt, pop1_p / (pop0_p+pop1_p))

    else: # no popularity in the visibility function psi
        psi_wrt_0 = psi(d_from_0, pw, pt)
        psi_wrt_1 = psi(d_from_1, pw, pt)


    # the values of the psi function determine who influences the z-group
    num_both_influence   = math.floor((psi_wrt_0*psi_wrt_1) * z_count)
    num_only_0_influence = math.floor(psi_wrt_0*z_count) - num_both_influence
    num_only_1_influence = math.floor(psi_wrt_1*z_count) - num_both_influence
    num_self_thinkers    = z_count - (
        num_both_influence + num_only_0_influence + num_only_1_influence) # rest
    
    if num_self_thinkers < 0:
        sys.exit('FATAL ERROR: number of self-thinkers went negative')


    # SITUATION 1: psi_wrt_0*psi_wrt_1 fraction of z-group (influence from both players)
    xi_both = (xi_0_p + xi_1_p) / 2.0
    bin_x_both = compute_x(z_val_bin, x_val_bin, xi_both, args_p)

    # move the users from one location (bin) to the updated one
    bin_z_group_p[bin_k_p][z_idx_p]    = bin_z_group_p[bin_k_p][z_idx_p]    - num_both_influence
    bin_z_group_p[bin_x_both][z_idx_p] = bin_z_group_p[bin_x_both][z_idx_p] + num_both_influence


    # SITUATION 2: psi_wrt_0 - psi_wrt_0*psi_wrt_1 fraction (only player 0 influence)
    bin_x_only_0 = compute_x(z_val_bin, x_val_bin, xi_0_p, args_p)

    bin_z_group_p[bin_k_p][z_idx_p]      = bin_z_group_p[bin_k_p][z_idx_p]      - num_only_0_influence
    bin_z_group_p[bin_x_only_0][z_idx_p] = bin_z_group_p[bin_x_only_0][z_idx_p] + num_only_0_influence


    # SITUATION 3: psi_wrt_1 - psi_wrt_0*psi_wrt_1 fraction (only player 1 influence)
    bin_x_only_1 = compute_x(z_val_bin, x_val_bin, xi_1_p, args_p)

    bin_z_group_p[bin_k_p][z_idx_p]      = bin_z_group_p[bin_k_p][z_idx_p]      - num_only_1_influence
    bin_z_group_p[bin_x_only_1][z_idx_p] = bin_z_group_p[bin_x_only_1][z_idx_p] + num_only_1_influence
    

    # SITUATION 4 (1 - (psi_0+psi_1-psi_0*psi_1)) - self-thinking
    ## print('the self-thinking individuals are {}'.format(self_think_move))
    bin_x_self = self_think(z_val_bin, x_val_bin, args_p)

    bin_z_group_p[bin_k_p][z_idx_p]    = bin_z_group_p[bin_k_p][z_idx_p]    - num_self_thinkers
    bin_z_group_p[bin_x_self][z_idx_p] = bin_z_group_p[bin_x_self][z_idx_p] + num_self_thinkers

    # NOTE: a bin does not necessarily empty (if the Delta_x is tiny the group 
    #       actually doe snot move, and this procedure ends up doing nothing)


    # whomever moves provides a positive feedback, pass it to the calling func
    return num_both_influence+num_only_0_influence , num_both_influence+num_only_1_influence


def population_mean(bin_z_group_p, B_p, U_p):
    """
    Compute the mean opinion of the user population.

    Parameters:
    - bin_z_group_p: The datastructure for the population (dict).
    - B_p          : Number of discrete bins.
    - U_p          : Total number of users in the population. 

    Returns:
    A float for the mean opinion value of the population.
    """
    total_mean = 0
    for bin_k, bin_info in bin_z_group_p.items():
        bin_prob = sum(z_count for z_count in bin_info) / U_p
        total_mean += bin_prob * bin_to_x(bin_k, B_p)
    
    return total_mean


# receives a np array (bidimensional) which represents the matrix of the pay-
# offs for both players (i.e., [payoff_player_0, payoff_player_1]) for each of 
# the strategies combinations. The strategies space is S^2, where S={ 0,1,..,N }
# same method as above which considers also possible ties in terms of payoffs in the
# payoff matrices, thus the best responses may have multiple actions for the same
# action of the other player
def nash_equilibria_with_ties(payoffs_matrix, args_p):
    """
    Find Nash equilibria from payoff matrix.

    Given a matrix (bidemensional numpy array) with the payoffs for each of the
    two players, where rows and columns represent all possiblr strategies for
    each player (A^2 elements) first computes the best responses to each other
    strategy and then outputs the Nash equilibria based on them, if any.

    Parameters:
    - payoffs_matrix: AxA matrix with payoffs pairs for each strategy combination.
    - args_p        : Command line param with specifications of the experiment.

    Returns:
    A list of pairs (can be empty) representig the Nash equilibria of the game.
    """

    # we extract the payoffs of the single players as np matrices
    payoffs_0 = np.array([[x[0] for x in row] for row in payoffs_matrix])
    payoffs_1 = np.array([[x[1] for x in row] for row in payoffs_matrix])

    # for player 0 the best responses are the argmax of the values in each column of
    # the payoff matrix (since each column corresponds to an action of player 1) to
    # consider ties I apply the max function and then find all corresponding indices

    best_responses_0 = [np.where(col == np.max(col))[0].tolist() for col in payoffs_0.T]
    if args_p.verbose:
        print('\tbest responses player 0', best_responses_0)
    
    candidate_pair_0 = []
    for i, BR_0 in enumerate(best_responses_0):
        for br_0 in BR_0:
            candidate_pair_0.append((br_0, i))
    if not args_p.existence:
        print(candidate_pair_0)

    # similarly for player 1 we do the argmax on the rows
    best_responses_1 = [np.where(row == np.max(row))[0].tolist() for row in payoffs_1]
    if args_p.verbose:
        print('\tbest responses player 1', best_responses_1)

    candidate_pair_1 = []
    for i, BR_1 in enumerate(best_responses_1):
        for br_1 in BR_1:
            candidate_pair_1.append((i, br_1))
    if not args_p.existence:    
        print(candidate_pair_1)

    # I have to take the INTERSECTION, if not empty, of the pairs of indices
    # specified by 'candidate_pair_0' and 'candidate_pair_0'
    # if there are some MUTUAL best responses we have NASH EQUILIBRIA
    intersection = [p for p in candidate_pair_0 if p in candidate_pair_1]

    return intersection
