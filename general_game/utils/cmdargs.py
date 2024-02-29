import sys
import argparse
import numpy as np
from scipy.stats import beta

def parse_args():
    """
    Parse command line parameters.

    Parses the command line parameters and provides info about the main script,
    moreover performs a check over the parameters and sets the defaults.

    Returns:
    Returns an object from with all passed (or default) arguments are accessed.

    """

    parser = argparse.ArgumentParser(
        description="Find Nash Equilibria in pure strategy of two-players game."
    )

    parser.add_argument(
        '-z',
        '--prejudice',
        metavar="e1 .. eN",
        type=float,
        nargs='+',
        default=[0.25, 0.75], 
        help='Prejudice values list of user groups\n' + 
                '\t-1: radicalized Beta (a=b < 1)\n' +
                '\t-2: moderate Beta (a=b > 1)\n' +
                '\t-3: unbalanced Beta (a!=b, a<b)'
    )
    parser.add_argument(
        '-x',
        '--init-x',
        metavar="e1 .. eN",
        type=float,
        nargs='+',
        default=[0.25, 0.75], 
        help='Initial opinion values of user groups, hp. x_0=z'
    )
    parser.add_argument(
        '-r',
        '--rho',
        metavar="e1 .. eN",
        type=float,
        nargs='+',
        default=[0.5, 0.5], 
        help='Proportion (float) of users in each group'
    )
    parser.add_argument(
        '-f',
        '--fixed-avg',
        metavar="v",
        type=float,
        default=0.4, 
        help='Fixing the average of the distribution for -z -|9+|'
    )
    parser.add_argument(
        '-a',
        '--alpha',
        metavar="v",
        type=float,
        default=0.1, 
        help='1st coefficient in opinion update'
    )
    parser.add_argument(
        '-b',
        '--beta',
        metavar="v",
        type=float,
        default=0.5, 
        help='2nd coefficient in opinion update'
    )
    parser.add_argument(
        '-n',
        '--nmax',
        metavar="v",
        type=int,
        default=10, 
        help='Time horizon considered (# posts)'
    )
    parser.add_argument(
        '-d0',
        metavar="v",
        type=float,
        default=0.1, 
        help='Opinion shift in exploration for player 1 (x^T=0)'
    )
    parser.add_argument(
        '-d1',
        metavar="v",
        type=float,
        default=0.1, 
        help='Opinion shift in exploration for player 2 (x^T=1)'
    )
    parser.add_argument(
        '-i',
        '--intervals',
        metavar="v",
        type=int,
        default=101,
        help='Number of discrete intervals for the opinion space'
    )
    parser.add_argument(
        '-u',
        '--users',
        metavar="v",
        type=int,
        default=100000,
        help='Number of users for the discrete dynamics'
    )
    parser.add_argument(
        '-p',
        '--psi',
        metavar="id",
        type=str,
        choices=['rect', 'exp_lin', 'exp_exp', 'exp_exp_pow'],
        default='exp_lin', 
        help='Possible psi functions:\n' + 
                '\trect   : rectangular function\n' +
                '\texp_lin: product of exp proposition and linear feedback\n' +
                '\texp_exp: product of exp proposition and exp feedback '
    )
    parser.add_argument(
        '--param',
        metavar="v",
        type=float,
        nargs=2,
        default=[5.0, 1.0], # p[0]: omega, p[1]: theta
        help='Values of the two parameters controlling the shape of psi'
    )
    parser.add_argument(
        '--no-recall',
        default=False,
        action='store_true', 
        help='If set users DO NOT self-think when not reached'
    )
    parser.add_argument(
        '-v', '--verbose',
        default=False,
        action='store_true', 
        help='Detailed program output'
    )
    parser.add_argument(
        '--existence',
        default=False,
        action='store_true', 
        help='''Set by 'ne_existence.py', only list of NE printed to output '''
    )
    parser.add_argument(
        '--closed',
        default=False,
        action='store_true', 
        help='Closed loop scenario, with user feedback and popularity update'
    )
    parser.add_argument(
        '--norm',
        default=False,
        action='store_true', 
        help='Normalize total feedback when updating the popularity of players'
    )
    parser.add_argument(
        '--filename',
        metavar="root",
        type=str,
        default='fast_vs_slow', 
        help='Root of the filename where the results are written'
    )


    args = parser.parse_args()

    if len(args.prejudice) == 1: # 1-element-list is used to specify defaults
                                 # only 1 group population not interesting case
        
        args.prejudice, args.init_x, args.rho = beta_defaults(
            args.prejudice[0], args.intervals, args.fixed_avg)

    
    # ************************ Check parameters values *************************

    if not (len(args.prejudice) == len(args.init_x) == len(args.rho)):
        zl, x0l, rl = len(args.prejudice), len(args.init_x), len(args.rho)
        sys.exit(f'ERROR: groups mismatch: {zl} z, {x0l} x0, and {rl} rhos')

    rho_sum = sum([rv for rv in args.rho])
    if abs(1-rho_sum) > 1e-8:
        sys.exit(f'ERROR: rho values need to sum to 1, got {rho_sum}')
    
    if any([zv<0.0 or zv>1.0 for zv in args.prejudice]):
        sys.exit('ERROR: all prejudice values must be in [0,1]')

    if args.fixed_avg<0.0 or args.fixed_avg>1.0:
        sys.exit('ERROR: the average population opinion must be in [0,1]')

    if args.d0<0.0 or args.d0>1.0 or args.d1<0.0 or args.d1>1.0:
        sys.exit('ERROR: all influencers shifts must be in [0,1]')

    if args.intervals % 2 == 0:
        sys.exit('ERROR: for technical reasons the number of bins must be odd')

    if args.existence:       # print just the number of Nash Equilibria
        args.verbose = False # force the flag if necessary

    return args


def print_info(args_p):
    """
    Prints info for the current experiment.

    Parameters:
    - args_p: The parsed command line parameters

    """
    print(' experimetn info '.center(54, '='))
    print(f'\n\t{args_p.intervals} discrete bins for {args_p.nmax} time instants')
    loop_dict = {True: 'closed loop', False: 'open loop'}
    print(f'\n\tINFO: {args_p.psi} psi function and {loop_dict[args_p.closed]}')
    print('\tweights alpha={:.3f}, beta={:.3f}'.format(args_p.alpha, args_p.beta))
    
    print(('\nGroup specification:'))
    for i_z, z_v in enumerate(args_p.prejudice):
        print('\tz-group with z={:.3f}, x0={:.3} and r={:.3}'.format(
            z_v, args_p.init_x[i_z], args_p.rho[i_z])
        )


def discrete_beta(a_p, b_p, B_p):
    """
    Provide B_p samples of the Beta pdf.

    Parameters:
    - a_p: First parameter of the Beta distribution.
    - b_p: Second parameter of the Beta distribution.
    - B_p: The number of samples to retur.

    Returns:
    Returns two lisst of float with
     - (z_vals)   the B_p z-values the Beta pdf is sampled at,
     - (rho_vals) the B_p equispaced samples of the Beta pdf.

    """
    if a_p <= 0 or b_p <= 0:
        sys.exit('ERROR: the parameters of the Beta must be positve')

    x = np.linspace(0, 1, B_p+1)  # endpoints of the bins
    z_vals = (x[1:] + x[:-1]) / 2 # midpoints of the bins

    pdf_values = beta.pdf(z_vals, a_p, b_p)
    rho_vals = pdf_values / np.sum(pdf_values) # make a discrete probability
    
    return z_vals, rho_vals


def bimodal_beta(a1_p, b1_p, a2_p, b2_p, w1_p, B_p):
    """
    Samples a bimodal Beta distribution.

    Behaves similarly as 'discrete_beta'. The only difference is in that the
    distribution is a convex combination of samples from two Beta with a1_p,b1_p
    and a2_p,b2_p parameters respectively. Weighting the first Beta w1.

    Parameters:
    - a1_p: First parameter of the first Beta distribution.
    - b1_p: Second parameter of the first Beta distribution.
    - a2_p: First parameter of the second Beta distribution.
    - b2_p: Second parameter of the second Beta distribution.
    - w1_p: Weight of the samples from the first Beta.
    - B_p : The number of samples to retur.

    Returns:
     - (z_vals)   the B_p z-values the bimodal pdf is sampled at,
     - (rho_vals) the B_p equispaced samples of the bimodal pdf.

    """
    if w1_p < 0 or w1_p > 1:
        sys.exit('ERROR: the weight for the bimodal distrib must be in [0,1]')
    
    l1, l2 = discrete_beta(a1_p, b1_p, B_p) # 1st Beta samples
    r1, r2 = discrete_beta(a2_p, b2_p, B_p) # 2nd Beta samples

    z_vals   = [w1_p*x + (1-w1_p)*y for x,y in zip(l1, r1)]
    rho_vals = [w1_p*x + (1-w1_p)*y for x,y in zip(l2, r2)]
    
    return z_vals, rho_vals


def w_for_fix_avg(avg_p, x_beta1_p, pdf1_p, x_beta2_p, pdf2_p):
    """
    Compute parameter w to weight two Betas.

    Determines the value of in the convex combination of samples from two Betas
    with a1_p,b1_p and a2_p,b2_p so that to obtain a fixed average of 'avg_p'.

    Parameters:
    - avg_p    : Target average value for the overall distribution.
    - x_beta1_p: Samples at which the pdf of the first Beta has been evaluated.
    - pdf1_p   : Probability density function values of the first Beta.
    - x_beta2_p: Samples of the second Beta.
    - pdf2_p   : Probability density function of the second Beta.

    Returns:
     - w : the first weight of the convex combination between two Betas.

    """

    avg0 = sum([z*v for z,v in zip(x_beta1_p, pdf1_p)])
    avg1 = sum([z*v for z,v in zip(x_beta2_p, pdf2_p)])

    ret_w = (avg_p - avg1) / (avg0 - avg1)
    if ret_w < 0 or ret_w > 1:
        sys.exit("FATAL ERROR: cannot find w in [0,1] for a,b of two Betas")

    return ret_w


def bimodal_beta_fix_avg(a1_p, b1_p, a2_p, b2_p, B_p, fix_avg_p):
    """
    Samples a bimodal Beta distribution.

    Parameters:
    - a1_p     : First parameter of the first Beta distribution.
    - b1_p     : Second parameter of the first Beta distribution.
    - a2_p     : First parameter of the second Beta distribution.
    - b2_p     : Second parameter of the second Beta distribution.
    - w1_p     : Weight of the samples from the first Beta.
    - B_p      : The number of samples to retur.
    - fix_avg_p: Avergae value of the overall opinion distribution.

    Returns:
     - (z_vals)   the B_p z-values the bimodal pdf is sampled at,
     - (rho_vals) the B_p equispaced samples of the bimodal pdf.

    """
    l1, l2 = discrete_beta(a1_p, b1_p, B_p) # 1st Beta samples
    r1, r2 = discrete_beta(a2_p, b2_p, B_p) # 2nd Beta samples

    w1 = w_for_fix_avg(fix_avg_p, l1, l2, r1, r2)

    z_vals   = [w1*x + (1-w1)*y for x,y in zip(l1, r1)]
    rho_vals = [w1*x + (1-w1)*y for x,y in zip(l2, r2)]
    
    return z_vals, rho_vals


def beta_defaults(z_default_p, B_p, fix_avg_p):
    """
    Samples the Beta distribution.

    Define three lists according to the number of bins and the default chosen
    by the user: 'z_vals, x0_vals, rho_vals'. Those are respectively the mid-
    point of the discrete prejudice bin, the initial opinion, and the proportion
    of users with that prejudice and initial opinion. Hypothesis: z=x_0.

    Parameters:
    - z_default_p: Signals the type of default distribution.
    - B_p        : Number of discretization bins.
    - fix_avg_p  : (only for defaults > 9) average value of user distribution.

    Returns:
    Description of the return value, if applicable.

    Raises:
    Any exceptions that the function may raise.

    """

    print(f"\n\tINFO: using {z_default_p} of -z, ignoring -rho and -x-init\n")

    z_default_list, rho_default_list = [], []

    # radicalized distribution (Beta with horns)
    if z_default_p == -1:   
        beta_a, beta_b = 0.25, 0.25
        z_default_list, rho_default_list = discrete_beta(beta_a, beta_b, B_p)

    # moderate distribution (Gaussian-like), around 0.5
    elif z_default_p == -2: 
        beta_a, beta_b = 5, 5
        z_default_list, rho_default_list = discrete_beta(beta_a, beta_b, B_p)

    # Unbalanced distribution (mass skewed towards 0)
    elif z_default_p == -3: 
        beta_a, beta_b = 2, 4
        z_default_list, rho_default_list = discrete_beta(beta_a, beta_b, B_p)

    # bimodal, radicalized but very few users close to 1
    elif z_default_p == -4:
        w1 = 0.9
        beta_a1, beta_b1 = 2, 8
        beta_a2, beta_b2 = 14, 2
        z_default_list, rho_default_list = bimodal_beta(
            beta_a1, beta_b1, beta_a2, beta_b2, w1, B_p
        )
    
    # bimodal, less but more radicalized users towards 1
    elif z_default_p == -5:
        w1 = 0.6
        beta_a1, beta_b1 = 2, 8
        beta_a2, beta_b2 = 14, 2
        z_default_list, rho_default_list = bimodal_beta(
            beta_a1, beta_b1, beta_a2, beta_b2, w1, B_p
        )

    # bimodal, 'symmetric' Beta but weighting factor (strong)
    elif z_default_p == -6:
        w1 = 0.9
        beta_a1, beta_b1 = 2, 8
        beta_a2, beta_b2 = 8, 2
        z_default_list, rho_default_list = bimodal_beta(
            beta_a1, beta_b1, beta_a2, beta_b2, w1, B_p
        )

    # bimodal, 'symmetric' Beta but weighting factor (moderate)
    elif z_default_p == -7:
        w1 = 0.6
        beta_a1, beta_b1 = 2, 8
        beta_a2, beta_b2 = 8, 2
        z_default_list, rho_default_list = bimodal_beta(
            beta_a1, beta_b1, beta_a2, beta_b2, w1, B_p
        )
    
    # unbalanced distribution (similar to -3) but more polarized (higher beta)
    elif z_default_p == -8: 
        beta_a, beta_b = 2, 8
        z_default_list, rho_default_list = discrete_beta(beta_a, beta_b, B_p)


    # ******************* Init for the Fixed Mean Experiments ******************
    
    # Bimodal radicalized to 1 (min pdf point around 0.7)
    elif z_default_p == -9:

        beta_a1, beta_b1 = 1.5, 3
        beta_a2, beta_b2 = 5, 2        
        z_default_list, rho_default_list = bimodal_beta_fix_avg(
            beta_a1, beta_b1, beta_a2, beta_b2, B_p, fix_avg_p
        )
    
    # Bimodal radicalized to 1 (min pdf point around 0.75)
    elif z_default_p == -10:

        beta_a1, beta_b1 = 1.5, 3
        beta_a2, beta_b2 = 10, 2        
        z_default_list, rho_default_list = bimodal_beta_fix_avg(
            beta_a1, beta_b1, beta_a2, beta_b2, B_p, fix_avg_p
        )

    # Bimodal radicalized to 1 (min pdf point around 0.8)
    elif z_default_p == -11:

        beta_a1, beta_b1 = 1.5, 3
        beta_a2, beta_b2 = 20, 2        
        z_default_list, rho_default_list = bimodal_beta_fix_avg(
            beta_a1, beta_b1, beta_a2, beta_b2, B_p, fix_avg_p
        )

    # Bimodal radicalized to 1 (min pdf point around 0.85)
    elif z_default_p == -12:

        beta_a1, beta_b1 = 1.5, 3
        beta_a2, beta_b2 = 30, 2        
        z_default_list, rho_default_list = bimodal_beta_fix_avg(
            beta_a1, beta_b1, beta_a2, beta_b2, B_p, fix_avg_p
        )

    else:
        sys.exit(f"FATAL ERROR: unsupported default {z_default_p} user init")

    return z_default_list, z_default_list, rho_default_list

