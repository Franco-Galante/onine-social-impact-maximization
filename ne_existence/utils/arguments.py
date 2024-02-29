import argparse


def parse_args():

    parser = argparse.ArgumentParser(description=
        "Find Nash Equilibria of the two-players game on a square delta-beta grid."
    )
    parser.add_argument(
        '-z', 
        '--prejudice', 
        metavar="e1 .. eN", 
        type=float, 
        nargs='*', 
        default=[0.25, 0.75], 
        help='Prejudice of the user groups (delta)'
    ) 
    parser.add_argument(
        '-x', 
        '--init-x', 
        metavar="e1 .. eN", 
        type=float, 
        nargs='*', 
        default=[0.25, 0.75], 
        help='Initial opinion of the user groups (delta)'
    )
    parser.add_argument(
        '-r', 
        '--rho', 
        metavar="e1 .. eN", 
        type=float, 
        nargs='*', 
        default=[0.5, 0.5], 
        help='Proportion of users in each group'
    )
    parser.add_argument(
        '-n', 
        '--nmax', 
        metavar='v', 
        type=int, 
        default=5, 
        help='Maximum number of posts'
    )
    parser.add_argument(
        '-d0', 
        metavar='v', 
        type=float, 
        default=0.1, 
        help='Opinion delta for player 1 (x^T=0)'
    )
    parser.add_argument(
        '-d1', 
        metavar='v', 
        type=float, 
        default=0.1, 
        help='Opinion delta for player 2 (x^T=1)'
    )
    parser.add_argument(
        '-p', 
        '--psi', 
        metavar="id", 
        type=str, 
        choices=['rect'], 
        default='rect',
        help='rect (only choice): rectangular function'
    )
    parser.add_argument(
        '--param', 
        metavar='v1 v2', 
        type=float, 
        nargs=2, 
        default=[1.0, 0.7], 
        help='Values of the two parameters controlling the shape of psi'
    ) # used only for 'general'
    parser.add_argument(
        '--n-points', 
        metavar='v', 
        type=int, 
        default=50,
        help='Number of points in each direction of the grid'
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
        '-v', 
        '--verbose', 
        default=False, 
        action='store_true', 
        help='Detailed program output'
    )
    parser.add_argument(
        '--only-plot', 
        default=False, 
        action='store_true',
        help='Number of points in each direction of the grid'
    )
    parser.add_argument(
        '--no-plot', 
        default=False, 
        action='store_true',
        help='Suppress the plot (to iteratively call the script)'
    )
    parser.add_argument(
        '--all', 
        default=False, 
        action='store_true',
        help='Plots all the existence diagrams'
    )
    parser.add_argument(
        '--general', 
        default=False, 
        action='store_true',
        help='Triggers the general simulator'
    ) 

    args = parser.parse_args()

    if not args.general:
        args.psi = 'rect'   # simplified scenario ('rect' function)
        args.param[0] = 1.0 # effectively only the second value is used,
                            # the first is supposed to be 1

    return args # checks on params done in the script that is be called later
