import csv
import os


# I need the distribution associated only to the Nash equilibria, to avoid saving all
# I do not pass x because i suppose that x_0=z always (hold for all two players exp)
# ne, bin_z and netropy may be list if the NE are multiple

def create_dict_param(args_p, ne_p, bin_z_group_ne_p, x0_p, pop_p, 
                      payoff_list_p, row0_p, col0_p):
    
    final_x_list = [] # (final) distribution of the bins at the NE
    for bin_z_group_v in bin_z_group_ne_p:
        tmp_list = []
        for k, v in bin_z_group_v.items():
            tmp_list.append(sum(v))
        final_x_list.append(tmp_list)
    
    ret_dict = {
        'closed': args_p.closed,
        'z': list(args_p.prejudice), # numpy to list
        'rho': list(args_p.rho),
        'alpha': args_p.alpha,
        'beta': args_p.beta, 
        'n_max': args_p.nmax,
        'd0': args_p.d0,
        'd1': args_p.d1,
        'B': args_p.intervals,
        'U': args_p.users,
        'psi': args_p.psi,
        'psi_param': args_p.param,
        'ne': ne_p,
        'x_0': x0_p,
        'x_N': final_x_list,
        'pop': pop_p,
        'payoffs': payoff_list_p,    # of the NE
        'row0': row0_p,              # numpy to list
        'col0': col0_p
    }
    
    return ret_dict


def create_or_append_csv(args_p, ne_p, bin_z_group_ne_p, x0_p, pop_p, 
                         payoff_list_p, row0_p, col0_p):

    dict_param_vals = create_dict_param(args_p, ne_p, bin_z_group_ne_p, x0_p, pop_p, 
                                        payoff_list_p, row0_p, col0_p)

    filename = args_p.filename + '_open.csv'
    if args_p.closed:
        filename = args_p.filename + '_closed.csv'

    save_folder = os.path.join('res') # go to parent folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    filename_path = os.path.join(save_folder, filename)

    header = list(dict_param_vals.keys())
    data   = list(dict_param_vals.values())
    
    if not os.path.isfile(filename_path):
        # create the file if not existing
        with open(filename_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(header)
    else:
        # check if the header matches
        with open(filename_path, 'r') as f:
            existing_header = next(csv.reader(f, delimiter='\t'))
            if existing_header != header:
                raise ValueError("Header does not match expected columns.")

    with open(filename_path, 'a', newline='') as csvfile:
        # append the row in the csv file
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(data)
