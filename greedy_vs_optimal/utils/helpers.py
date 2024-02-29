import os
import ast
import sys
import subprocess



# I need to transalte the integer ID of the state, to the pair opinion values it
# comes from. Recall that the state is a number in [0, B*N-1] where B is the
# number of bins and N the number of users. It enumerates the possible combina-
# tions of the positions of the two groups (delta) of users.
def state_to_pair(states: list, B_p : int):
    mid_bin = [(1+2*i)/(2*B_p) for i in range(B_p)] # mid bin opinion value
    
    pair_list = []
    for state in states:
        idx_x1 = state // B_p
        idx_x2 = state % B_p
        pair_list.append([mid_bin[idx_x1], mid_bin[idx_x2]])

    return pair_list



# compile the cpp file provided as parameter (can be done once and then use args)
def compile_cpp(cpp_filename: str):
    basename, extension = os.path.splitext(cpp_filename)

    if extension != '.cpp':
        sys.exit(f'FATAL ERROR: trying to compile a {extension} file')
    else: 
        res = subprocess.run(
            [
                'g++', 
                '-std=c++17', 
                '-Wall', 
                cpp_filename, 
                '-o', 
                basename
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if res.returncode != 0:
            sys.exit(f'ERROR: compilation failed: {res.stderr.decode()}')



# runs the exe file of the c++ script (previously compiled) with the options
def call_cpp(opt_p: dict, exe: str):

    options = [
        '--call', 
        '-p2', str(opt_p['w']), 
        '-i', str(opt_p['B']),
        '-z1', str(opt_p['z1']), 
        '-z2', str(opt_p['z2']),
        '-x1', str(opt_p['x1']), 
        '-x2', str(opt_p['x2']),
        '-r', str(0.5), # proportion of the two groups
        '-n', str(opt_p['N']),
        '-a', str(opt_p['a']),
        '-b', str(opt_p['b']),
        '--xitarget', str(opt_p['xi_target'])
    ]

    if opt_p['recall'] != 0:
        options.append('--recall')

    exe_path = os.path.join('.', exe) # ./
    res = subprocess.run(
        [exe_path] + options,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    xi_time, ex_time, states_time = [], [], []
    if res.returncode != 0:
        print(f'ERROR STDOUT: {res.stdout.decode().strip()}')
        print(f'ERROR STDERR: {res.stderr.decode().strip()}')
        sys.exit(f'ERROR cpp script exited with err code {res.returncode}')
    else:
        output = res.stdout.decode().strip().split('\t')
        if len(output) != 3:
            sys.exit(f'FATAL ERROR: expected 3 from cpp got {len(output)} instead')

        xi_time     = ast.literal_eval(output[0])
        ex_time     = ast.literal_eval(output[1])
        states_time = ast.literal_eval(output[2]) # ID od the state, I need opinions
                                        
        # process the sequence of states
        pair_list = state_to_pair(states_time, opt_p['B'])

    return xi_time, ex_time, pair_list



# calls the python script to get the 'greedy' solution with the options
# specified by the dictionary 'options' which has as keys the experiment vars
# returns two lists: ex over time, and the actions xi over time
def call_py(opt_p: dict):

    options = [
        '--call', 
        '--param', str(1.0), str(opt_p['w']),
        '-z', str(opt_p['z1']), str(opt_p['z2']),
        '-c', '0.5', '0.5',\
        '-x', str(opt_p['x1']), str(opt_p['x2']),
        '-a', str(opt_p['a']),
        '-b', str(opt_p['b']),
        '-i', str(opt_p['B']),
        '-n', str(opt_p['N']),
        '-t', 'greedy',
        '--xi-target', str(opt_p['xi_target'])
    ]
    
    if opt_p['recall'] != 0:
        options.append('--recall')

    py_path = os.path.join('utils', 'one_player_multi_group.py')

    python_cmd = 'python3' if sys.platform == 'linux' else 'python'
    res = subprocess.run(
        [python_cmd, py_path] + options,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )


    xi_time, ex_time = [], []
    if res.returncode != 0:
        print(f'ERROR MESSAGE: {res.stdout.decode().strip()}')
        sys.exit(f'ERROR py script exited with err code {res.returncode}')
    else:
        # parse the output to get the variables
        output = res.stdout.decode().strip().split('\t')

        xi_time     = ast.literal_eval(output[0])
        ex_time     = ast.literal_eval(output[1])
        states_time = ast.literal_eval(output[2])
        
        # process the sequence of states
        pair_list = state_to_pair(states_time, opt_p['B'])

    return xi_time, ex_time, pair_list
