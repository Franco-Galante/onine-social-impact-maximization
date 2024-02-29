All scripts in Python to be called as (Python version 3.8.0):

- in windows: 'python script_name.py args' ('python script_name.py -h' for info)
- in linux  : 'python3 script_name.py args'


The c++ program should be compiled as:
- g++ -std=c++17 -Wall dynsolve3.cpp -o dynsolve3

And to run the program:
- in windows: 'dynsolve3 -h' (Command Prompt) or '.\out -h' (in PowerShell)
- in linux  : './dynsolve3 -h'

'res' folder: will be create in runtime if not yet present and will contain the
              output (csv) of the simulator and pdf figures after processing.


*********************** OPTIMAL vs GREEDY (single player) **********************

- one_player_multi_group.py: 
   single influencer scenario  where it tries to 'attract' the users' population
   towards the extreme opinion x=1. It considers a 0-1 \psi functions, and the 
   opinion space is discretized. The state of the system is kept in a dictionay
   whose keys are the (discrete) bins IDs and the values lists of numer of users
   in that bin from each prejudice group.

- dynsolve3.cpp: 
   std:map c++ implementation of the 'trellis-like' solution. The solved optimi-
   zation problem is:
   \min_{\{\bm{x}^{(i)}_n\}_{n=1}^N} \mathbb{E}_{x} || \bm{x}^T - \bm{x}_N ||
   The opinion space is discretized into B intervals. More details in the script

- compare_strategies.py:
   flexible script which allows to input set of parameters (--opt "[[], .., []]"
   in double quotes) and calls both the Python (greedy strategy) script and the 
   cpp (optimal strategy) one with the provided parameters. Then, it eiter shows
   or saves to file the comparison between thetwo strategies.


******************************** PAPER FIGURES *********************************

To obtain the figures in the WEBSCI paper, one should run from this folder:

python .\compare_strategies.py --recall --hide-title -s

this produces both the csv output file with the results and context informations
and saves as a PDF the Figures which have been included in the WEBSCI paper.