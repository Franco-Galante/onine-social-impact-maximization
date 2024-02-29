TO obtain the data (csv) to construct the Figures in the paper run (recall it 
needs to be 'python3' in Linux), in this sequence, the following commands:

python .\two_players_game.py -z -3 --closed --norm
python .\two_players_game.py -z -3 --closed
python .\two_players_game.py -b 0.8 -z -3 --closed --norm
python .\two_players_game.py -b 0.8 -z -3 --closed

Then to obtain all the figures in Fig 7 of the paper:

python plot_paper_figs.py

The 'two_players_game.py' script is flexible (type '-h' for help), while that
for ploting is customized to obtain the paper figures. Minimal modifications
would make it work in a general setting.
