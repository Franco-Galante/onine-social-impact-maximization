********************* EVALUATE EXISTENCE AND TYPE OF N.E. **********************

- ne_existence.py: 
    evaluate existence and  type of Nash Equilibria for a grid of beta and delta
    datapoints. This gives an idea of how the equilibria behave as a function of
    the stubborness/inertia of the regular users. It also PLOTS the existence 
    and the types of NE.



******************************** PAPER FIGURES *********************************

To obtain the figures in the paper, execute the following comands:

python .\ne_existence.py --strict -z 0.15 0.75 -x 0.15 0.75  -d0 0.2 -d1 0.2 --no-plot
python .\ne_existence.py --strict -z 0.15 0.75 -x 0.15 0.75  -d0 0.1 -d1 0.2 --no-plot
python .\ne_existence.py --strict -z 0.15 0.75 -x 0.15 0.75  -d0 0.2 -d1 0.3 --no-plot
python .\ne_existence.py --strict -z 0.25 0.75 -x 0.25 0.75  -d0 0.2 -d1 0.2 --no-plot


To obtain the figures put in the paper (Fig. 6 and Fig. 5):

python plot_paper_figs.py
