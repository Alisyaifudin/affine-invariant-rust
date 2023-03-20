python generate_p0.py 1 1 &&
python run_mcmc.py 1000 1 1 &&
python plot_chain.py 1 1 &&
python run_mcmc_again.py 10000 1 1 &&
python plot_chain_again.py 1 1 &&
python plot_corner.py 1 1 &&
python plot_fitting.py 1 1
#-329.85677357351585