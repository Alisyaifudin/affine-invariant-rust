python generate_p0.py 2 1 &&
python run_mcmc.py 500 2 1 &&
python plot_chain.py 2 1 &&
python run_mcmc_again.py 2000 1 1 &&
python plot_chain_again.py 2 1 &&
python plot_corner.py 2 1 &&
python plot_fitting.py 2 1