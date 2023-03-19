python generate_p0.py 1 1 &&
python run_mcmc.py 500 1 1 &&
python plot_chain.py 1 1 &&
python run_mcmc_again.py 2000 1 1 &&
python plot_chain_again.py 1 1 &&
python plot_corner.py 1 1 &&
python plot_fitting.py 1 1