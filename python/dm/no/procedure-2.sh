python generate_p0.py 2 &&
python run_mcmc.py 500 2 &&
python plot_chain.py 2 &&
python run_mcmc_again.py 2000 2 &&
python plot_chain_again.py 2 &&
python plot_corner.py 2 &&
python plot_fitting.py 2