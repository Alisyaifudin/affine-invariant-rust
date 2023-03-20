python generate_p0.py 2 2 &&
python run_mcmc.py 2000 2 2 &&
python plot_chain.py 2 2 &&
python run_mcmc_again.py 10000 2 2 &&
python plot_chain_again.py 2 2 &&
python plot_corner.py 2 2 &&
python plot_fitting.py 2 2
#-350.40796382338476