python generate_p0.py 1 2 &&
python run_mcmc.py 1000 1 2 &&
python plot_chain.py 1 2 &&
python run_mcmc_again.py 10000 1 2 &&
python plot_chain_again.py 1 2 &&
python plot_corner.py 1 2 &&
python plot_fitting.py 1 2
# -326.16413368340625