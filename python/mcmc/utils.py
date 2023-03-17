from matplotlib import pyplot as plt
import corner

def plot_chain(chain, labels, figsize=(10, 10), alpha=0.1, path=None):
    fig, ax = plt.subplots(len(labels), 1, figsize=figsize)
    ax[0].set_title("MCMC Chain")
    for i, label in enumerate(labels):
        ax[i].plot(chain[:, :, i], alpha=alpha, color="k")
        ax[i].set_ylabel(label)
        ax[i].set_xlim(0, len(chain))
        ax[i].grid(False)
    ax[-1].set_xlabel("step number")
    if path is not None:
        fig.savefig(path)
    plt.show()


def plot_corner(chain, labels, burn=0, truths=None, path=None):
    ndim = chain.shape[2]
    flat = chain[burn:].reshape((-1, ndim)).copy()
    if flat.shape[1] != len(labels):
        raise ValueError("labels must have same length as chain dimension")
    fig = corner.corner(
        flat, 
        labels=labels, 
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        truths=truths,
        title_kwargs={"fontsize": 12},
    )
    if path is not None:
        fig.savefig(path)
    plt.show()