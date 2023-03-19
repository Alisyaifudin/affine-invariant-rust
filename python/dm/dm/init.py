def init(kind):
    if not kind in [1, 2]:
        raise ValueError('kind must be 1 or 2')
    ndim = 31 if kind == 1 else 33
    nwalkers = 2*ndim+2
    return ndim, nwalkers