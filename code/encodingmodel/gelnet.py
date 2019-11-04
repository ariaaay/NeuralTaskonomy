from functools import partial
import torch
from gel.gelfista import make_A, gel_solve
from gel.gelpaths import gel_paths2


def summary(_support, _b, X_val, y_val, y_tr_μ, y_tr_σ, ns):
    if _support is None:
        # empty support, so just use the computed training mean
        yhat_val = y_tr_μ
        _support_set = {}
    else:
        X_val_supp = X_val[_support]
        yhat_val = (X_val_supp.t() @ _b) * y_tr_σ + y_tr_μ  # we have to rescale
        _support_set = set([s.item() for s in _support])

    rmse = ((y_val - yhat_val) ** 2).mean().sqrt().item()

    # compute group support
    group_support = []
    start_idx = 0
    for n_j in ns:
        group_idxs = range(start_idx, start_idx + n_j)
        if all(i in _support_set for i in group_idxs):
            group_support.append(1)
        elif not any(i in _support_set for i in group_idxs):
            group_support.append(0)
        else:
            # This shouldn't happen! Groups are either all in or all out.
            raise AssertionError
        start_idx += n_j

    return group_support, rmse


def gelnet(A_trs, y_tr, A_vals, y_val, l_rs, ns, verbose=False, device="cpu"):

    # regularization strengths for ridge regression

    # We need to standardize the data to prevent repeated computations.
    A_tr_μs = [A_j.mean(dim=0, keepdim=True) for A_j in A_trs]
    A_tr_σs = [A_j.std(dim=0, keepdim=True).clamp_min_(1e-8) for A_j in A_trs]
    A_tr_stans = [
        (A_j - A_μ_j) / A_σ_j for A_j, A_μ_j, A_σ_j in zip(A_trs, A_tr_μs, A_tr_σs)
    ]
    A_val_stans = [
        (A_j - A_μ_j) / A_σ_j for A_j, A_μ_j, A_σ_j in zip(A_vals, A_tr_μs, A_tr_σs)
    ]

    X_val = torch.cat(A_val_stans, dim=1).t()  # note the transpose

    y_tr_μ = y_tr.mean()
    y_tr_σ = y_tr.std().clamp_min_(1e-8)
    y_tr_stan = (y_tr - y_tr_μ) / y_tr_σ

    summaries = gel_paths2(
        gel_solve,
        {
            # arguments to gel_solve
            "t_init": None,
            "ls_beta": 0.99,
            "max_iters": 3000,
            "rel_tol": 1e-5,
        },
        make_A,  # we don't explicitly call make_A here
        A_tr_stans,  # we need to pass the standardized A values
        y_tr_stan,  # note: standardized output vector
        ks=[
            0.01,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            0.99,
        ],  # this is a list of values to trade-off
        #   between l_1 and l_2 regularizations, with
        #   l_1 = k*l and l_2 = (1 - k)*l
        n_ls=50,  # number of l values for each k value
        l_eps=1e-5,  # ratio of minimum to maximum l value;
        #    n_ls number of l values are generated
        #    according to this ratio, for each k
        l_rs=l_rs,  # ridge regularization values
        summ_fun=partial(
            summary, X_val=X_val, y_val=y_val, y_tr_μ=y_tr_μ, y_tr_σ=y_tr_σ, ns=ns
        ),
        supp_thresh=1e-4,  # norm threshold for considering coefficient vectors
        #     to be 0 (default 1e-6)
        device=device,  # pytorch device (default cpu)
        verbose=verbose,  # verbosity (default False)
        ls_grid=None,  # override l value computation with a fixed grid
        #     (default is None)
        aux_rel_tol=1e-3,  # tolerance for solving auxiliary problems
        #     to get better initializations (default 1e-3)
        dtype=torch.float32,  # torch dtype (default float32)
    )

    return summaries
