"""
Functions to identify states in coupled oscillators

"""

import numpy as np
from numpy.linalg import norm

__all__ = [
    "identify_state",  
    "identify_k_clusters",
    "identify_winding_number",
    "order_parameter",
]


def identify_state(thetas, t=-1, atol=1e-3):

    R1 = order_parameter(thetas, order=1)
    # R2 = order_parameter(thetas, order=2)
    # R3 = order_parameter(thetas, order=3)
    diff = np.diff(thetas[:, t], append=thetas[0, t]) % (2 * np.pi)
    is_diff_zero = np.isclose(diff, 0, atol=atol) + np.isclose(
        diff, 2 * np.pi, atol=atol
    )

    q, is_twisted = identify_winding_number(thetas, t=-1)
    sorted_thetas = np.sort(thetas, axis=0)  # sort along node axis
    q_sorted, is_splay = identify_winding_number(sorted_thetas, t=-1)

    try:
        is_2clust, sizes2 = identify_k_clusters(thetas, k=2, t=-1, atol=1e-2)
    except Exception as err:
        is_2clust = False
        sizes2 = []
        print(err)

    try:
        is_3clust, sizes3 = identify_k_clusters(thetas, k=3, t=-1, atol=1e-2)
    except Exception as err:
        is_3clust = False
        sizes3 = []
        print(err)

    if is_twisted:
        return f"{q}-twisted"
    elif is_splay and q_sorted == 1:
        return "splay"
    elif np.isclose(R1[t], 1, atol=atol) and np.all(is_diff_zero):
        return "sync"
    elif is_2clust:
        return "2-cluster"
    elif is_3clust:
        return "3-cluster"
    else:
        return "other"


def identify_k_clusters(thetas, k, t, atol=1e-2):

    n_clust = k
    dist = 2 * np.pi / n_clust
    N = len(thetas)

    psi = thetas[:, t] % (2 * np.pi)
    psi = np.sort(psi)

    diff = np.diff(psi)
    idcs = np.where(diff >= 0.45 * dist)[0]

    clusters = []
    n_changes = len(idcs)
    for i in range(n_changes + 1):
        start = idcs[i - 1] + 1 if i > 0 else None
        end = idcs[i] + 1 if i < n_changes else None
        clusters.append(psi[start:end])

    if len(clusters) < k:
        return False, []

    is_k_clusters = True  
    sizes = [0] * k

    for i in range(n_changes + 1):
        if np.mean(np.diff(clusters[i])) > atol: 
            is_k_clusters = False

    for i in range(n_changes):
        dist_ij = abs(np.mean(clusters[i]) - np.mean(clusters[i + 1]))
        if abs(dist_ij - dist) > atol:
            is_k_clusters = False 

    if n_clust == len(clusters):
        sizes = [len(cluster) / N for cluster in clusters]
    elif n_clust == len(clusters) - 1:
        sizes = [len(cluster) / N for cluster in clusters[:-1]]
        sizes[0] += len(clusters[-1])  
    else:
        raise ValueError("k must be equal to or one unit below len(cluster)")

    return is_k_clusters, sizes


def identify_winding_number(thetas, t, atol=1e-1):

    thetas = thetas % (2 * np.pi)  # ensure it's mod 2 pi

    diff = np.diff(thetas[:, t], prepend=thetas[-1, t])

    # ensure phase diffs are in [-pi, pi]
    diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
    diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)

    q = np.sum(diff)
    w_no = round(q / (2 * np.pi))
    is_twisted_state = norm(diff - np.mean(diff)) < atol

    return w_no, is_twisted_state


def order_parameter(thetas, order=1, complex=False, axis=0):

    N = len(thetas)
    Z = np.sum(np.exp(1j * order * thetas), axis=axis) / N

    if complex:
        return Z
    else:
        return np.abs(Z)
