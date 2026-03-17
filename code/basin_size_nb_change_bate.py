"""
Call with 
Data Collection on Basin Stability --num_threads 5000 -t 800    
"""

# imports
import argparse
import multiprocessing
import shutil
from datetime import datetime
from itertools import combinations
from math import sin
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import xgi
from hypersync_draw import *
from hypersync_generate import *
from hypersync_identify import *
from hypersync_integrate import *
from numba import jit

mpl.use("agg")  

sb.set_theme(style="ticks", context="notebook")

results_dir = "../results/"
data_dir = "../data/"

Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)



def simulate_iteration(
    i,
    H,
    k1,
    k2,
    omega,
    t_end,
    dt,
    ic,
    noise,
    rhs,
    integrator,
    args,
    t_eval,
    n_reps,
    run_dir="",
    suffix="",
    **options,
):
    N = len(H)
    times = np.arange(0, t_end + dt / 2, dt)

    nrep_thetas = np.zeros((n_reps, N)) 

    fig, axs = plt.subplots(2, 2, figsize=(4, 2), width_ratios=[3, 1], sharex="col")
    cluster_save_count = 0
    other_save_count = 0

    for j in range(n_reps):
        psi_init = generate_state(N, kind=ic, noise=noise)

        thetas, times = simulate_kuramoto(
            H,
            k1=k1,
            k2=k2,
            omega=omega,
            theta_0=psi_init,
            t_end=t_end,
            dt=dt,
            rhs=rhs,
            integrator=integrator,
            args=args,
            t_eval=False,
            **options,
        )

        r1, r2, alpha, beta = args

        nrep_thetas[j] = thetas[:, -1]

        state = identify_state(thetas)

        if (
            (j <= 5)
            or (("cluster" in state) and (cluster_save_count <= 2))
            or (("other" in state and other_save_count <= 5))
        ):


            tag_params = f"k1_{k1}_k2_{k2}_{j}_alpha_{alpha}_beta_{beta:.2f}"

            fig = plot_summary(thetas, times, H, tag_params, suffix=suffix)

            fig_name = f"sync_{suffix}_{tag_params}"  # _{k2s[j]}_{i}"
            plt.savefig(f"{run_dir}{fig_name}.png", dpi=300, bbox_inches="tight")
            #plt.close()

            for ax in axs.flatten():
                ax.clear()

        if "cluster" in state: 
            cluster_save_count += 1
        elif "other" in state:
            other_save_count += 1
    plt.close("all")

    return i, nrep_thetas


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description="Run Kuramoto simulation")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallelization",
    )
    parser.add_argument(
        "-n", "--n_reps", type=int, default=10000, help="Number of repetitions"  
    )
    parser.add_argument("-t", "--t_end", type=float, default=800, help="End time")   

    parser.add_argument(
        "-i", "--integrator", type=str, default="RK45", help="ODE integrator"
    )

    args = parser.parse_args()

    n_reps = args.n_reps

    alpha = 0 
    betas = np.arange(0, 2.1, 0.1)
    N = 83

    r = 2
    r1 = r
    r2 = r
    
    suffix = f"ring_r_{r1}_sym_beta"  
    H = xgi.trivial_hypergraph(N)    

    k1 = 1 
    k2 = 5 
    omega = 0  

    ic = "random"  
    noise = 1e-1  

    t_end = args.t_end  
    dt = 0.01
    times = np.arange(0, t_end + dt / 2, dt)

    t_eval = False  
    integrator = args.integrator
    options = {"atol": 1e-8, "rtol": 1e-8}

    tag_params = f"tend_{t_end}_nreps_{n_reps}_r_{r1}"

    run_dir = f"{results_dir}run_{tag_params}/"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    current_script_path = Path(__file__).resolve()
    shutil.copy2(current_script_path, run_dir)
    shutil.copy2("hypersync_integrate.py", run_dir)
    shutil.copy2("hypersync_identify.py", run_dir)

    print("============")
    print("simulating..")
    print("============")

    with multiprocessing.Pool(processes=args.num_threads) as pool:
        results = []
        for i, beta in enumerate(betas):
            args = (r1, r2, alpha, beta)

            results.append(
                pool.apply_async(
                    simulate_iteration,
                    (
                        i,
                        H,
                        k1,
                        k2,
                        omega,
                        t_end,
                        dt,
                        ic,
                        noise,
                        rhs_ring_nb,
                        integrator,
                        args,
                        t_eval,
                        n_reps,
                        run_dir,
                        suffix,
                    ),
                    options,
                )
            )

        print("============")
        print("collecting parallel results..")
        print("============")

        thetas_arr = np.zeros((len(betas), n_reps, N))  
        for result in results:
            print(i)
            i, beta_thetas = result.get()
            thetas_arr[i] = beta_thetas
        print("============")
        print("done..")
        print("============")

    print("============")
    print("saving simulations..")
    print("============")

    np.save(
        f"{run_dir}thetas_arr_beta.npy", thetas_arr
    )  

    print("============")
    print("identifying states..")
    print("============")

    results = {}

    thetas_arr = thetas_arr.swapaxes(0, 1) 
    thetas_arr = thetas_arr[:, :, :, None]   
    for j, beta in enumerate(betas):
        states = [identify_state(thetas, atol=0.05) for thetas in thetas_arr[:, j]]
        states_unique, counts = np.unique(states, return_counts=True)
        probs = counts / n_reps

        results[beta] = {}
        for state, prob in zip(states_unique, probs):
            results[beta][state] = prob

    print("============")
    print("plotting end states..")
    print("============")
    wsize = 1
    fig, axs = plt.subplots(5, len(betas), figsize=(len(betas) * wsize, 5 * wsize))
    for i, thetas_rep in enumerate(thetas_arr[:5]):
        for j, thetas_k in enumerate(thetas_rep):
            plot_phases(thetas_k, -1, ax=axs[i, j], color="b")
            axs[i, j].text(
                0,
                0,
                f"{identify_state(thetas_k)}",
                fontsize="xx-small",
                va="center",
                ha="center",
            )

    for i, ax in enumerate(axs[0, :]):
        ax.set_title(f"beta={betas[i]}")

    fig_name = f"end_state_{tag_params}" 
    plt.savefig(f"{run_dir}{fig_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("============")
    print("plotting summary..")
    print("============")

    print(results)

    df = pd.DataFrame.from_dict(results, orient="index").reset_index(names="beta")
    df_long = df.melt(id_vars="beta", var_name="state", value_name="proba")

    df_long.to_csv(f"{run_dir}df_long_{tag_params}.csv")

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    g = sb.lineplot(
        data=df_long,
        x="beta",
        y="proba",
        hue="state",
        markers=True,
        ax=ax,
        alpha=0.7,
        style="state",
    )

    g.set(yscale="log")
    sb.move_legend(g, loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_xlabel("beta")

    title = f"ring {suffix}, {ic} ic, {n_reps} reps"
    ax.set_title(title)

    sb.despine()
    ax.set_ylim(ymax=1.1)

    fig_name = f"basin_size_ring_{suffix}_ic_{ic}_nreps_{n_reps}"

    plt.savefig(f"{run_dir}{fig_name}.png", dpi=300, bbox_inches="tight")

    print("============")
    print("done..")
    print("============")

    end_time = datetime.now()

    elapsed_time = end_time - start_time

    days = elapsed_time.days
    seconds = elapsed_time.total_seconds()
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    formatted_time = (
        f"{days} days {hours} hours {minutes} minutes {seconds:.0f} seconds"
    )

    print(f"Execution time: {formatted_time}")

    with open(f"{run_dir}execution_report.txt", "w") as f:
        f.write(f"Execution time: {formatted_time}")


  
  


