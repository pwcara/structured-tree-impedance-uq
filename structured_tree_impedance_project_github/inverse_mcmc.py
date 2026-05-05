from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import default_objects, PARAM_RANGES, FIG_DIR, DATA_DIR
from io_utils import finalize_figure, save_json
from core_impedance import make_tree_wall_from_values, compute_pressure_solution, build_time_array, synthetic_inflow_waveform

def log_prior(r_min, lrr):
    rlo, rhi = PARAM_RANGES['r_min_cm']
    llo, lhi = PARAM_RANGES['length_to_radius']
    if rlo <= r_min <= rhi and llo <= lrr <= lhi:
        return 0.0
    return -np.inf

def main():
    blood, base_wall, base_tree, config = default_objects()
    rng = np.random.default_rng(config.random_seed + 2025)
    t = build_time_array(config)
    q = synthetic_inflow_waveform(t, config.cycle_length_s)
    true_values = {'r_min_cm': 0.018, 'length_to_radius': 58.0}
    tree_true, wall_true = make_tree_wall_from_values(true_values, base_tree, base_wall)
    true_sol = compute_pressure_solution(tree_true, wall_true, blood, config, t=t, q=q)
    p_true = true_sol['p']
    sigma = 1.0
    p_obs = p_true + rng.normal(0, sigma, size=len(p_true))
    k = 4
    idx = np.arange(0, len(t), k)
    t_obs = t[idx]
    p_obs_sub = p_obs[idx]

    def model_pressure(r_min, lrr):
        tree_i, wall_i = make_tree_wall_from_values({'r_min_cm': r_min, 'length_to_radius': lrr}, base_tree, base_wall)
        sol = compute_pressure_solution(tree_i, wall_i, blood, config, t=t, q=q)
        return sol['p'][idx]

    def log_likelihood(r_min, lrr):
        pred = model_pressure(r_min, lrr)
        res = p_obs_sub - pred
        return -0.5 * np.sum((res / sigma) ** 2)

    def log_posterior(r_min, lrr):
        lp = log_prior(r_min, lrr)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(r_min, lrr)
    n_steps = 1200
    burn = 300
    proposal_std = np.array([0.0025, 3.0])
    chain = np.zeros((n_steps, 2))
    logp = np.zeros(n_steps)
    accepted = np.zeros(n_steps, dtype=bool)
    current = np.array([base_tree.r_min_cm, base_tree.length_to_radius], dtype=float)
    current_lp = log_posterior(current[0], current[1])
    for s in range(n_steps):
        proposal = current + rng.normal(0, proposal_std)
        prop_lp = log_posterior(proposal[0], proposal[1])
        accept_prob = np.exp(prop_lp - current_lp) if np.isfinite(prop_lp) else 0.0
        if rng.uniform() < accept_prob:
            current = proposal
            current_lp = prop_lp
            accepted[s] = True
        chain[s, :] = current
        logp[s] = current_lp
    chain_df = pd.DataFrame({'step': np.arange(n_steps), 'r_min_cm': chain[:, 0], 'length_to_radius': chain[:, 1], 'log_posterior': logp, 'accepted': accepted})
    chain_df.to_csv(DATA_DIR / 'mcmc_chain.csv', index=False)
    post = chain[burn:, :]
    accept_rate = float(accepted.mean())
    summary = {'true_r_min_cm': true_values['r_min_cm'], 'true_length_to_radius': true_values['length_to_radius'], 'posterior_mean_r_min_cm': float(np.mean(post[:, 0])), 'posterior_std_r_min_cm': float(np.std(post[:, 0])), 'posterior_mean_length_to_radius': float(np.mean(post[:, 1])), 'posterior_std_length_to_radius': float(np.std(post[:, 1])), 'acceptance_rate': accept_rate, 'sigma_mmhg': sigma, 'burn_in': burn, 'n_steps': n_steps}
    save_json(summary, DATA_DIR / 'mcmc_summary.json')
    fig, axs = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axs[0].plot(chain[:, 0], linewidth=1)
    axs[0].axhline(true_values['r_min_cm'], color='k', linestyle='--', label='true')
    axs[0].set_ylabel('r_min_cm')
    axs[0].legend()
    axs[0].grid(alpha=0.25)
    axs[1].plot(chain[:, 1], linewidth=1)
    axs[1].axhline(true_values['length_to_radius'], color='k', linestyle='--', label='true')
    axs[1].set_ylabel('length_to_radius')
    axs[1].set_xlabel('MCMC step')
    axs[1].legend()
    axs[1].grid(alpha=0.25)
    plt.suptitle('MCMC traces')
    finalize_figure(FIG_DIR / '22_mcmc_trace.png')
    plt.figure(figsize=(6, 5))
    plt.scatter(post[:, 0], post[:, 1], s=10, alpha=0.35)
    plt.scatter([true_values['r_min_cm']], [true_values['length_to_radius']], color='red', s=60, label='true')
    plt.xlabel('r_min_cm')
    plt.ylabel('length_to_radius')
    plt.title('Posterior samples')
    plt.legend()
    plt.grid(alpha=0.25)
    finalize_figure(FIG_DIR / '23_mcmc_posterior_scatter.png')
    mean_values = {'r_min_cm': summary['posterior_mean_r_min_cm'], 'length_to_radius': summary['posterior_mean_length_to_radius']}
    tree_mean, wall_mean = make_tree_wall_from_values(mean_values, base_tree, base_wall)
    p_mean = compute_pressure_solution(tree_mean, wall_mean, blood, config, t=t, q=q)['p']
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, p_true, label='true pressure', linewidth=2.2)
    plt.scatter(t_obs, p_obs_sub, s=18, alpha=0.6, label='noisy observations')
    plt.plot(t, p_mean, '--', label='posterior-mean fit', linewidth=2.2)
    plt.xlabel('Time [s]')
    plt.ylabel('Pressure [mmHg]')
    plt.title('Synthetic inverse problem: pressure fit')
    plt.legend()
    plt.grid(alpha=0.25)
    finalize_figure(FIG_DIR / '24_mcmc_pressure_fit.png')
    print('MCMC inverse problem done.')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
