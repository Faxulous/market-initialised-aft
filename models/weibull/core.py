"""Weibull survival model (Model 2 from Leriou & Ntzoufras 2025)."""

import math

import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_conditional_weibull(
    u: torch.Tensor, lam: torch.Tensor, g: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """Sample residual time from Weibull given survival time s.

    Uses inverse CDF: t = (s^g + (-log(u))/lam)^(1/g) - s
    When s=0, reduces to standard inverse CDF: t = (-log(u)/lam)^(1/g)
    """
    x = -torch.log(u) / lam
    age = torch.where(s > 0, s, 1.)  # Keep the zero-age branch finite.
    residual = s * torch.expm1(torch.log1p(x / age.pow(g)) / g)
    return torch.where(s > 0, residual, x.pow(1 / g))


def _tally(diff: torch.Tensor) -> tuple[float, float, float]:
    """Convert final score diffs to win/draw/loss probabilities."""
    w, d = (diff > 0).float().mean().item(), (diff == 0).float().mean().item()
    return w, d, 1 - w - d


def simulate_single_gamma_phi(
    eta1: float, eta2: float, sd: int, minute: int, rem: float,
    gamma: float, phi: float, elapsed: float = 0, n: int = 1000,
    *, period: int, h1_left: float,
) -> tuple[float, float, float]:
    """Half-specific Weibull with continuous goal age and no covariates."""
    return simulate_covariates_single_gamma_phi(
        eta1, eta2, sd, minute, rem, {}, {}, gamma, phi, elapsed, n,
        period=period, h1_left=h1_left)


def _compute_cov_effects(betas: dict, covs: dict):
    """Sum covariate contributions; absent fitted inputs are errors."""
    cov_h, cov_a = 0., 0.
    for name, b in betas.items():
        if name in ("red_diff", "score_diff", "cum_psxg_diff", "cum_ng_psxg_diff", "intv_ng_psxg_diff"):
            cov_h += b * covs[name]
            cov_a -= b * covs[name]
        else:
            cov_h += b * covs[f"{name}_home"]
            cov_a += b * covs[f"{name}_away"]
    return cov_h, cov_a


def simulate_goals(eta1, eta2, minute, rem, betas, covs, gamma, phi,
                   elapsed=0., n=1000, *, period, h1_left):
    """Sample remaining goals; calibration and prediction use this same process.

    Observed shot deviations/red cards stay fixed. Time covariates advance per
    path. Goal age continues across half-time; only a goal resets it.
    h1_left is forecast first-half time remaining, never the future realised end.
    """
    if (period not in (1, 2) or rem < 0 or not 0 <= h1_left <= rem or
            (period == 2 and h1_left != 0) or elapsed < 0 or n <= 0 or
            gamma <= 0 or phi <= 0 or
            not all(math.isfinite(v) for v in [eta1, eta2, rem, h1_left, elapsed, gamma, phi])):
        raise ValueError("Invalid Weibull state, shape or half duration")
    gh, ga = torch.zeros(n, device=DEV), torch.zeros(n, device=DEV)
    surv = torch.full((n,), float(elapsed), device=DEV)
    static = {k: v for k, v in betas.items() if not k.startswith('game_time')}
    ch, ca = _compute_cov_effects(static, covs)
    phases = [(gamma, h1_left, minute), (gamma*phi, rem-h1_left, 45)] if period == 1 else [(gamma*phi, rem, minute)]
    for g, duration, start in phases:
        left = torch.full((n,), float(duration), device=DEV)
        time = torch.full((n,), float(start), device=DEV)
        lgf = math.lgamma(1 + 1/g)
        while (left > 0).any():
            effect = torch.zeros(n, device=DEV)
            for name, b in betas.items():
                if name.startswith('game_time'):
                    effect += b * (time - covs['_game_time_center'] if name == 'game_time_c' else time)
            lh = torch.exp(-g * (eta1 + ch + effect - lgf))
            la = torch.exp(-g * (eta2 + ca + effect - lgf))
            if not (torch.isfinite(lh).all() and torch.isfinite(la).all() and (lh > 0).all() and (la > 0).all()):
                raise ValueError("Nonfinite or underflowed Weibull rate")
            u, v = torch.rand(n, device=DEV).clamp_min(1e-10), torch.rand(n, device=DEV).clamp_min(1e-10)
            th = sample_conditional_weibull(u, lh, g, surv)
            ta = sample_conditional_weibull(v, la, g, surv)
            sh, sa = (th < ta) & (th < left), (ta < th) & (ta < left)
            dt = torch.where(sh | sa, torch.minimum(th, ta), left)
            if ((left > 0) & (dt <= 0)).any():
                raise RuntimeError("Weibull simulation made no time progress")
            gh += sh; ga += sa
            left -= dt; time += dt
            surv = torch.where(sh | sa, 0., surv + dt)
    return gh, ga


def simulate_covariates_single_gamma_phi(
    eta1_base: float, eta2_base: float, sd: int, minute: int, rem: float,
    betas: dict, covs: dict, gamma: float, phi: float,
    elapsed: float = 0, n: int = 1000, *, period: int, h1_left: float,
) -> tuple[float, float, float]:
    """Outcome probabilities from the common continuous-clock goal simulator."""
    gh, ga = simulate_goals(eta1_base, eta2_base, minute, rem, betas, covs,
                           gamma, phi, elapsed, n, period=period, h1_left=h1_left)
    return _tally(sd + gh - ga)
