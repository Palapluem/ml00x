"""Train and validate the Task 1 account-cohort + 5-profile Tobit-MVN classifier (experiment 006)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, logit, ndtr
from scipy.stats import norm as sp_norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "006"
PREDICTIONS = ROOT / "predictions" / "experiments"
TARGET = "is_cheater"
WEIGHTS_COMPLETE = (0.70, 0.10, 0.20)
WEIGHTS_INCOMPLETE = (0.45, 0.40, 0.15)
MVN_SHRINK = 0.30
MVN_EM_ITERS = 12
EPSILON = 1e-5
ACCOUNT_FIELDS = ("account_age_days", "level", "level_progression_speed")
SUPPORT_FIELDS = (
    *ACCOUNT_FIELDS[:2],
    "friend_network_size",
    "reports_received",
    "crosshair_placement",
)
SKILL_FIELDS = (
    "kill_death_ratio",
    "headshot_percentage",
    "win_rate",
    "accuracy_score",
    "kill_consistency",
    "reaction_time_ms",
    "aiming_smoothness",
    "spray_control_score",
    "game_sense_score",
    "input_consistency_score",
    "movement_pattern_score",
    "buy_decision_score",
    "first_blood_rate",
    "weapon_switch_speed",
)
GAME_SENSE_IDX = SKILL_FIELDS.index("game_sense_score")
# Each row gives the label and inclusive ranges for age, friends, and reports.
# The ranges were inferred from the labeled file and checked for contradictions per fold.
COHORTS = (
    (0, (1, 110), (0, 49), (10, 29)),
    (0, (120, 900), (45, 249), (5, 19)),
    (0, (350, 1300), (120, 349), (8, 24)),
    (0, (600, 1500), (100, 319), (5, 17)),
    (1, (15, 180), (10, 89), (4, 15)),
    (1, (15, 180), (10, 89), (6, 19)),
    (1, (200, 1000), (60, 269), (4, 15)),
    (1, (200, 1000), (60, 269), (6, 19)),
)
COHORT_LABELS = np.array([label for label, *_ in COHORTS])
# The 8 account cohorts share 5 distinct underlying skill profiles:
# C0->0, (C1,C2)->1, C3->2, (C4,C6)->3, (C5,C7)->4.
COHORT_TO_PROFILE = np.array([0, 1, 1, 2, 3, 4, 3, 4])
PROFILE_LABELS = np.array([0, 0, 0, 1, 1])


def uniform_density(values, low, high, *, discrete=False):
    """Return one for missing observations, which add no independent evidence."""
    return np.where(
        np.isnan(values), 1.0, ((low <= values) & (values <= high)) / (high - low + discrete)
    )


def account_density(values, age_range, level_range):
    """Marginalize progression = level / age × Uniform(0.15, 1.08)."""
    age, level, speed = values.T
    age_min, age_max = age_range
    level_min, level_max = level_range
    mult_min, mult_max = 0.15, 1.08
    age_width = age_max - age_min
    level_width = level_max - level_min
    mult_width = mult_max - mult_min
    has_age, has_level, has_speed = np.isfinite(values).T
    density = uniform_density(age, *age_range) * uniform_density(level, *level_range)
    density[has_speed & (speed <= 0)] = 0
    has_speed &= speed > 0

    complete = has_age & has_level & has_speed
    ratio = age[complete] / np.maximum(level[complete], level_min)
    density[complete] *= uniform_density(speed[complete] * ratio, mult_min, mult_max) * ratio

    missing_age = ~has_age & has_level & has_speed
    level_seen = np.maximum(level[missing_age], level_min)
    age_max_ratio = np.minimum(age_max, mult_max * level_seen / speed[missing_age])
    age_min_ratio = np.maximum(age_min, mult_min * level_seen / speed[missing_age])
    density[missing_age] *= np.maximum(age_max_ratio**2 - age_min_ratio**2, 0) / (
        2 * age_width * mult_width * level_seen
    )

    missing_level = has_age & ~has_level & has_speed
    age_seen = np.maximum(age[missing_level], age_min)
    speed_seen = speed[missing_level]
    upper_level = np.minimum(level_max, speed_seen * age_seen / mult_min)
    lower_level = np.maximum(level_min, speed_seen * age_seen / mult_max)
    density[missing_level] *= (
        np.maximum(np.log(upper_level / lower_level), 0) * age_seen / (level_width * mult_width)
    )

    speed_only = ~has_age & ~has_level & has_speed
    speed_seen = speed[speed_only]
    # Integrate over level, splitting at the points where either age bound starts to bind.
    breakpoints = [np.full_like(speed_seen, edge) for edge in level_range]
    for age_bound in age_range:
        for multiplier_bound in (mult_min, mult_max):
            breakpoints.append(np.clip(speed_seen * age_bound / multiplier_bound, *level_range))
    edges = np.sort(np.column_stack(breakpoints), axis=1)
    half_width = np.diff(edges, axis=1) / 2
    midpoint = edges[:, :-1] + half_width
    nodes, weights = np.polynomial.legendre.leggauss(12)
    level_nodes = midpoint[..., None] + half_width[..., None] * nodes
    lower_age = np.maximum(age_min, mult_min * level_nodes / speed_seen[:, None, None])
    upper_age = np.minimum(age_max, mult_max * level_nodes / speed_seen[:, None, None])
    integrand = np.maximum(upper_age**2 - lower_age**2, 0) / (
        2 * age_width * level_width * mult_width * level_nodes
    )
    density[speed_only] *= ((integrand @ weights) * half_width).sum(axis=1)
    return density


def cohort_likelihoods(frame):
    account = frame[list(ACCOUNT_FIELDS)].to_numpy(dtype=float)
    friends = frame["friend_network_size"].to_numpy(dtype=float)
    reports = frame["reports_received"].to_numpy(dtype=float)
    crosshair = frame["crosshair_placement"].to_numpy(dtype=float)
    columns = []
    for label, age_range, friend_range, report_range in COHORTS:
        level_range = (25, 96) if label else (28, 100)
        crosshair_range = (0.80, 0.94) if label else (0.82, 0.96)
        columns.append(
            account_density(account, age_range, level_range)
            * uniform_density(friends, *friend_range, discrete=True)
            * uniform_density(reports, *report_range, discrete=True)
            * uniform_density(crosshair, *crosshair_range)
        )
    return np.column_stack(columns)


def cohort_weights(likelihoods, labels):
    compatible = likelihoods * (labels[:, None] == COHORT_LABELS)
    if not compatible.any(axis=1).all():
        raise ValueError("A labeled row falls outside every cohort of its class")
    weights = np.full(len(COHORTS), 1 / len(COHORTS))
    for _ in range(100):
        joint = compatible * weights
        updated = (joint / joint.sum(axis=1, keepdims=True)).mean(axis=0)
        if np.max(np.abs(updated - weights)) < 1e-9:
            return updated
        weights = updated
    raise RuntimeError("Cohort weight estimation did not converge")


def design_matrix(likelihoods, priors, skill_scores):
    joint = likelihoods * priors
    total = joint.sum(axis=1, keepdims=True)
    membership = np.divide(joint, total, out=np.zeros_like(joint), where=total > 0)
    legit = membership[:, COHORT_LABELS == 0].sum(axis=1)
    cheater = membership[:, COHORT_LABELS == 1].sum(axis=1)
    cheater[total[:, 0] == 0] = 0.5
    # Shared skill terms plus deviations for cohorts with different skill distributions.
    interactions = np.column_stack([np.ones(len(legit)), membership[:, [0, 2, 3, 5, 7]], legit])
    linear = (interactions[:, :, None] * skill_scores[:, None, :]).reshape(len(legit), -1)
    quadratic = (interactions[:, :, None] * skill_scores[:, None, :] ** 2).reshape(len(legit), -1)
    prior_log_odds = logit(cheater.clip(EPSILON, 1 - EPSILON))
    return np.column_stack([prior_log_odds, interactions[:, 1:], linear, quadratic])


def skill_scores(transformer, skills):
    return np.nan_to_num(transformer.transform(skills).clip(-3.3, 3.3))


def fit_and_predict(train_likelihoods, train_skills, labels, eval_likelihoods, eval_skills):
    priors = cohort_weights(train_likelihoods, labels)
    transformer = QuantileTransformer(
        n_quantiles=300, output_distribution="normal", subsample=None
    ).fit(train_skills)
    train = design_matrix(train_likelihoods, priors, skill_scores(transformer, train_skills))
    model = LogisticRegression(C=0.04, max_iter=600).fit(train, labels)
    if model.n_iter_[0] == model.max_iter:
        raise RuntimeError("Logistic regression did not converge")
    design = design_matrix(eval_likelihoods, priors, skill_scores(transformer, eval_skills))
    return model.predict_proba(design)[:, 1], priors


def apply_support(probabilities, likelihoods):
    legit = likelihoods[:, COHORT_LABELS == 0].any(axis=1)
    cheater = likelihoods[:, COHORT_LABELS == 1].any(axis=1)
    scores = np.where(legit | cheater, probabilities, np.nan)
    return np.where(legit & ~cheater, EPSILON, np.where(cheater & ~legit, 1 - EPSILON, scores))


def best_threshold(y, probability):
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    ends = np.flatnonzero(np.r_[np.diff(sorted_probability) != 0, True])
    scores = 5 * np.cumsum(y[order])[ends] / (4 * y.sum() + ends + 1)
    return float(sorted_probability[ends[np.argmax(scores)]])


def f2(y, prediction):
    return float(5 * (y & prediction).sum() / (4 * y.sum() + prediction.sum()))


def evaluate(y, probability, folds, complete):
    thresholds = {
        fold: best_threshold(y[folds != fold], probability[folds != fold])
        for fold in np.unique(folds)
    }
    prediction = probability >= np.array([thresholds[fold] for fold in folds])
    metrics = {
        "f2": f2(y, prediction),
        "f2_complete": f2(y[complete], prediction[complete]),
        "f2_incomplete": f2(y[~complete], prediction[~complete]),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "log_loss": log_loss(y, probability),
    }
    return metrics, prediction, thresholds


def profile_likelihood_matrix(likelihoods_8, priors_8):
    """Collapse 8 account cohort likelihoods into the 5 distinct skill profiles."""
    weighted_8 = likelihoods_8 * priors_8[None, :]
    weighted_5 = np.zeros((len(likelihoods_8), 5), dtype=np.float64)
    for cohort_idx, profile_idx in enumerate(COHORT_TO_PROFILE):
        weighted_5[:, profile_idx] += weighted_8[:, cohort_idx]
    priors_5 = np.array(
        [priors_8[COHORT_TO_PROFILE == g].sum() for g in range(5)], dtype=np.float64
    )
    return weighted_5, priors_5


def combine_acct_skill_probs(weighted_5, priors_5, skill_probs_5):
    """Combine account-cohort likelihoods with 5-profile skill posteriors by Bayes' rule."""
    safe_probs = np.clip(skill_probs_5, 1e-12, 1.0)
    joint = weighted_5 * (safe_probs / priors_5[None, :])
    total = joint.sum(axis=1)
    cheater_prob = np.where(total > 0, joint[:, 3:5].sum(axis=1) / np.maximum(total, 1e-300), 0.5)
    return np.clip(cheater_prob, EPSILON, 1.0 - EPSILON)


def semi_cohort_weights(labeled_lik, labeled_y, unlab_lik):
    """Estimate 8-cohort mixture weights via EM on labeled and unlabeled train.csv rows."""
    compat_lab = labeled_lik * (labeled_y[:, None] == COHORT_LABELS)
    if not compat_lab.any(axis=1).all():
        raise ValueError("A labeled row falls outside every cohort of its class")
    weights = np.full(len(COHORTS), 1.0 / len(COHORTS))
    n_total = len(labeled_lik) + len(unlab_lik)
    for _ in range(150):
        joint_lab = compat_lab * weights
        resp_lab = joint_lab / joint_lab.sum(axis=1, keepdims=True)
        joint_unlab = unlab_lik * weights
        total_unlab = joint_unlab.sum(axis=1, keepdims=True)
        resp_unlab = np.divide(
            joint_unlab, total_unlab, out=np.zeros_like(joint_unlab), where=total_unlab > 0
        )
        updated = (resp_lab.sum(axis=0) + resp_unlab.sum(axis=0)) / n_total
        if np.max(np.abs(updated - weights)) < 1e-9:
            return updated
        weights = updated
    raise RuntimeError("Semi-supervised cohort weight estimation did not converge")


def _mvn_logpdf_missing(values, means, covs, masks_int, unique_masks):
    """Exact marginalized MVN log-density grouped by observed-feature bitmask."""
    n_rows = len(values)
    n_groups = len(means)
    out = np.zeros((n_rows, n_groups), dtype=np.float64)
    n_dims = values.shape[1]
    for mask_int in unique_masks:
        if mask_int == 0:
            continue
        idx = np.flatnonzero(masks_int == mask_int)
        obs = [j for j in range(n_dims) if (mask_int >> j) & 1]
        sub = values[np.ix_(idx, obs)]
        log_norm_const = len(obs) * np.log(2.0 * np.pi)
        for group_idx in range(n_groups):
            chol = np.linalg.cholesky(covs[group_idx][np.ix_(obs, obs)])
            sol = np.linalg.solve(chol, (sub - means[group_idx, obs]).T)
            logdet = 2.0 * np.sum(np.log(np.diag(chol)))
            out[idx, group_idx] = -0.5 * (log_norm_const + logdet + np.sum(sol * sol, axis=0))
    return out


def _tobit_1d_loglik(values, means, stds, upper=1.0):
    """Right-censored Gaussian (Tobit) log-likelihood for game_sense_score."""
    out = np.zeros((len(values), len(means)), dtype=np.float64)
    obs = np.isfinite(values)
    clipped = obs & (values >= upper - 1e-9)
    interior = obs & ~clipped
    if interior.any():
        z = (values[interior, None] - means[None, :]) / stds[None, :]
        out[interior] = -0.5 * np.log(2.0 * np.pi) - np.log(stds[None, :]) - 0.5 * z * z
    if clipped.any():
        out[clipped] = np.log(np.maximum(ndtr((means[None, :] - upper) / stds[None, :]), 1e-300))
    return out


def _fit_tobit_1d_weighted(values, weights, upper=1.0, n_iter=8):
    """Weighted EM MLE for a right-censored univariate Gaussian."""
    obs = np.isfinite(values)
    xo, wo = values[obs], weights[obs]
    clipped = xo >= upper - 1e-9
    w_sum = np.maximum(wo.sum(), 1e-9)
    mu = float(np.sum(wo * xo) / w_sum)
    sd = float(np.sqrt(np.sum(wo * (xo - mu) ** 2) / w_sum + 1e-6))
    for _ in range(n_iter):
        alpha = (upper - mu) / sd
        inv_mills = sp_norm.pdf(alpha) / np.maximum(ndtr(-alpha), 1e-12)
        exp_clip = mu + sd * inv_mills
        var_clip = sd**2 * max(1.0 + alpha * inv_mills - inv_mills**2, 1e-6)
        ex = np.where(clipped, exp_clip, xo)
        ex2 = np.where(clipped, exp_clip**2 + var_clip, xo**2)
        mu = float(np.sum(wo * ex) / w_sum)
        sd = float(np.sqrt(max(np.sum(wo * ex2) / w_sum - mu**2, 1e-6)))
    return mu, sd


def fit_predict_mvn5_semi(
    tr_lik,
    tr_skills,
    tr_y,
    unlab_lik,
    unlab_skills,
    eval_lik,
    eval_skills,
    *,
    shrink=MVN_SHRINK,
    n_em=MVN_EM_ITERS,
):
    """Semi-supervised 5-profile Tobit(game_sense) + Yeo-Johnson shrunk-cov MVN-EM."""
    other_cols = [j for j in range(len(SKILL_FIELDS)) if j != GAME_SENSE_IDX]
    sk_fit = np.vstack([tr_skills, unlab_skills])
    gs_fit = sk_fit[:, GAME_SENSE_IDX]
    gs_eval = eval_skills[:, GAME_SENSE_IDX]
    x_fit = sk_fit[:, other_cols].copy()
    x_eval = eval_skills[:, other_cols].copy()
    n_dims, n_groups = len(other_cols), 5

    for col_idx in range(n_dims):
        obs_fit_col = np.isfinite(x_fit[:, col_idx])
        obs_eval_col = np.isfinite(x_eval[:, col_idx])
        pt = PowerTransformer(method="yeo-johnson", standardize=True).fit(
            x_fit[obs_fit_col, col_idx : col_idx + 1]
        )
        x_fit[obs_fit_col, col_idx] = pt.transform(x_fit[obs_fit_col, col_idx : col_idx + 1]).ravel()
        if obs_eval_col.any():
            x_eval[obs_eval_col, col_idx] = pt.transform(
                x_eval[obs_eval_col, col_idx : col_idx + 1]
            ).ravel()

    obs_fit = np.isfinite(x_fit)
    obs_eval = np.isfinite(x_eval)
    bits = 1 << np.arange(n_dims)
    masks_fit = (obs_fit @ bits).astype(np.int64)
    masks_eval = (obs_eval @ bits).astype(np.int64)
    unique_fit = np.unique(masks_fit)
    unique_eval = np.unique(masks_eval)

    priors_8 = semi_cohort_weights(tr_lik, tr_y, unlab_lik)
    w5_tr, priors_5 = profile_likelihood_matrix(tr_lik, priors_8)
    w5_unlab, _ = profile_likelihood_matrix(unlab_lik, priors_8)
    w5_eval, _ = profile_likelihood_matrix(eval_lik, priors_8)

    compat_all = np.vstack([w5_tr * (tr_y[:, None] == PROFILE_LABELS[None, :]), w5_unlab])
    gamma = compat_all / np.maximum(compat_all.sum(axis=1, keepdims=True), 1e-300)
    log_compat = np.where(compat_all > 0, np.log(np.maximum(compat_all, 1e-300)), -np.inf)

    mu = np.zeros((n_groups, n_dims))
    cov = np.tile(np.eye(n_dims)[None, :, :], (n_groups, 1, 1))
    gs_mu = np.zeros(n_groups)
    gs_sd = np.ones(n_groups)
    for group_idx in range(n_groups):
        w_col = gamma[:, group_idx]
        for col_idx in range(n_dims):
            oj = obs_fit[:, col_idx]
            mu[group_idx, col_idx] = np.sum(w_col[oj] * x_fit[oj, col_idx]) / np.maximum(
                np.sum(w_col[oj]), 1e-9
            )

    for iteration in range(n_em):
        sum_w = gamma.sum(axis=0) + 1e-9
        for group_idx in range(n_groups):
            gs_mu[group_idx], gs_sd[group_idx] = _fit_tobit_1d_weighted(
                gs_fit, gamma[:, group_idx], upper=1.0, n_iter=8
            )
        if iteration == 0:
            for group_idx in range(n_groups):
                wg = gamma[:, group_idx]
                centered = np.where(obs_fit, x_fit - mu[group_idx], 0.0)
                obs_counts = np.maximum((obs_fit * wg[:, None]).T @ obs_fit, 1.0)
                cg = ((centered * wg[:, None]).T @ centered) / obs_counts
                eigvals, eigvecs = np.linalg.eigh(0.5 * (cg + cg.T))
                cov[group_idx] = (eigvecs * np.maximum(eigvals, 0.05)) @ eigvecs.T
        else:
            new_mu = np.zeros((n_groups, n_dims))
            new_cov = np.zeros((n_groups, n_dims, n_dims))
            for mask_int in unique_fit:
                idx = np.flatnonzero(masks_fit == mask_int)
                obs = [j for j in range(n_dims) if (mask_int >> j) & 1]
                mis = [j for j in range(n_dims) if not ((mask_int >> j) & 1)]
                gk = gamma[idx]
                sub_obs = x_fit[np.ix_(idx, obs)] if obs else None
                for group_idx in range(n_groups):
                    wg = gk[:, group_idx]
                    sw = wg.sum()
                    if sw < 1e-12:
                        continue
                    imputed = np.empty((len(idx), n_dims))
                    if obs:
                        imputed[:, obs] = sub_obs
                    if mis:
                        if obs:
                            s_oo = cov[group_idx][np.ix_(obs, obs)]
                            s_mo = cov[group_idx][np.ix_(mis, obs)]
                            s_mm = cov[group_idx][np.ix_(mis, mis)]
                            reg_mat = np.linalg.solve(s_oo, s_mo.T).T
                            imputed[:, mis] = (
                                mu[group_idx, mis] + (sub_obs - mu[group_idx, obs]) @ reg_mat.T
                            )
                            s_cond = s_mm - reg_mat @ s_mo.T
                        else:
                            imputed[:, mis] = mu[group_idx, mis]
                            s_cond = cov[group_idx]
                    new_mu[group_idx] += wg @ imputed
                    new_cov[group_idx] += (imputed * wg[:, None]).T @ imputed
                    if mis:
                        new_cov[group_idx][np.ix_(mis, mis)] += sw * s_cond
            for group_idx in range(n_groups):
                mu[group_idx] = new_mu[group_idx] / sum_w[group_idx]
                cg = new_cov[group_idx] / sum_w[group_idx] - np.outer(mu[group_idx], mu[group_idx])
                cov[group_idx] = 0.5 * (cg + cg.T) + 1e-4 * np.eye(n_dims)

        pooled_cov = sum((sum_w[g] / sum_w.sum()) * cov[g] for g in range(n_groups))
        for group_idx in range(n_groups):
            cov[group_idx] = (1.0 - shrink) * cov[group_idx] + shrink * pooled_cov

        lp_fit = _mvn_logpdf_missing(x_fit, mu, cov, masks_fit, unique_fit) + _tobit_1d_loglik(
            gs_fit, gs_mu, gs_sd, upper=1.0
        )
        log_joint = log_compat + lp_fit - np.log(priors_5)[None, :]
        gamma = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    lp_eval = _mvn_logpdf_missing(x_eval, mu, cov, masks_eval, unique_eval) + _tobit_1d_loglik(
        gs_eval, gs_mu, gs_sd, upper=1.0
    )
    log_ps_eval = lp_eval + np.log(priors_5)[None, :]
    skill_probs_eval = np.exp(log_ps_eval - logsumexp(log_ps_eval, axis=1, keepdims=True))
    return combine_acct_skill_probs(w5_eval, priors_5, skill_probs_eval), priors_8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promote", action="store_true", help="Update submit/index.csv if all folds improve"
    )
    args = parser.parse_args()

    raw = pd.read_csv(ROOT / "data" / "train.csv")
    train = raw.dropna(subset=[TARGET]).reset_index(drop=True)
    unlab = raw[raw[TARGET].isna()].reset_index(drop=True)
    baseline = pd.read_csv(PREDICTIONS / "004" / "oof.csv")
    prev005 = pd.read_csv(PREDICTIONS / "005" / "oof.csv")
    if (
        not train[["player_id", TARGET]]
        .astype({TARGET: int})
        .equals(baseline[["player_id", TARGET]])
        or not baseline[["player_id", "fold", TARGET]].equals(
            prev005[["player_id", "fold", TARGET]]
        )
    ):
        raise ValueError("Experiment 004/005 OOF rows and labels do not match the training file")

    labels = train[TARGET].to_numpy(dtype=bool)
    folds = baseline["fold"].to_numpy()
    oof_baseline = baseline["proba_exp004"].to_numpy()
    oof_005 = prev005["proba_exp005"].to_numpy()

    likelihoods = cohort_likelihoods(train)
    likelihoods_unlab = cohort_likelihoods(unlab)
    skills = train[list(SKILL_FIELDS)].to_numpy(dtype=float)
    skills_unlab = unlab[list(SKILL_FIELDS)].to_numpy(dtype=float)
    complete = train[list(SUPPORT_FIELDS)].notna().all(axis=1).to_numpy()

    oof_lr = np.empty(len(labels))
    oof_mvn = np.empty(len(labels))
    for fold in np.unique(folds):
        valid = folds == fold
        oof_lr[valid], _ = fit_and_predict(
            likelihoods[~valid],
            skills[~valid],
            labels[~valid],
            likelihoods[valid],
            skills[valid],
        )
        oof_mvn[valid], _ = fit_predict_mvn5_semi(
            likelihoods[~valid],
            skills[~valid],
            labels[~valid],
            likelihoods_unlab,
            skills_unlab,
            likelihoods[valid],
            skills[valid],
        )

    oof_004_sup = apply_support(oof_baseline, likelihoods)
    oof_lr_sup = apply_support(oof_lr, likelihoods)
    oof_mvn_sup = apply_support(oof_mvn, likelihoods)
    wc0, wc1, wc2 = WEIGHTS_COMPLETE
    wi0, wi1, wi2 = WEIGHTS_INCOMPLETE
    oof_comp = wc0 * oof_004_sup + wc1 * oof_lr_sup + wc2 * oof_mvn_sup
    oof_incomp = wi0 * oof_004_sup + wi1 * oof_lr_sup + wi2 * oof_mvn_sup
    oof_final = apply_support(np.where(complete, oof_comp, oof_incomp), likelihoods)

    base_metrics, _, _ = evaluate(labels, oof_004_sup, folds, complete)
    prev_metrics, prev_preds, _ = evaluate(labels, oof_005, folds, complete)
    mvn_metrics, _, _ = evaluate(labels, oof_mvn_sup, folds, complete)
    metrics, predictions, thresholds = evaluate(labels, oof_final, folds, complete)

    per_fold = []
    for fold, threshold_k in thresholds.items():
        mask = folds == fold
        old = f2(labels[mask], prev_preds[mask])
        new = f2(labels[mask], predictions[mask])
        per_fold.append(
            {
                "fold": int(fold),
                "exp005_f2": old,
                "f2": new,
                "delta_f2": new - old,
                "threshold": threshold_k,
            }
        )
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(1000):
        indices = rng.integers(len(labels), size=len(labels))
        differences.append(
            f2(labels[indices], predictions[indices]) - f2(labels[indices], prev_preds[indices])
        )
    interval = np.quantile(differences, [0.025, 0.975]).tolist()
    threshold = best_threshold(labels, oof_final)
    promote = interval[0] > 0 and all(row["delta_f2"] > 0 for row in per_fold)
    summary_df = pd.DataFrame(
        [base_metrics, prev_metrics, mvn_metrics, metrics],
        index=["exp004", "exp005", "mvn5_semi", "exp006"],
    )
    print(summary_df.to_string())
    print(pd.DataFrame(per_fold).to_string(index=False))
    print(f"Paired bootstrap 95% interval vs exp005: {interval}; all-OOF threshold: {threshold:.6f}")
    if args.promote and not promote:
        parser.error("The candidate did not pass validation for promotion")

    test = pd.read_csv(ROOT / "data" / "test.csv")
    test_baseline = pd.read_csv(PREDICTIONS / "004" / "test.csv")
    if not test["player_id"].equals(test_baseline["player_id"]):
        raise ValueError("Experiment 004 prediction rows do not match the test file")
    test_likelihoods = cohort_likelihoods(test)
    test_skills = test[list(SKILL_FIELDS)].to_numpy(dtype=float)
    test_complete = test[list(SUPPORT_FIELDS)].notna().all(axis=1).to_numpy()
    test_lr, lr_priors = fit_and_predict(likelihoods, skills, labels, test_likelihoods, test_skills)
    test_mvn, semi_priors = fit_predict_mvn5_semi(
        likelihoods,
        skills,
        labels,
        likelihoods_unlab,
        skills_unlab,
        test_likelihoods,
        test_skills,
    )
    original = test_baseline["proba"].to_numpy()
    orig_sup = apply_support(original, test_likelihoods)
    lr_sup = apply_support(test_lr, test_likelihoods)
    mvn_sup = apply_support(test_mvn, test_likelihoods)
    test_comp = wc0 * orig_sup + wc1 * lr_sup + wc2 * mvn_sup
    test_incomp = wi0 * orig_sup + wi1 * lr_sup + wi2 * mvn_sup
    final = apply_support(np.where(test_complete, test_comp, test_incomp), test_likelihoods)
    # Leave unsupported test rows to the existing classifier and its existing cutoff.
    unsupported = np.isnan(final)
    final[unsupported] = original[unsupported]
    with (ROOT / "models" / "experiments" / "004" / "stack_meta.json").open() as file:
        old_threshold = json.load(file)["threshold"]
    prediction = final >= np.where(unsupported, old_threshold, threshold)
    sample_path = next(
        p
        for p in (
            ROOT.parent / "submission" / "sample_submission.csv",
            ROOT.parent / "sample_submission.csv",
        )
        if p.exists()
    )
    sample = pd.read_csv(sample_path, usecols=["id"])
    if len(sample) != len(test) or not sample["id"].is_unique:
        raise ValueError("Official sample submission does not match the test row count")
    submission = sample.assign(task1=prediction.astype(int))

    assets = ROOT / "assets" / "experiments" / EXPERIMENT
    artifacts = ROOT / "models" / "experiments" / EXPERIMENT
    output = PREDICTIONS / EXPERIMENT
    for directory in (assets, artifacts, output):
        directory.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(assets / "results.csv", index_label="model")
    pd.DataFrame(per_fold).to_csv(assets / "per-fold-f2.csv", index=False)
    baseline[["player_id", "fold", TARGET]].assign(
        proba_exp006=oof_final, prediction=predictions.astype(int)
    ).to_csv(output / "oof.csv", index=False)
    test[["player_id"]].assign(proba=final, prediction=prediction.astype(int)).to_csv(
        output / "test.csv", index=False
    )
    metrics_summary = {
        "experiment": EXPERIMENT,
        "cohort_weights_labeled": lr_priors.tolist(),
        "cohort_weights_semi": semi_priors.tolist(),
        "weights_complete": list(WEIGHTS_COMPLETE),
        "weights_incomplete": list(WEIGHTS_INCOMPLETE),
        "mvn_shrink": MVN_SHRINK,
        "mvn_em_iters": MVN_EM_ITERS,
        "threshold": threshold,
        "metrics_exp004": base_metrics,
        "metrics_exp005": prev_metrics,
        "metrics_mvn5_semi": mvn_metrics,
        "metrics_exp006": metrics,
        "paired_bootstrap_delta_f2_95_percent": interval,
        "test_rows_outside_all_cohorts": int(unsupported.sum()),
        "passed_promotion_gate": promote,
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    if args.promote:
        submission.to_csv(ROOT / "submit" / "index.csv", index=False)


if __name__ == "__main__":
    main()
