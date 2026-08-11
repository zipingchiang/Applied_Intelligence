import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt


RANDOM_SEED = 42

CAPABILITY_ORDER = [
    "Scale",
    "Reasoning",
    "World-Interaction",
    "Agency",
]

REGIME_NAMES = [
    "R1_Predictive_Scaling",
    "R2_Reasoning_Centered",
    "R3_World-Interaction_Agentic",
]

REQUIRED_COLUMNS = [
    "model",
    "publication_date",
    "year",
    "log10_parameters",
    "log10_training_compute_flop",
    "log10_training_dataset_size_total",
    "reasoning_flag",
    "tool_use_flag",
    "agentic_flag",
    "chat_flag",
    "code_flag",
    "multimodal_flag",
    "frontier_model_flag",
]


@dataclass
class CapabilityBuildResult:
    values: pd.DataFrame
    metadata: pd.DataFrame
    proxy_weights: pd.DataFrame
    active_names: List[str]
    direct_reasoning_inspectability_available: bool
    direct_reasoning_faithfulness_available: bool


# -----------------------------------------------------------------------------
# CLI and input
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Continuous nonlinear capability dynamics + data-driven ANP + "
            "technological-regime Markov model for LLM evolution, with an auxiliary "
            "reasoning-inspectability layer that is explicitly separated from XAI faithfulness."
        )
    )
    p.add_argument("--csv", type=str, default=None, help="Path to completed LLM CSV.")
    p.add_argument("--outdir", type=str, default="llm_revised_dynamics_results")
    p.add_argument(
        "--period", choices=["year", "quarter"], default="quarter",
        help="Time aggregation. Quarter is preferred for regime dynamics because it supplies more transitions."
    )
    p.add_argument("--anp-self-loop", type=float, default=0.10)
    p.add_argument("--ridge-capability", type=float, default=0.03)
    p.add_argument("--ridge-regime", type=float, default=0.05)
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--include-incomplete-last-period", action="store_true",
        help="Include the final incomplete year/quarter in dynamic fitting. Default: exclude it from fitting."
    )
    p.add_argument("--r2-threshold", type=float, default=0.35)
    p.add_argument("--r3-agency-threshold", type=float, default=0.35)
    p.add_argument("--r3-world-threshold", type=float, default=0.35)
    p.add_argument(
        "--break-bootstrap", type=int, default=1000,
        help="Optional within-period multinomial bootstrap replicates for structural-break strength. Default 1000 for the manuscript robustness analysis; set 0 only for a faster exploratory run."
    )
    p.add_argument(
        "--threshold-grid", type=str, default="0.30,0.35,0.40",
        help="Comma-separated regime thresholds used for structural-break sensitivity."
    )
    p.add_argument(
        "--skip-publication-robustness", action="store_true",
        help="Skip threshold, temporal-aggregation, and provenance-restricted break sensitivity analyses."
    )
    return p.parse_args()


def locate_csv(csv_arg: Optional[str]) -> str:
    expected = "llm_history_SRWA_analysis_input.csv"
    candidates = []
    if csv_arg:
        candidates.append(Path(csv_arg))
    legacy = "llm_history_postimputation_complete_matrix.csv"
    candidates.extend([Path(expected), Path("/content") / expected, Path("/mnt/data") / expected,
                       Path(legacy), Path("/content") / legacy, Path("/mnt/data") / legacy])

    for path in candidates:
        if path.exists():
            return str(path)

    if "google.colab" in sys.modules:
        try:
            from google.colab import files  # type: ignore
            print(f"CSV not found. Please upload {expected}.")
            uploaded = files.upload()
            if expected in uploaded:
                return expected
            if len(uploaded) == 1:
                return next(iter(uploaded.keys()))
        except Exception as exc:  # pragma: no cover
            raise FileNotFoundError(f"Colab upload failed: {exc}") from exc

    raise FileNotFoundError(f"Could not find {expected}; provide --csv explicitly.")


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    req_missing = df[REQUIRED_COLUMNS].isna().sum()
    req_missing = req_missing[req_missing > 0]
    if len(req_missing):
        raise ValueError(f"Required input columns contain missing values: {req_missing.to_dict()}")

    for c in [
        "log10_parameters",
        "log10_training_compute_flop",
        "log10_training_dataset_size_total",
    ]:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError(f"Column {c} contains non-finite values.")

    dates = pd.to_datetime(df["publication_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("publication_date contains unparsable values.")


# -----------------------------------------------------------------------------
# Scaling helpers and capability construction
# -----------------------------------------------------------------------------

def robust_scale_01(x: Sequence[float], qlo: float = 0.05, qhi: float = 0.95) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, np.nan, dtype=float)
    lo, hi = np.quantile(x[finite], [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.min(x[finite]), np.max(x[finite])
    if hi <= lo:
        z = np.full_like(x, 0.5, dtype=float)
        z[~finite] = np.nan
        return z
    z = (x - lo) / (hi - lo)
    z = np.clip(z, 0.0, 1.0)
    z[~finite] = np.nan
    return z


def construct_equal_weight_proxy(
    arrays: Sequence[np.ndarray],
    component_names: Sequence[str],
    capability_name: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Construct one [0,1] proxy using equal component weights.

    Equal weighting is a deliberately non-informative aggregation rule at the
    measurement layer: it does not assert empirically estimated relative
    importance among the prespecified components. Cross-capability dependence
    is estimated later by the data-driven ANP.
    """
    X = np.column_stack([np.asarray(a, dtype=float) for a in arrays])
    if X.ndim != 2 or X.shape[1] != len(component_names):
        raise ValueError(f"Equal-weight input shape mismatch for {capability_name}.")
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values supplied to equal weighting for {capability_name}.")
    m = X.shape[1]
    if m < 1:
        raise ValueError(f"No components supplied for {capability_name}.")
    weights = np.full(m, 1.0 / m, dtype=float)
    diagnostics = pd.DataFrame({
        "capability": capability_name,
        "component": list(component_names),
        "method": "equal_weight",
        "weight": weights,
        "mean": np.mean(X, axis=0),
        "std": np.std(X, axis=0),
        "nonzero_count": np.sum(X > 0.0, axis=0).astype(int),
        "n_models": int(X.shape[0]),
    })
    return np.clip(X @ weights, 0.0, 1.0), diagnostics


def normalize_optional_score(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for c in candidates:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        finite = np.isfinite(x)
        if finite.sum() == 0:
            continue
        xf = x[finite]
        if np.nanmin(xf) >= 0.0 and np.nanmax(xf) <= 1.0:
            z = x.copy()
        else:
            z = robust_scale_01(x)
        return np.clip(z, 0.0, 1.0), c
    return None, None


def combine_optional_binary_flags(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[Optional[np.ndarray], List[str]]:
    """Combine optional binary/graded observability indicators by row-wise maximum.

    This is intentionally conservative: the result indicates availability/visibility
    of a reasoning trace, not faithfulness of that trace to internal computation.
    """
    present: List[str] = []
    cols: List[np.ndarray] = []
    for c in candidates:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if np.isfinite(x).sum() == 0:
            continue
        z = np.where(np.isfinite(x), np.clip(x, 0.0, 1.0), 0.0)
        cols.append(z)
        present.append(c)
    if not cols:
        return None, []
    return np.max(np.vstack(cols), axis=0), present


def build_continuous_capabilities(
    df: pd.DataFrame,
) -> CapabilityBuildResult:
    """Construct the de-nested equal-weight state x_t=[S,R,W,A]^T.

    The primary measurement specification uses equal weighting within each
    prespecified construct after robust [0,1] scaling of the three continuous
    Scale inputs:

        S = (P_scaled + C_scaled + D_scaled) / 3
        R = (I_reason + I_code + I_chat) / 3
        W = (I_multimodal + I_tool) / 2
        A = (I_tool + I_agentic + I_reason + I_frontier) / 4

    Direct standardized benchmark scores, if supplied in a future dataset, take
    precedence over the corresponding proxy. No Scale->Reasoning or
    Reasoning->World-interaction algebraic nesting is used.
    """
    lp = robust_scale_01(df["log10_parameters"].to_numpy(float))
    lc = robust_scale_01(df["log10_training_compute_flop"].to_numpy(float))
    ld = robust_scale_01(df["log10_training_dataset_size_total"].to_numpy(float))

    reasoning_flag = df["reasoning_flag"].to_numpy(float)
    tool = df["tool_use_flag"].to_numpy(float)
    agent_flag = df["agentic_flag"].to_numpy(float)
    chat = df["chat_flag"].to_numpy(float)
    code = df["code_flag"].to_numpy(float)
    mm = df["multimodal_flag"].to_numpy(float)
    frontier = df["frontier_model_flag"].to_numpy(float)

    weight_tables: List[pd.DataFrame] = []

    scale, wtab = construct_equal_weight_proxy(
        [lp, lc, ld],
        ["log10_parameters_scaled", "log10_training_compute_scaled", "log10_training_data_scaled"],
        "Scale",
    )
    weight_tables.append(wtab)

    reasoning_direct, reasoning_source = normalize_optional_score(
        df, ["reasoning_score", "reasoning_benchmark_score"]
    )
    if reasoning_direct is not None:
        reasoning = reasoning_direct
        reasoning_mode = "direct"
        reasoning_note = f"Direct normalized score from {reasoning_source}."
    else:
        reasoning, wtab = construct_equal_weight_proxy(
            [reasoning_flag, code, chat],
            ["reasoning_flag", "code_flag", "chat_flag"],
            "Reasoning",
        )
        weight_tables.append(wtab)
        reasoning_mode = "proxy_equal_denested"
        reasoning_note = (
            "Equal-weight proxy from reasoning, code, and chat indicators; "
            "Scale is not included in the Reasoning proxy."
        )

    # ------------------------------------------------------------------
    # Auxiliary reasoning inspectability layer
    # ------------------------------------------------------------------
    # Inspectability remains auxiliary and is not part of S/R/W/A.
    inspect_direct, inspect_source = normalize_optional_score(
        df, [
            "reasoning_inspectability_score",
            "reasoning_trace_score",
            "cot_observability_score",
            "chain_of_thought_observability_score",
            "reasoning_process_visibility_score",
        ]
    )
    trace_flag, trace_flag_sources = combine_optional_binary_flags(
        df, [
            "reasoning_trace_flag",
            "cot_flag",
            "chain_of_thought_flag",
            "reasoning_process_visible_flag",
            "deliberative_reasoning_flag",
        ]
    )
    if inspect_direct is not None:
        reasoning_inspectability = inspect_direct
        inspect_mode = "direct"
        inspect_note = (
            f"Direct normalized reasoning-inspectability score from {inspect_source}; "
            "not a faithfulness score."
        )
    elif trace_flag is not None:
        reasoning_inspectability = np.clip(
            0.80 * trace_flag + 0.20 * reasoning_flag, 0.0, 1.0
        )
        inspect_mode = "proxy_trace_flag"
        inspect_note = (
            "Partial-inspectability proxy from explicit reasoning-trace indicators "
            f"({', '.join(trace_flag_sources)}) plus reasoning flag; not faithful XAI."
        )
    else:
        reasoning_inspectability = np.clip(
            0.85 * reasoning_flag + 0.10 * chat + 0.05 * code, 0.0, 1.0
        )
        inspect_mode = "proxy_reasoning_flag"
        inspect_note = (
            "Weak auxiliary inspectability proxy dominated by reasoning_flag, with "
            "small chat/code terms; excluded from the S/R/W/A state and ANP."
        )

    faith_direct, faith_source = normalize_optional_score(
        df, [
            "reasoning_faithfulness_score",
            "cot_faithfulness_score",
            "explanation_faithfulness_score",
            "mechanistic_faithfulness_score",
        ]
    )
    if faith_direct is None:
        reasoning_faithfulness = np.full(len(df), np.nan)
        faith_mode = "unmeasured"
        faith_note = (
            "No direct reasoning-faithfulness measure supplied. Faithfulness is never "
            "inferred from reasoning visibility or metadata flags."
        )
    else:
        reasoning_faithfulness = faith_direct
        faith_mode = "direct"
        faith_note = f"Direct normalized reasoning-faithfulness score from {faith_source}."

    world_direct, world_source = normalize_optional_score(
        df, ["world_model_score", "world_model_benchmark_score"]
    )
    if world_direct is not None:
        world = world_direct
        world_mode = "direct"
        world_note = f"Direct normalized score from {world_source}."
    else:
        world, wtab = construct_equal_weight_proxy(
            [mm, tool],
            ["multimodal_flag", "tool_use_flag"],
            "World_Model",
        )
        weight_tables.append(wtab)
        world_mode = "proxy_equal_denested"
        world_note = (
            "Equal-weight world-interaction proxy from multimodality and tool use; "
            "Reasoning is not included and this is not a direct World-Interaction benchmark."
        )

    agency_direct, agency_source = normalize_optional_score(
        df, ["autonomous_agent_score", "agentic_score", "agent_benchmark_score"]
    )
    if agency_direct is not None:
        agency = agency_direct
        agency_mode = "direct"
        agency_note = f"Direct normalized score from {agency_source}."
    else:
        agency, wtab = construct_equal_weight_proxy(
            [tool, agent_flag, reasoning_flag, frontier],
            ["tool_use_flag", "agentic_flag", "reasoning_flag", "frontier_model_flag"],
            "Agency",
        )
        weight_tables.append(wtab)
        agency_mode = "proxy_equal"
        agency_note = (
            "Equal-weight proxy from tool-use, agentic, reasoning, and frontier indicators."
        )

    values = pd.DataFrame({
        "Scale": scale,
        "Reasoning": reasoning,
        "Reasoning_Inspectability": reasoning_inspectability,
        "Reasoning_Faithfulness": reasoning_faithfulness,
        "World_Model": world,
        "Agency": agency,
    }, index=df.index)

    rows = [
        ("Scale", "constructed_equal", "Equal mean of robustly scaled log parameters, compute, and data size."),
        ("Reasoning", reasoning_mode, reasoning_note),
        ("Reasoning_Inspectability", inspect_mode, inspect_note),
        ("Reasoning_Faithfulness", faith_mode, faith_note),
        ("World_Model", world_mode, world_note),
        ("Agency", agency_mode, agency_note),
    ]
    metadata = pd.DataFrame(rows, columns=["capability", "measurement_mode", "note"])
    proxy_weights = pd.concat(weight_tables, ignore_index=True) if weight_tables else pd.DataFrame()
    if not proxy_weights.empty:
        proxy_weights.insert(0, "proxy_weighting", "equal")

    active_names: List[str] = list(CAPABILITY_ORDER)
    for c in active_names:
        x = values[c].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError(
                f"Empirical capability {c} contains non-finite values; x_t must remain four-dimensional."
            )
        if np.nanstd(x) <= 1e-10:
            warnings.warn(f"Empirical capability {c} is nearly constant in the supplied dataset.")

    direct_inspectability = (
        inspect_mode == "direct" and np.isfinite(reasoning_inspectability).mean() >= 0.80
    )
    direct_faithfulness = (
        faith_mode == "direct" and np.isfinite(reasoning_faithfulness).mean() >= 0.80
    )

    return CapabilityBuildResult(
        values, metadata, proxy_weights, active_names,
        direct_inspectability, direct_faithfulness
    )


# -----------------------------------------------------------------------------
# Data-driven ANP
# -----------------------------------------------------------------------------

def build_empirical_anp(
    capability: pd.DataFrame,
    active_names: Sequence[str],
    self_loop: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw MI, column-stochastic supermatrix W, limit matrix, cross-influence matrix."""
    if not (0.0 <= self_loop < 1.0):
        raise ValueError("--anp-self-loop must be in [0,1).")
    names = list(active_names)
    X = capability[names].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Active capability matrix contains non-finite values.")
    d = X.shape[1]
    if d < 2:
        raise ValueError("At least two active capabilities are needed for ANP.")

    raw = np.zeros((d, d), dtype=float)  # target rows, source columns
    for target in range(d):
        y = X[:, target]
        for source in range(d):
            if target == source:
                continue
            mi = mutual_info_regression(
                X[:, [source]], y,
                random_state=seed + target * d + source,
                n_neighbors=min(5, max(3, len(X) // 50)),
            )[0]
            raw[target, source] = max(float(mi), 0.0)

    eps = max(float(np.nanmax(raw)), 1.0) * 1e-8
    base = raw + eps
    W = base / base.sum(axis=0, keepdims=True)
    W = (1.0 - self_loop) * W + self_loop * np.eye(d)
    W = W / W.sum(axis=0, keepdims=True)
    limit = np.linalg.matrix_power(W, 1000)

    # Cross-capability influence used in the nonlinear model: target rows receive
    # from source columns. Remove self-dependence so the separate autoregressive
    # coefficient is not duplicated, then row-normalize for comparable magnitude.
    cross = W.copy()
    np.fill_diagonal(cross, 0.0)
    rs = cross.sum(axis=1, keepdims=True)
    cross = np.divide(cross, rs, out=np.zeros_like(cross), where=rs > 0)
    return raw, W, limit, cross


# -----------------------------------------------------------------------------
# Period aggregation and completeness
# -----------------------------------------------------------------------------

def period_labels(df: pd.DataFrame, period: str) -> pd.Series:
    d = pd.to_datetime(df["publication_date"])
    if period == "year":
        return d.dt.year.astype(str)
    return d.dt.to_period("Q").astype(str)


def last_period_is_incomplete(df: pd.DataFrame, period: str) -> bool:
    dmax = pd.to_datetime(df["publication_date"]).max()
    if period == "year":
        return not (dmax.month == 12 and dmax.day >= 15)
    q_end_month = 3 * ((dmax.month - 1) // 3 + 1)
    return not (dmax.month == q_end_month and dmax.day >= 15)


def aggregate_capabilities(
    df: pd.DataFrame,
    capability: pd.DataFrame,
    active_names: Sequence[str],
    period: str,
) -> pd.DataFrame:
    tmp = capability[list(active_names)].copy()
    tmp["period"] = period_labels(df, period).values
    tmp["n_models"] = 1
    agg = tmp.groupby("period", sort=True).agg(
        {**{c: "mean" for c in active_names}, "n_models": "sum"}
    ).reset_index()
    return agg


def aggregate_reasoning_interpretability(
    df: pd.DataFrame,
    capability: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """Aggregate reasoning, inspectability and (if available) faithfulness by period.

    The output is descriptive and is intentionally separate from the default ANP/
    nonlinear/Markov core so that a proxy for trace visibility cannot masquerade as
    an additional independent capability or as faithful XAI.
    """
    tmp = pd.DataFrame({
        "period": period_labels(df, period).values,
        "Reasoning": capability["Reasoning"].to_numpy(float),
        "Reasoning_Inspectability": capability["Reasoning_Inspectability"].to_numpy(float),
        "Reasoning_Faithfulness": capability["Reasoning_Faithfulness"].to_numpy(float),
    })
    rows: List[Dict[str, object]] = []
    for label, g in tmp.groupby("period", sort=False):
        faith = g["Reasoning_Faithfulness"].to_numpy(float)
        rows.append({
            "period": label,
            "n_models": int(len(g)),
            "mean_reasoning": float(np.nanmean(g["Reasoning"])),
            "mean_reasoning_inspectability": float(np.nanmean(g["Reasoning_Inspectability"])),
            "mean_reasoning_faithfulness": float(np.nanmean(faith)) if np.isfinite(faith).any() else np.nan,
            "faithfulness_n_observed": int(np.isfinite(faith).sum()),
        })
    return pd.DataFrame(rows)


def plot_reasoning_interpretability(aux: pd.DataFrame, outdir: Path) -> None:
    if aux.empty:
        return
    x = np.arange(len(aux))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, aux["mean_reasoning"], marker="o", label="Reasoning capability")
    ax.plot(x, aux["mean_reasoning_inspectability"], marker="o", label="Reasoning inspectability")
    if aux["mean_reasoning_faithfulness"].notna().any():
        ax.plot(x, aux["mean_reasoning_faithfulness"], marker="o", label="Reasoning faithfulness (direct)")
    ax.set_xticks(x)
    ax.set_xticklabels(aux["period"], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized score / proxy")
    ax.legend()
    ax.text(
        0.01, -0.24,
        "Inspectability indicates observable/reportable reasoning behavior; it is not a faithful XAI measure unless direct faithfulness data are supplied.",
        transform=ax.transAxes, fontsize=10, va="top", wrap=True,
    )
    fig.tight_layout()
    fig.savefig(outdir / "reasoning_inspectability_evolution.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Nonlinear continuous capability dynamics
# -----------------------------------------------------------------------------

def unpack_capability_params(theta: np.ndarray, d: int) -> Dict[str, np.ndarray | float]:
    # c[d], rho[d], gamma[d], trend[d], alpha[1]
    if len(theta) != 4 * d + 1:
        raise ValueError("Unexpected capability parameter length.")
    return {
        "c": theta[0:d],
        "rho": theta[d:2*d],
        "gamma": theta[2*d:3*d],
        "trend": theta[3*d:4*d],
        "alpha_anp": float(theta[-1]),
    }


def capability_map(x: np.ndarray, tau: float, theta: np.ndarray, cross_anp: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-5, 1 - 1e-5)
    d = len(x)
    p = unpack_capability_params(theta, d)
    z = logit(x)
    network_z = cross_anp @ z
    network_x = cross_anp @ x
    h = (
        np.asarray(p["c"])
        + np.asarray(p["rho"]) * z
        + float(p["alpha_anp"]) * network_z
        + np.asarray(p["gamma"]) * (x * network_x)
        + np.asarray(p["trend"]) * tau
    )
    return expit(h)


def fit_capability_dynamics(
    agg: pd.DataFrame,
    active_names: Sequence[str],
    cross_anp: np.ndarray,
    ridge: float,
    starts: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
    names = list(active_names)
    X = agg[names].to_numpy(float)
    d = len(names)
    T = len(X)
    if T < 5:
        raise ValueError("At least five complete time periods are required for nonlinear capability dynamics.")
    taus = np.linspace(0.0, 1.0, T)
    n = agg["n_models"].to_numpy(float)
    trans_w = np.sqrt(n[:-1] * n[1:])
    trans_w = trans_w / max(np.mean(trans_w), 1e-12)

    def loss(theta: np.ndarray) -> float:
        sq = 0.0
        total_w = 0.0
        for t in range(T - 1):
            pred = capability_map(X[t], taus[t], theta, cross_anp)
            err = pred - X[t + 1]
            sq += float(trans_w[t] * np.sum(err ** 2))
            total_w += float(trans_w[t] * d)
        reg = ridge * float(np.sum(theta ** 2))
        return sq / max(total_w, 1e-12) + reg

    rng = np.random.default_rng(seed)
    bounds = (
        [(-5.0, 5.0)] * d
        + [(-2.5, 2.5)] * d
        + [(-5.0, 5.0)] * d
        + [(-5.0, 5.0)] * d
        + [(-5.0, 5.0)]
    )
    best = None
    for _ in range(max(1, starts)):
        x0 = np.zeros(4 * d + 1)
        x0[0:d] = rng.normal(0.0, 0.2, d)
        x0[d:2*d] = 0.8 + rng.normal(0.0, 0.1, d)
        x0[2*d:3*d] = rng.normal(0.0, 0.1, d)
        x0[3*d:4*d] = rng.normal(0.0, 0.2, d)
        x0[-1] = rng.normal(0.2, 0.1)
        res = minimize(
            loss, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-9},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    theta = np.asarray(best.x, dtype=float)

    pred_rows = []
    errs = []
    for t in range(T - 1):
        pred = capability_map(X[t], taus[t], theta, cross_anp)
        row: Dict[str, object] = {"from_period": agg.loc[t, "period"], "to_period": agg.loc[t+1, "period"]}
        for j, c in enumerate(names):
            row[f"observed_{c}"] = X[t+1, j]
            row[f"predicted_{c}"] = pred[j]
        pred_rows.append(row)
        errs.append(pred - X[t+1])
    E = np.vstack(errs)
    sse = float(np.sum(E ** 2))
    nobs = int(E.size)
    k_eff = int(4 * d + 1)
    diagnostics = {
        "objective": float(best.fun),
        "optimizer_success": bool(best.success),
        "optimizer_status": int(best.status),
        "rmse_one_step": float(np.sqrt(np.mean(E ** 2))),
        "mae_one_step": float(np.mean(np.abs(E))),
        "sse_one_step": sse,
        "n_observations": nobs,
        "effective_parameters": k_eff,
        "pseudo_bic": float(nobs * math.log(max(sse / nobs, 1e-15)) + k_eff * math.log(nobs)),
        "n_periods": int(T),
        "n_transitions": int(T - 1),
        "n_active_capabilities": int(d),
        "n_parameters": int(len(theta)),
    }
    return theta, diagnostics, pd.DataFrame(pred_rows)


def capability_map_no_network(x: np.ndarray, tau: float, theta: np.ndarray) -> np.ndarray:
    """Reduced capability model used for ANP/network ablation.

    It preserves own-state persistence and technological trend but removes both
    ANP cross-capability dependence and network-mediated nonlinear interactions.
    """
    x = np.clip(np.asarray(x, dtype=float), 1e-5, 1 - 1e-5)
    d = len(x)
    if len(theta) != 3 * d:
        raise ValueError("Unexpected no-network capability parameter length.")
    c = theta[:d]
    rho = theta[d:2*d]
    trend = theta[2*d:3*d]
    return expit(c + rho * logit(x) + trend * tau)


def fit_capability_no_network(
    agg: pd.DataFrame,
    active_names: Sequence[str],
    ridge: float,
    starts: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
    """Fit the reduced no-ANP/no-network capability baseline."""
    names = list(active_names)
    X = agg[names].to_numpy(float)
    d = len(names); T = len(X)
    taus = np.linspace(0.0, 1.0, T)
    n = agg["n_models"].to_numpy(float)
    trans_w = np.sqrt(n[:-1] * n[1:])
    trans_w = trans_w / max(np.mean(trans_w), 1e-12)

    def loss(theta: np.ndarray) -> float:
        sq = 0.0; total_w = 0.0
        for t in range(T - 1):
            pred = capability_map_no_network(X[t], taus[t], theta)
            err = pred - X[t + 1]
            sq += float(trans_w[t] * np.sum(err ** 2))
            total_w += float(trans_w[t] * d)
        return sq / max(total_w, 1e-12) + ridge * float(np.sum(theta ** 2))

    rng = np.random.default_rng(seed + 3333)
    bounds = [(-5.0, 5.0)] * d + [(-2.5, 2.5)] * d + [(-5.0, 5.0)] * d
    best = None
    for _ in range(max(1, starts)):
        x0 = np.zeros(3*d)
        x0[:d] = rng.normal(0.0, 0.2, d)
        x0[d:2*d] = 0.8 + rng.normal(0.0, 0.1, d)
        x0[2*d:3*d] = rng.normal(0.0, 0.2, d)
        res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    theta = np.asarray(best.x, dtype=float)

    rows=[]; errs=[]
    for t in range(T-1):
        pred = capability_map_no_network(X[t], taus[t], theta)
        row={"from_period": agg.loc[t,"period"], "to_period": agg.loc[t+1,"period"]}
        for j,c in enumerate(names):
            row[f"observed_{c}"]=X[t+1,j]; row[f"predicted_{c}"]=pred[j]
        rows.append(row); errs.append(pred-X[t+1])
    E=np.vstack(errs)
    sse=float(np.sum(E**2)); nobs=int(E.size); k_eff=int(3*d)
    diag={
        "objective": float(best.fun), "optimizer_success": bool(best.success),
        "rmse_one_step": float(np.sqrt(np.mean(E**2))),
        "mae_one_step": float(np.mean(np.abs(E))),
        "sse_one_step": sse, "n_observations": nobs, "effective_parameters": k_eff,
        "pseudo_bic": float(nobs*math.log(max(sse/nobs,1e-15))+k_eff*math.log(nobs)),
    }
    return theta, diag, pd.DataFrame(rows)


def capability_model_comparison(
    agg: pd.DataFrame, active_names: Sequence[str], cross_anp: np.ndarray,
    full_theta: np.ndarray, full_diag: Dict[str, float], ridge: float, starts: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the full ANP/network capability model against a reduced baseline."""
    X=agg[list(active_names)].to_numpy(float); T=len(X); d=len(active_names)
    taus=np.linspace(0.0,1.0,T); errs=[]
    for t in range(T-1):
        errs.append(capability_map(X[t],taus[t],full_theta,cross_anp)-X[t+1])
    E=np.vstack(errs); sse=float(np.sum(E**2)); nobs=int(E.size); k_eff=int(4*d+1)
    full_bic=float(nobs*math.log(max(sse/nobs,1e-15))+k_eff*math.log(nobs))
    _, base_diag, base_pred=fit_capability_no_network(agg,active_names,ridge,starts,seed)
    rows=[
        {"model":"Full_ANP_nonlinear","rmse":full_diag["rmse_one_step"],"mae":full_diag["mae_one_step"],
         "sse":sse,"effective_parameters":k_eff,"pseudo_bic":full_bic},
        {"model":"No_ANP_network_baseline","rmse":base_diag["rmse_one_step"],"mae":base_diag["mae_one_step"],
         "sse":base_diag["sse_one_step"],"effective_parameters":base_diag["effective_parameters"],
         "pseudo_bic":base_diag["pseudo_bic"]},
    ]
    comp=pd.DataFrame(rows)
    base_rmse=float(base_diag["rmse_one_step"]); full_rmse=float(full_diag["rmse_one_step"])
    comp["rmse_improvement_vs_no_anp_pct"] = 100.0*(base_rmse-full_rmse)/max(base_rmse,1e-15)
    return comp, base_pred


def capability_fixed_point(
    x0: np.ndarray,
    theta: np.ndarray,
    cross_anp: np.ndarray,
    tau: float = 1.0,
    damping: float = 0.5,
    tol: float = 1e-12,
    max_iter: int = 100000,
) -> Tuple[np.ndarray, int, float]:
    x = np.clip(np.asarray(x0, dtype=float), 1e-6, 1 - 1e-6)
    diff = np.inf
    for it in range(1, max_iter + 1):
        mapped = capability_map(x, tau, theta, cross_anp)
        new = (1.0 - damping) * x + damping * mapped
        diff = float(np.max(np.abs(new - x)))
        x = new
        if diff < tol:
            break
    residual = float(np.max(np.abs(capability_map(x, tau, theta, cross_anp) - x)))
    return x, it, residual


def full_jacobian_spectral_radius(
    xstar: np.ndarray,
    theta: np.ndarray,
    cross_anp: np.ndarray,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    d = len(xstar)
    J = np.zeros((d, d), dtype=float)
    for j in range(d):
        xp = xstar.copy(); xm = xstar.copy()
        xp[j] = min(1 - 1e-7, xp[j] + eps)
        xm[j] = max(1e-7, xm[j] - eps)
        fp = capability_map(xp, tau, theta, cross_anp)
        fm = capability_map(xm, tau, theta, cross_anp)
        denom = xp[j] - xm[j]
        J[:, j] = (fp - fm) / denom
    eig = np.linalg.eigvals(J)
    return eig, float(np.max(np.abs(eig)))


# -----------------------------------------------------------------------------
# Exclusive technological regimes
# -----------------------------------------------------------------------------

def assign_regimes(
    df: pd.DataFrame,
    capability: CapabilityBuildResult,
    r2_threshold: float,
    r3_agency_threshold: float,
    r3_world_threshold: float,
) -> pd.DataFrame:
    cap = capability.values
    labels: List[str] = []
    reasons: List[str] = []

    for i in range(len(df)):
        reasoning = float(cap.loc[i, "Reasoning"])
        world = float(cap.loc[i, "World_Model"])
        agency = float(cap.loc[i, "Agency"])

        r3 = (
            (agency >= r3_agency_threshold and world >= r3_world_threshold)
            or (
                float(df.loc[i, "tool_use_flag"]) >= 1.0
                and (
                    float(df.loc[i, "multimodal_flag"]) >= 1.0
                    or float(df.loc[i, "frontier_model_flag"]) >= 1.0
                )
            )
        )
        r2 = (
            reasoning >= r2_threshold
            or float(df.loc[i, "reasoning_flag"]) >= 1.0
            or float(df.loc[i, "code_flag"]) >= 1.0
        )

        if r3:
            labels.append(REGIME_NAMES[2]); reasons.append("World-Interaction/Agentic and agency threshold")
        elif r2:
            labels.append(REGIME_NAMES[1]); reasons.append("reasoning-centered threshold")
        else:
            labels.append(REGIME_NAMES[0]); reasons.append("predictive-scaling baseline")

    out = df[["model", "publication_date", "year"]].copy()
    out["regime"] = labels
    out["assignment_reason"] = reasons
    out["reasoning_inspectability"] = cap["Reasoning_Inspectability"].to_numpy(float)
    out["reasoning_faithfulness"] = cap["Reasoning_Faithfulness"].to_numpy(float)
    out["reasoning_inspectability_mode"] = (
        "direct" if capability.direct_reasoning_inspectability_available else "proxy_or_partial"
    )
    out["reasoning_faithfulness_directly_measured"] = bool(
        capability.direct_reasoning_faithfulness_available
    )
    out["xai_boundary"] = (
        "Reasoning-centered / visible reasoning may be partially inspectable; "
        "inspectability is not faithful explanation unless a direct faithfulness measure is supplied."
    )
    return out


def aggregate_regimes(df: pd.DataFrame, assignments: pd.DataFrame, period: str) -> pd.DataFrame:
    tmp = assignments.copy()
    tmp["period"] = period_labels(df, period).values
    counts = pd.crosstab(tmp["period"], tmp["regime"])
    for r in REGIME_NAMES:
        if r not in counts.columns:
            counts[r] = 0
    counts = counts[REGIME_NAMES]
    n = counts.sum(axis=1)
    q = counts.div(n, axis=0)
    out = q.copy()
    for r in REGIME_NAMES:
        out[f"count_{r}"] = counts[r]
    out["n_models"] = n
    out = out.reset_index()
    return out


# -----------------------------------------------------------------------------
# Non-homogeneous regime Markov model
# -----------------------------------------------------------------------------

def unpack_regime_params(theta: np.ndarray, d: int) -> Dict[str, np.ndarray | float]:
    """Parsimonious state-dependent regime model.

    Capability dimensions are collapsed to one ANP-weighted architectural-readiness
    score before entering the destination logits. This retains state dependence while
    avoiding d x K free capability coefficients that are poorly identified with only
    a few dozen historical transitions.
    """
    k = len(REGIME_NAMES)
    expected = 3 * k + 2  # b[K], trend[K], readiness[K], kappa, eta
    if len(theta) != expected:
        raise ValueError(f"Unexpected regime parameter length: {len(theta)} != {expected}")
    pos = 0
    b = theta[pos:pos+k]; pos += k
    g = theta[pos:pos+k]; pos += k
    readiness = theta[pos:pos+k]; pos += k
    kappa = float(theta[pos]); eta = float(theta[pos+1])
    b = b - b.mean()
    g = g - g.mean()
    readiness = readiness - readiness.mean()
    return {"b": b, "g": g, "readiness": readiness, "kappa": kappa, "eta": eta}


def regime_transition_matrix(
    q: np.ndarray,
    x: np.ndarray,
    tau: float,
    theta: np.ndarray,
    anp_weights: np.ndarray,
) -> np.ndarray:
    k = len(REGIME_NAMES)
    d = len(x)
    p = unpack_regime_params(theta, d)
    q = np.asarray(q, dtype=float)
    q = np.clip(q, 0.0, None)
    q = q / max(q.sum(), 1e-15)
    x = np.asarray(x, dtype=float)
    w = np.asarray(anp_weights, dtype=float)
    w = np.clip(w, 0.0, None); w = w / max(w.sum(), 1e-15)
    readiness_score = float(np.dot(x, w))

    destination = (
        np.asarray(p["b"])
        + tau * np.asarray(p["g"])
        + readiness_score * np.asarray(p["readiness"])
        + float(p["eta"]) * q
    )
    logits = destination[None, :] + float(p["kappa"]) * np.eye(k)
    return softmax(logits, axis=1)


def fit_regime_markov(
    regime_agg: pd.DataFrame,
    capability_agg: pd.DataFrame,
    active_names: Sequence[str],
    anp_weights: np.ndarray,
    ridge: float,
    starts: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    merged = regime_agg.merge(capability_agg, on=["period", "n_models"], how="inner")
    q = merged[REGIME_NAMES].to_numpy(float)
    X = merged[list(active_names)].to_numpy(float)
    T = len(q); d = len(active_names); k = len(REGIME_NAMES)
    if T < 5:
        raise ValueError("At least five complete time periods are required for regime Markov fitting.")
    taus = np.linspace(0.0, 1.0, T)
    n = merged["n_models"].to_numpy(float)
    wobs = np.sqrt(n[:-1] * n[1:])
    wobs = wobs / max(np.mean(wobs), 1e-12)

    pcount = 3*k + 2

    def loss(theta: np.ndarray) -> float:
        total = 0.0; denom = 0.0
        for t in range(T - 1):
            P = regime_transition_matrix(q[t], X[t], taus[t], theta, anp_weights)
            pred = q[t] @ P; target = q[t+1]
            mse = np.sum((pred - target) ** 2)
            ce = -np.sum(target * np.log(np.clip(pred, 1e-12, 1.0)))
            total += float(wobs[t] * (mse + 0.03 * ce)); denom += float(wobs[t])
        return total / max(denom, 1e-12) + ridge * float(np.sum(theta ** 2))

    rng = np.random.default_rng(seed + 7000)
    bounds = [(-8.0, 8.0)] * pcount
    best = None
    for _ in range(max(1, starts)):
        x0 = rng.normal(0.0, 0.10, pcount)
        x0[-2] = 1.0 + rng.normal(0.0, 0.10)
        x0[-1] = 0.2 + rng.normal(0.0, 0.10)
        res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-9})
        if best is None or res.fun < best.fun: best = res
    assert best is not None
    theta = np.asarray(best.x, dtype=float)

    preds=[]; trans_long=[]; errors=[]
    for t in range(T-1):
        P=regime_transition_matrix(q[t],X[t],taus[t],theta,anp_weights)
        pred=q[t]@P
        row={"from_period":merged.loc[t,"period"],"to_period":merged.loc[t+1,"period"]}
        for j,r in enumerate(REGIME_NAMES):
            row[f"observed_{r}"]=q[t+1,j]; row[f"predicted_{r}"]=pred[j]
        preds.append(row); errors.append(pred-q[t+1])
        for i,src in enumerate(REGIME_NAMES):
            for j,dst in enumerate(REGIME_NAMES):
                trans_long.append({"period":merged.loc[t,"period"],"source_regime":src,
                                   "destination_regime":dst,"transition_probability":P[i,j]})
    E=np.vstack(errors); sse=float(np.sum(E**2)); nobs=int(E.size)
    k_eff=3*(k-1)+2
    diagnostics={
        "objective":float(best.fun),"optimizer_success":bool(best.success),"optimizer_status":int(best.status),
        "rmse_one_step":float(np.sqrt(np.mean(E**2))),"mae_one_step":float(np.mean(np.abs(E))),
        "sse_one_step":sse,"n_observations":nobs,"effective_parameters":int(k_eff),
        "pseudo_bic":float(nobs*math.log(max(sse/nobs,1e-15))+k_eff*math.log(nobs)),
        "n_periods":int(T),"n_transitions":int(T-1),"n_parameters":int(pcount),
        "markov_capability_parameterization":"ANP-weighted architectural readiness score",
    }
    return theta, diagnostics, pd.DataFrame(preds), pd.DataFrame(trans_long)


# -----------------------------------------------------------------------------
# Reviewer baselines for regime Markov justification
# -----------------------------------------------------------------------------

def simple_regime_transition(theta: np.ndarray, tau: float, mode: str) -> np.ndarray:
    """Homogeneous or time-varying destination model with source-state persistence."""
    k = len(REGIME_NAMES)
    if mode == "homogeneous":
        b = np.asarray(theta[:k], float); kappa=float(theta[k]); g=np.zeros(k)
    elif mode == "time_varying":
        b=np.asarray(theta[:k],float); g=np.asarray(theta[k:2*k],float); kappa=float(theta[2*k])
    else:
        raise ValueError(mode)
    b=b-b.mean(); g=g-g.mean()
    logits=(b+tau*g)[None,:]+kappa*np.eye(k)
    return softmax(logits,axis=1)


def fit_simple_regime_baseline(regime_agg: pd.DataFrame, mode: str, ridge: float, starts: int, seed: int) -> Dict[str,float]:
    q=regime_agg[REGIME_NAMES].to_numpy(float); T=len(q); k=len(REGIME_NAMES)
    taus=np.linspace(0.0,1.0,T); n=regime_agg["n_models"].to_numpy(float)
    w=np.sqrt(n[:-1]*n[1:]); w=w/max(np.mean(w),1e-12)
    pcount=k+1 if mode=="homogeneous" else 2*k+1
    k_eff=k if mode=="homogeneous" else 2*k-1
    def loss(theta):
        total=0.0; denom=0.0
        for t in range(T-1):
            P=simple_regime_transition(theta,taus[t],mode); pred=q[t]@P; target=q[t+1]
            mse=np.sum((pred-target)**2); ce=-np.sum(target*np.log(np.clip(pred,1e-12,1.0)))
            total += float(w[t]*(mse+0.03*ce)); denom += float(w[t])
        return total/max(denom,1e-12)+ridge*float(np.sum(theta**2))
    rng=np.random.default_rng(seed + (101 if mode=="homogeneous" else 202))
    best=None
    for _ in range(max(1,starts)):
        x0=rng.normal(0,0.1,pcount); x0[-1]=1.0+rng.normal(0,0.1)
        res=minimize(loss,x0,method="L-BFGS-B",bounds=[(-8,8)]*pcount,
                     options={"maxiter":4000,"ftol":1e-12,"gtol":1e-9})
        if best is None or res.fun<best.fun: best=res
    theta=np.asarray(best.x,float)
    errs=[]
    for t in range(T-1):
        pred=q[t]@simple_regime_transition(theta,taus[t],mode); errs.append(pred-q[t+1])
    E=np.vstack(errs); sse=float(np.sum(E**2)); nobs=int(E.size)
    return {"model":mode,"rmse":float(np.sqrt(np.mean(E**2))),"mae":float(np.mean(np.abs(E))),
            "sse":sse,"effective_parameters":int(k_eff),
            "pseudo_bic":float(nobs*math.log(max(sse/nobs,1e-15))+k_eff*math.log(nobs)),
            "optimizer_success":bool(best.success)}


def regime_model_comparison(regime_agg: pd.DataFrame, full_diag: Dict[str,float], active_names: Sequence[str], ridge: float, starts: int, seed: int) -> pd.DataFrame:
    q=regime_agg[REGIME_NAMES].to_numpy(float); nobs=(len(q)-1)*len(REGIME_NAMES)
    full_sse=float(full_diag.get("sse_one_step", full_diag["rmse_one_step"]**2*nobs))
    full_keff=int(full_diag.get("effective_parameters", 3*(len(REGIME_NAMES)-1)+2))
    full_bic=float(full_diag.get("pseudo_bic", nobs*math.log(max(full_sse/nobs,1e-15))+full_keff*math.log(nobs)))
    rows=[fit_simple_regime_baseline(regime_agg,"homogeneous",ridge,starts,seed),
          fit_simple_regime_baseline(regime_agg,"time_varying",ridge,starts,seed),
          {"model":"full_state_dependent_nonlinear","rmse":float(full_diag["rmse_one_step"]),
           "mae":float(full_diag["mae_one_step"]),"sse":full_sse,"effective_parameters":int(full_keff),
           "pseudo_bic":full_bic,"optimizer_success":bool(full_diag.get("optimizer_success",True))}]
    df=pd.DataFrame(rows)
    h=float(df.loc[df.model=="homogeneous","rmse"].iloc[0])
    df["rmse_improvement_vs_homogeneous_pct"]=100.0*(h-df["rmse"])/max(h,1e-15)
    return df

# -----------------------------------------------------------------------------
# Reduced homogeneous Markov model for structural-break testing
# -----------------------------------------------------------------------------

def reduced_transition(theta: np.ndarray) -> np.ndarray:
    k = len(REGIME_NAMES)
    b = np.asarray(theta[:k], dtype=float)
    b = b - b.mean()
    kappa = float(theta[k])
    return softmax(b[None, :] + kappa * np.eye(k), axis=1)


def fit_reduced_segment(
    q: np.ndarray, start_t: int, end_t: int, ridge: float = 1e-5,
    x0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Fit transitions t=start_t,...,end_t-1, each predicting t+1.

    A warm start can be supplied. This matters for break scans and makes the
    robustness suite practical in Colab.
    """
    k = len(REGIME_NAMES)
    if end_t <= start_t:
        raise ValueError("Empty structural-break segment.")

    def obj(theta: np.ndarray) -> float:
        P = reduced_transition(theta)
        pred = q[start_t:end_t] @ P
        target = q[start_t+1:end_t+1]
        sse = float(np.sum((pred - target) ** 2))
        return sse + ridge * float(np.sum(theta**2))

    if x0 is None:
        x0 = np.zeros(k+1); x0[-1] = 1.0
    res = minimize(
        obj, np.asarray(x0,float), method="L-BFGS-B", bounds=[(-8,8)]*(k+1),
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
    )
    theta = np.asarray(res.x, dtype=float)
    P = reduced_transition(theta)
    pred = q[start_t:end_t] @ P
    target = q[start_t+1:end_t+1]
    sse = float(np.sum((pred - target) ** 2))
    return theta, sse


def structural_break_test(regime_agg: pd.DataFrame, min_transitions: int = 3) -> Tuple[pd.DataFrame, Dict[str, object]]:
    q = regime_agg[REGIME_NAMES].to_numpy(float)
    T = len(q); ntrans = T - 1
    if ntrans < 2 * min_transitions:
        return pd.DataFrame(), {"available": False,"reason":f"Need at least {2*min_transitions} transitions; found {ntrans}."}

    theta0, sse0 = fit_reduced_segment(q, 0, ntrans)
    nobs = ntrans * len(REGIME_NAMES)
    k0 = len(REGIME_NAMES)
    bic0 = nobs * math.log(max(sse0 / nobs, 1e-15)) + k0 * math.log(nobs)

    candidates=list(range(min_transitions,ntrans-min_transitions+1))
    left={}; xleft=theta0
    for b in candidates:
        xleft,sse=fit_reduced_segment(q,0,b,x0=xleft)
        left[b]=(xleft.copy(),sse)
    right={}; xright=theta0
    for b in reversed(candidates):
        xright,sse=fit_reduced_segment(q,b,ntrans,x0=xright)
        right[b]=(xright.copy(),sse)

    rows=[]
    for b in candidates:
        sse=left[b][1]+right[b][1]
        k1=2*len(REGIME_NAMES)+1
        bic=nobs*math.log(max(sse/nobs,1e-15))+k1*math.log(nobs)
        rows.append({"break_after_period":regime_agg.loc[b,"period"],
                     "first_post_break_period":regime_agg.loc[b+1,"period"] if b+1<T else "",
                     "break_transition_index":b,"sse_two_regime":sse,"bic_two_regime":bic,
                     "bic_homogeneous":bic0,"delta_bic_two_minus_homogeneous":bic-bic0})
    table=pd.DataFrame(rows).sort_values("bic_two_regime").reset_index(drop=True)
    best=table.iloc[0].to_dict()
    summary={"available":True,"homogeneous_sse":float(sse0),"homogeneous_bic":float(bic0),
             "best_break_after_period":str(best["break_after_period"]),
             "best_first_post_break_period":str(best["first_post_break_period"]),
             "best_break_transition_index":int(best["break_transition_index"]),
             "best_delta_bic_two_minus_homogeneous":float(best["delta_bic_two_minus_homogeneous"]),
             "interpretation":"Negative delta BIC favors the two-segment reduced Markov diagnostic; this remains exploratory unless supported by bootstrap and sensitivity analyses."}
    return table,summary


def bootstrap_break_stability(regime_agg: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Fast fixed-candidate bootstrap for the existence/strength of the selected break.

    The baseline break location is selected once from the observed data. Each
    bootstrap replicate resamples within-period regime counts and compares the
    homogeneous reduced Markov model with a two-segment model at that fixed candidate.
    Threshold/aggregation/provenance sensitivity is used separately to assess location.
    """
    if n_boot <= 0: return pd.DataFrame()
    baseline_table,baseline_summary=structural_break_test(regime_agg)
    if not baseline_summary.get("available"): return pd.DataFrame()
    bidx=int(baseline_summary["best_break_transition_index"])
    q0=regime_agg[REGIME_NAMES].to_numpy(float); T=len(q0); ntrans=T-1; k=len(REGIME_NAMES); nobs=ntrans*k
    rng=np.random.default_rng(seed+9000); rows=[]
    for rep in range(n_boot):
        qb=np.zeros_like(q0)
        for t in range(T):
            n=int(regime_agg.loc[t,"n_models"]); probs=np.clip(q0[t],0,None); probs=probs/probs.sum()
            qb[t]=rng.multinomial(n,probs)/max(n,1)
        th0,s0=fit_reduced_segment(qb,0,ntrans)
        _,s1=fit_reduced_segment(qb,0,bidx,x0=th0)
        _,s2=fit_reduced_segment(qb,bidx,ntrans,x0=th0)
        bic0=nobs*math.log(max(s0/nobs,1e-15))+k*math.log(nobs)
        bic1=nobs*math.log(max((s1+s2)/nobs,1e-15))+(2*k+1)*math.log(nobs)
        rows.append({"bootstrap":rep,"tested_break_after_period":baseline_summary["best_break_after_period"],
                     "delta_bic":bic1-bic0})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Publication robustness for the structural-break claim
# -----------------------------------------------------------------------------

def parse_threshold_grid(text: str) -> List[float]:
    vals=[]
    for token in text.split(","):
        token=token.strip()
        if token:
            v=float(token)
            if not 0.0 < v < 1.0: raise ValueError("Thresholds must lie in (0,1).")
            vals.append(v)
    return sorted(set(vals))


def break_summary_row(label: str, regime_fit: pd.DataFrame, extra: Optional[Dict[str,object]]=None) -> Dict[str,object]:
    table, summary=structural_break_test(regime_fit)
    row={"analysis":label,"n_periods":len(regime_fit),"n_transitions":max(len(regime_fit)-1,0),
         "break_available":bool(summary.get("available",False)),
         "best_break_after_period":summary.get("best_break_after_period",""),
         "first_post_break_period":summary.get("best_first_post_break_period",""),
         "delta_bic":summary.get("best_delta_bic_two_minus_homogeneous",np.nan)}
    if extra: row.update(extra)
    return row


def threshold_break_sensitivity(df: pd.DataFrame, cap: CapabilityBuildResult, period: str,
                                include_incomplete: bool, grid: Sequence[float]) -> pd.DataFrame:
    rows=[]
    settings=[("baseline",0.35,0.35)]
    for v in grid:
        settings.append((f"r2={v:.2f}",v,0.35))
        settings.append((f"r3={v:.2f}",0.35,v))
    seen=set()
    for label,r2,r3 in settings:
        key=(round(r2,6),round(r3,6))
        if key in seen: continue
        seen.add(key)
        ass=assign_regimes(df,cap,r2,r3,r3)
        agg=aggregate_regimes(df,ass,period)
        incomplete=last_period_is_incomplete(df,period)
        if incomplete and not include_incomplete: agg=agg.iloc[:-1].reset_index(drop=True)
        counts=ass["regime"].value_counts().reindex(REGIME_NAMES,fill_value=0)
        row=break_summary_row(label,agg,{"r2_threshold":r2,"r3_threshold":r3})
        for r in REGIME_NAMES: row[f"count_{r}"]=int(counts[r])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregation_break_sensitivity(df: pd.DataFrame, assignments: pd.DataFrame, include_incomplete: bool) -> pd.DataFrame:
    rows=[]
    for period in ["quarter","year"]:
        agg=aggregate_regimes(df,assignments,period)
        if last_period_is_incomplete(df,period) and not include_incomplete: agg=agg.iloc[:-1].reset_index(drop=True)
        rows.append(break_summary_row(period,agg,{"aggregation":period}))
    return pd.DataFrame(rows)


def provenance_masks(df: pd.DataFrame) -> Dict[str,np.ndarray]:
    masks={"full":np.ones(len(df),dtype=bool)}
    provcols=["parameters_provenance","training_compute_provenance","training_dataset_provenance"]
    if all(c in df.columns for c in provcols):
        noalg=np.ones(len(df),dtype=bool); strict=np.ones(len(df),dtype=bool)
        for c in provcols:
            vals=df[c].astype(str).to_numpy()
            noalg &= vals != "mice_pmm"
            strict &= vals == "strict_observed"
        masks["no_algorithmic_imputation"]=noalg; masks["all_strict_observed"]=strict
    else:
        flagcols=["parameters_algorithm_imputed_flag","training_compute_algorithm_imputed_flag","training_dataset_algorithm_imputed_flag"]
        if all(c in df.columns for c in flagcols):
            noalg=np.ones(len(df),dtype=bool)
            for c in flagcols: noalg &= pd.to_numeric(df[c],errors="coerce").fillna(1).to_numpy()==0
            masks["no_algorithmic_imputation"]=noalg
    return masks


def provenance_break_sensitivity(df: pd.DataFrame, period: str, include_incomplete: bool,
                                 r2: float, r3a: float, r3w: float) -> pd.DataFrame:
    rows=[]
    for label,mask in provenance_masks(df).items():
        sub=df.loc[mask].reset_index(drop=True)
        if len(sub)<30: continue
        capsub=build_continuous_capabilities(sub)
        ass=assign_regimes(sub,capsub,r2,r3a,r3w)
        agg=aggregate_regimes(sub,ass,period)
        if last_period_is_incomplete(sub,period) and not include_incomplete: agg=agg.iloc[:-1].reset_index(drop=True)
        row=break_summary_row(label,agg,{"n_models":len(sub)})
        counts=ass["regime"].value_counts().reindex(REGIME_NAMES,fill_value=0)
        for r in REGIME_NAMES: row[f"count_{r}"]=int(counts[r])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_break_summary(boot: pd.DataFrame, baseline_break: str, period_order: Sequence[str]) -> Dict[str,object]:
    if boot.empty: return {"bootstrap_replicates":0}
    col="delta_bic" if "delta_bic" in boot.columns else "best_delta_bic"
    return {"bootstrap_replicates":int(len(boot)),
            "bootstrap_tested_break_after_period":baseline_break,
            "bootstrap_fraction_delta_bic_lt_0":float(np.mean(boot[col]<0)),
            "bootstrap_fraction_delta_bic_lt_minus6":float(np.mean(boot[col]<-6)),
            "bootstrap_fraction_delta_bic_lt_minus10":float(np.mean(boot[col]<-10)),
            "bootstrap_delta_bic_median":float(np.median(boot[col])),
            "bootstrap_delta_bic_p05":float(np.quantile(boot[col],0.05)),
            "bootstrap_delta_bic_p95":float(np.quantile(boot[col],0.95))}


def _quarter_index(label: str) -> Optional[int]:
    try:
        if "Q" not in str(label): return None
        y,q=str(label).split("Q"); return int(y)*4+int(q)
    except Exception: return None


def reviewer_validation_summary(
    anp_comp: pd.DataFrame, markov_comp: pd.DataFrame, break_summary: Dict[str,object],
    threshold_sens: pd.DataFrame, aggregation_sens: pd.DataFrame, provenance_sens: pd.DataFrame,
) -> pd.DataFrame:
    """Create manuscript-ready answers to the three likely reviewer questions."""
    fulla=anp_comp.loc[anp_comp["model"]=="Full_ANP_nonlinear"].iloc[0]
    basea=anp_comp.loc[anp_comp["model"]=="No_ANP_network_baseline"].iloc[0]
    anp_supported=bool(fulla["rmse"]<basea["rmse"] and fulla["pseudo_bic"]<basea["pseudo_bic"])

    fullm=markov_comp.loc[markov_comp["model"]=="full_state_dependent_nonlinear"].iloc[0]
    basem=markov_comp.loc[markov_comp["model"]=="homogeneous"].iloc[0]
    markov_supported=bool(fullm["rmse"]<basem["rmse"] and fullm["pseudo_bic"]<basem["pseudo_bic"])

    baseline=str(break_summary.get("best_break_after_period","")); bidx=_quarter_index(baseline)
    thresh_stable=None
    if bidx is not None and not threshold_sens.empty:
        inds=threshold_sens["best_break_after_period"].map(_quarter_index)
        valid=inds.dropna()
        thresh_stable=bool(len(valid)>0 and np.all(np.abs(valid.to_numpy(float)-bidx)<=1))

    prov_stable=None
    if bidx is not None and not provenance_sens.empty:
        other=provenance_sens[provenance_sens["analysis"]!="full"]
        inds=other["best_break_after_period"].map(_quarter_index).dropna()
        prov_stable=bool(len(inds)>0 and np.all(np.abs(inds.to_numpy(float)-bidx)<=1))

    boot_n=int(break_summary.get("bootstrap_replicates",0) or 0)
    boot_support=None
    if boot_n>0:
        boot_support=bool(float(break_summary.get("bootstrap_fraction_delta_bic_lt_0",0))>=0.80)

    overall_break_robust=bool(thresh_stable is True and prov_stable is True and boot_support is True)
    if boot_n==0:
        break_status="not_confirmed_bootstrap_not_run"
    elif overall_break_robust:
        break_status="robust_under_prespecified_checks"
    else:
        break_status="exploratory_not_robustly_confirmed"

    return pd.DataFrame([
        {"reviewer_question":"Why ANP?","status":"supported" if anp_supported else "not_supported",
         "evidence":f"RMSE {fulla['rmse']:.6f} vs {basea['rmse']:.6f}; pseudo-BIC {fulla['pseudo_bic']:.3f} vs {basea['pseudo_bic']:.3f}."},
        {"reviewer_question":"Why nonlinear/non-homogeneous Markov?","status":"supported" if markov_supported else "not_supported",
         "evidence":f"Full RMSE {fullm['rmse']:.6f} vs homogeneous {basem['rmse']:.6f}; pseudo-BIC {fullm['pseudo_bic']:.3f} vs {basem['pseudo_bic']:.3f}."},
        {"reviewer_question":"Is the candidate break robust?","status":break_status,
         "evidence":f"Observed best break={baseline}; threshold_within_1_period={thresh_stable}; provenance_within_1_period={prov_stable}; bootstrap_support={boot_support}; bootstrap_n={boot_n}."},
    ])

# -----------------------------------------------------------------------------
# Outputs and plotting
# -----------------------------------------------------------------------------

def save_matrix(matrix: np.ndarray, path: Path, rows: Sequence[str], cols: Sequence[str]) -> None:
    pd.DataFrame(matrix, index=list(rows), columns=list(cols)).to_csv(path, encoding="utf-8-sig")


def capability_parameter_table(theta: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    p = unpack_capability_params(theta, len(names))
    rows = []
    for key in ["c", "rho", "gamma", "trend"]:
        vals = np.asarray(p[key])
        for i, n in enumerate(names):
            rows.append({"parameter": f"{key}::{n}", "value": float(vals[i])})
    rows.append({"parameter": "alpha_ANP_cross_capability", "value": float(p["alpha_anp"])})
    return pd.DataFrame(rows)


def regime_parameter_table(theta: np.ndarray, active_names: Sequence[str]) -> pd.DataFrame:
    p = unpack_regime_params(theta, len(active_names))
    rows = []
    for j, r in enumerate(REGIME_NAMES):
        rows.append({"parameter": f"destination_intercept::{r}", "value": float(np.asarray(p["b"])[j])})
        rows.append({"parameter": f"time_trend::{r}", "value": float(np.asarray(p["g"])[j])})
        rows.append({"parameter": f"ANP_readiness_effect::{r}", "value": float(np.asarray(p["readiness"])[j])})
    rows.append({"parameter": "kappa_regime_persistence", "value": float(p["kappa"])})
    rows.append({"parameter": "eta_destination_occupancy_feedback", "value": float(p["eta"])})
    return pd.DataFrame(rows)


def make_plots(
    capability_agg: pd.DataFrame,
    active_names: Sequence[str],
    equilibrium: np.ndarray,
    regime_agg: pd.DataFrame,
    P_final: np.ndarray,
    break_table: pd.DataFrame,
    outdir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(capability_agg))
    for c in active_names:
        ax.plot(x, capability_agg[c], marker="o", label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(capability_agg["period"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean capability level")
    ax.set_xlabel("Period")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(outdir / "continuous_capability_evolution.png", dpi=600); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(len(active_names)), equilibrium)
    ax.set_xticks(np.arange(len(active_names)))
    ax.set_xticklabels(active_names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Conditional equilibrium capability")
    fig.tight_layout(); fig.savefig(outdir / "capability_conditional_equilibrium.png", dpi=600); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(regime_agg))
    for r in REGIME_NAMES:
        ax.plot(x, regime_agg[r], marker="o", label=r)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_agg["period"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of models in exclusive regime")
    ax.set_xlabel("Period")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(outdir / "regime_distribution_evolution.png", dpi=600); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(P_final, aspect="auto")
    ax.set_xticks(np.arange(len(REGIME_NAMES))); ax.set_yticks(np.arange(len(REGIME_NAMES)))
    ax.set_xticklabels(REGIME_NAMES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(REGIME_NAMES, fontsize=10)
    ax.set_xlabel("Destination regime"); ax.set_ylabel("Source regime")
    fig.colorbar(im, ax=ax, label="Transition probability")
    fig.tight_layout(); fig.savefig(outdir / "regime_final_transition_matrix.png", dpi=600); plt.close(fig)

    if len(break_table):
        bt = break_table.sort_values("break_transition_index")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(len(bt)), bt["delta_bic_two_minus_homogeneous"], marker="o")
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(np.arange(len(bt)))
        ax.set_xticklabels(bt["break_after_period"], rotation=45, ha="right")
        ax.set_ylabel("Delta BIC: two-regime minus homogeneous")
        ax.set_xlabel("Candidate break after period")
        fig.tight_layout(); fig.savefig(outdir / "structural_break_bic.png", dpi=600); plt.close(fig)


def make_manuscript_validation_plots(
    anp_comp: pd.DataFrame,
    markov_comp: pd.DataFrame,
    threshold_sens: pd.DataFrame,
    aggregation_sens: pd.DataFrame,
    provenance_sens: pd.DataFrame,
    boot: pd.DataFrame,
    outdir: Path,
) -> None:
    """Create the four reviewer-validation figures referenced by the manuscript.

    These complement ``make_plots`` and use the exact filenames included by
    ``applied_intelligence.tex``.
    """
    # Figure 2: ANP/network ablation.
    if not anp_comp.empty:
        order = ["Full_ANP_nonlinear", "No_ANP_network_baseline"]
        tmp = anp_comp.set_index("model").reindex(order).dropna(subset=["rmse"]).reset_index()
        labels = ["Full ANP nonlinear", "No-ANP baseline"][:len(tmp)]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x = np.arange(len(tmp))
        vals = tmp["rmse"].to_numpy(float)
        bars = ax.bar(x, vals)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("One-step RMSE")
        upper = max(vals) * 1.22 if len(vals) else 1.0
        ax.set_ylim(0.0, upper)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v + upper*0.015, f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout(); fig.savefig(outdir / "anp_ablation_rmse.png", dpi=600); plt.close(fig)

    # Figure 3: homogeneous, time-varying, and state-dependent Markov comparison.
    if not markov_comp.empty:
        order = ["homogeneous", "time_varying", "full_state_dependent_nonlinear"]
        tmp = markov_comp.set_index("model").reindex(order).dropna(subset=["rmse"]).reset_index()
        labels = ["Homogeneous", "Time-varying", "State-dependent\nnonlinear"][:len(tmp)]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x = np.arange(len(tmp))
        vals = tmp["rmse"].to_numpy(float)
        bars = ax.bar(x, vals)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("One-step RMSE")
        upper = max(vals) * 1.20 if len(vals) else 1.0
        ax.set_ylim(0.0, upper)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v + upper*0.015, f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout(); fig.savefig(outdir / "markov_baseline_rmse.png", dpi=600); plt.close(fig)

    # Figure 7: structural-break sensitivity across prespecified specifications.
    sensitivity_rows = []
    if not threshold_sens.empty:
        wanted = {
            "baseline": "baseline",
            "r2=0.30": "r2=0.30",
            "r3=0.30": "r3=0.30",
            "r2=0.40": "r2=0.40",
            "r3=0.40": "r3=0.40",
        }
        for key, label in wanted.items():
            hit = threshold_sens.loc[threshold_sens["analysis"].astype(str) == key]
            if len(hit): sensitivity_rows.append((label, float(hit.iloc[0]["delta_bic"])))
    if not aggregation_sens.empty:
        hit = aggregation_sens.loc[aggregation_sens["aggregation"].astype(str) == "year"]
        if len(hit): sensitivity_rows.append(("annual aggregation", float(hit.iloc[0]["delta_bic"])))
    if not provenance_sens.empty:
        for key, label in [
            ("no_algorithmic_imputation", "no algorithmic imputation"),
            ("all_strict_observed", "all strict observed"),
        ]:
            hit = provenance_sens.loc[provenance_sens["analysis"].astype(str) == key]
            if len(hit): sensitivity_rows.append((label, float(hit.iloc[0]["delta_bic"])))
    if sensitivity_rows:
        labels = [x[0] for x in sensitivity_rows]
        vals = np.asarray([x[1] for x in sensitivity_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        y = np.arange(len(labels))
        ax.barh(y, vals)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel(r"$\Delta$ pseudo-BIC (two-segment minus homogeneous)")
        fig.tight_layout(); fig.savefig(outdir / "break_robustness_sensitivity.png", dpi=600); plt.close(fig)

    # Figure 8: fixed-candidate bootstrap distribution.
    if not boot.empty:
        col = "delta_bic" if "delta_bic" in boot.columns else "best_delta_bic"
        vals = pd.to_numeric(boot[col], errors="coerce").dropna().to_numpy(float)
        if len(vals):
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            ax.hist(vals, bins=min(14, max(8, int(np.sqrt(len(vals))))))
            ax.axvline(0.0, linewidth=1)
            ax.set_xlabel(r"Bootstrap $\Delta$ pseudo-BIC")
            ax.set_ylabel("Frequency")
            fig.tight_layout(); fig.savefig(outdir / "break_bootstrap_delta_bic.png", dpi=600); plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    csv_path = locate_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/12] Loading {csv_path}")
    df = pd.read_csv(csv_path)
    validate_input(df)
    df = df.sort_values(["publication_date", "model"]).reset_index(drop=True)
    print(f"      models={len(df)}, required-column missing cells=0")

    print("[2/12] Constructing continuous non-exclusive capability dimensions")
    cap = build_continuous_capabilities(df)
    cap_out = df[["model", "publication_date", "year"]].join(cap.values)
    cap_out.to_csv(outdir / "model_continuous_capabilities.csv", index=False, encoding="utf-8-sig")
    cap.metadata.to_csv(outdir / "capability_measurement_metadata.csv", index=False, encoding="utf-8-sig")
    cap.proxy_weights.to_csv(outdir / "proxy_construction_weights.csv", index=False, encoding="utf-8-sig")
    inspect_cols = [
        "Reasoning", "Reasoning_Inspectability", "Reasoning_Faithfulness"
    ]
    df[["model", "publication_date", "year"]].join(cap.values[inspect_cols]).to_csv(
        outdir / "model_reasoning_inspectability.csv", index=False, encoding="utf-8-sig"
    )
    print(f"      active capabilities: {', '.join(cap.active_names)}")
    print("      proxy weighting: equal (fully de-nested SRWA)")
    if not cap.proxy_weights.empty:
        for capability_name, g in cap.proxy_weights.groupby("capability", sort=False):
            desc = ", ".join(f"{r.component}={r.weight:.4f}" for r in g.itertuples())
            print(f"      {capability_name} weights: {desc}")
    print(f"      reasoning inspectability directly measured: {cap.direct_reasoning_inspectability_available}")
    print(f"      reasoning faithfulness directly measured: {cap.direct_reasoning_faithfulness_available}")

    print("[3/12] Estimating data-driven ANP interdependence")
    raw_mi, W, Wlim, cross = build_empirical_anp(
        cap.values, cap.active_names, args.anp_self_loop, args.seed
    )
    save_matrix(raw_mi, outdir / "anp_raw_mutual_information.csv", cap.active_names, cap.active_names)
    save_matrix(W, outdir / "anp_supermatrix_column_stochastic.csv", cap.active_names, cap.active_names)
    save_matrix(Wlim, outdir / "anp_limit_matrix.csv", cap.active_names, cap.active_names)
    save_matrix(cross, outdir / "anp_cross_capability_influence.csv", cap.active_names, cap.active_names)
    weights = Wlim.mean(axis=1); weights = weights / weights.sum()
    pd.DataFrame({"capability": cap.active_names, "anp_limit_weight": weights}).to_csv(
        outdir / "anp_limit_weights.csv", index=False, encoding="utf-8-sig"
    )

    print(f"[4/12] Aggregating capabilities by {args.period}")
    cap_agg_all = aggregate_capabilities(df, cap.values, cap.active_names, args.period)
    incomplete_last = last_period_is_incomplete(df, args.period)
    cap_agg_all["used_for_dynamic_fit"] = True
    if incomplete_last and not args.include_incomplete_last_period:
        cap_agg_all.loc[cap_agg_all.index[-1], "used_for_dynamic_fit"] = False
    cap_agg_all.to_csv(outdir / "capability_period_means.csv", index=False, encoding="utf-8-sig")
    reasoning_aux = aggregate_reasoning_interpretability(df, cap.values, args.period)
    reasoning_aux.to_csv(
        outdir / "reasoning_inspectability_period_means.csv", index=False, encoding="utf-8-sig"
    )
    plot_reasoning_interpretability(reasoning_aux, outdir)
    cap_fit = cap_agg_all[cap_agg_all["used_for_dynamic_fit"]].reset_index(drop=True)
    print(f"      periods={len(cap_agg_all)}, fitted periods={len(cap_fit)}, final period incomplete={incomplete_last}")

    print("[5/12] Fitting nonlinear continuous capability dynamics")
    cap_theta, cap_diag, cap_pred = fit_capability_dynamics(
        cap_fit, cap.active_names, cross, args.ridge_capability, args.starts, args.seed
    )
    capability_parameter_table(cap_theta, cap.active_names).to_csv(
        outdir / "capability_dynamic_parameters.csv", index=False, encoding="utf-8-sig"
    )
    cap_pred.to_csv(outdir / "capability_one_step_predictions.csv", index=False, encoding="utf-8-sig")
    x0 = cap_fit[cap.active_names].iloc[-1].to_numpy(float)
    xstar, cap_fp_iter, cap_fp_resid = capability_fixed_point(x0, cap_theta, cross)
    cap_eig, cap_rho = full_jacobian_spectral_radius(xstar, cap_theta, cross)
    pd.DataFrame({"capability": cap.active_names, "conditional_equilibrium": xstar}).to_csv(
        outdir / "capability_conditional_equilibrium.csv", index=False, encoding="utf-8-sig"
    )

    print("[6/12] Running ANP/network ablation")
    anp_comp, noanp_pred = capability_model_comparison(
        cap_fit, cap.active_names, cross, cap_theta, cap_diag,
        args.ridge_capability, max(2,args.starts), args.seed
    )
    anp_comp.to_csv(outdir / "anp_ablation_comparison.csv", index=False, encoding="utf-8-sig")
    noanp_pred.to_csv(outdir / "capability_no_anp_predictions.csv", index=False, encoding="utf-8-sig")

    print("[7/12] Assigning mutually exclusive technological regimes")
    assignments = assign_regimes(
        df, cap, args.r2_threshold, args.r3_agency_threshold,
        args.r3_world_threshold,
    )
    assignments.to_csv(outdir / "model_regime_assignments.csv", index=False, encoding="utf-8-sig")
    regime_agg_all = aggregate_regimes(df, assignments, args.period)
    regime_agg_all["used_for_dynamic_fit"] = True
    if incomplete_last and not args.include_incomplete_last_period:
        regime_agg_all.loc[regime_agg_all.index[-1], "used_for_dynamic_fit"] = False
    regime_agg_all.to_csv(outdir / "regime_period_distributions.csv", index=False, encoding="utf-8-sig")
    regime_fit = regime_agg_all[regime_agg_all["used_for_dynamic_fit"]].drop(columns="used_for_dynamic_fit").reset_index(drop=True)
    cap_fit_markov = cap_fit.drop(columns="used_for_dynamic_fit", errors="ignore")
    counts = assignments["regime"].value_counts().reindex(REGIME_NAMES, fill_value=0)
    print("      model counts by regime:")
    for r, n in counts.items():
        print(f"        {r}: {int(n)}")

    print("[8/12] Fitting non-homogeneous technological-regime Markov model")
    reg_theta, reg_diag, reg_pred, reg_trans = fit_regime_markov(
        regime_fit, cap_fit_markov, cap.active_names, weights,
        args.ridge_regime, args.starts, args.seed,
    )
    regime_parameter_table(reg_theta, cap.active_names).to_csv(
        outdir / "regime_markov_parameters.csv", index=False, encoding="utf-8-sig"
    )
    reg_pred.to_csv(outdir / "regime_one_step_predictions.csv", index=False, encoding="utf-8-sig")
    reg_trans.to_csv(outdir / "regime_transition_matrices_long.csv", index=False, encoding="utf-8-sig")

    q_final = regime_fit[REGIME_NAMES].iloc[-1].to_numpy(float)
    x_final = cap_fit_markov[cap.active_names].iloc[-1].to_numpy(float)
    P_final = regime_transition_matrix(q_final, x_final, 1.0, reg_theta, weights)
    save_matrix(P_final, outdir / "regime_final_transition_matrix.csv", REGIME_NAMES, REGIME_NAMES)

    print("[9/12] Comparing homogeneous, time-varying, and full Markov models")
    markov_comp = regime_model_comparison(
        regime_fit, reg_diag, cap.active_names, args.ridge_regime, max(2,args.starts), args.seed
    )
    markov_comp.to_csv(outdir / "markov_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    denested_summary = pd.DataFrame(); denested_detail: Dict[str, object] = {}

    print("[10/12] Testing structural break in regime-transition dynamics")
    break_table, break_summary = structural_break_test(regime_fit)
    break_table.to_csv(outdir / "structural_break_test.csv", index=False, encoding="utf-8-sig")
    boot = bootstrap_break_stability(regime_fit, args.break_bootstrap, args.seed)
    if len(boot):
        boot.to_csv(outdir / "structural_break_bootstrap.csv", index=False, encoding="utf-8-sig")
    break_summary.update(bootstrap_break_summary(
        boot, str(break_summary.get("best_break_after_period", "")), regime_fit["period"].astype(str).tolist()
    ))

    print("[11/12] Running publication robustness analyses")
    threshold_sens = pd.DataFrame(); aggregation_sens = pd.DataFrame(); provenance_sens = pd.DataFrame()
    if not args.skip_publication_robustness:
        grid=parse_threshold_grid(args.threshold_grid)
        threshold_sens=threshold_break_sensitivity(
            df,cap,args.period,args.include_incomplete_last_period,grid
        )
        threshold_sens.to_csv(outdir / "break_robustness_thresholds.csv",index=False,encoding="utf-8-sig")
        aggregation_sens=aggregation_break_sensitivity(
            df,assignments,args.include_incomplete_last_period
        )
        aggregation_sens.to_csv(outdir / "break_robustness_aggregation.csv",index=False,encoding="utf-8-sig")
        provenance_sens=provenance_break_sensitivity(
            df,args.period,args.include_incomplete_last_period,args.r2_threshold,
            args.r3_agency_threshold,args.r3_world_threshold
        )
        provenance_sens.to_csv(outdir / "break_robustness_provenance.csv",index=False,encoding="utf-8-sig")

    reviewer_summary = reviewer_validation_summary(
        anp_comp, markov_comp, break_summary, threshold_sens, aggregation_sens, provenance_sens
    )
    reviewer_summary.to_csv(outdir / "reviewer_validation_summary.csv",index=False,encoding="utf-8-sig")

    print("[12/12] Verification and diagnostics")
    cap_diag.update({
        "fixed_point_iterations": int(cap_fp_iter),
        "fixed_point_residual": float(cap_fp_resid),
        "jacobian_spectral_radius": float(cap_rho),
        "locally_stable_if_spectral_radius_lt_1": bool(cap_rho < 1.0),
    })

    # Final Markov one-step prediction is a proper probability distribution.
    final_pred = q_final @ P_final
    checks = {
        "required input columns have no missing values": not df[REQUIRED_COLUMNS].isna().any().any(),
        "active capability matrix is finite": bool(np.isfinite(cap.values[cap.active_names].to_numpy(float)).all()),
        "reasoning inspectability auxiliary score is finite": bool(np.isfinite(cap.values["Reasoning_Inspectability"].to_numpy(float)).all()),
        "reasoning faithfulness is not fabricated when unmeasured": bool(
            cap.direct_reasoning_faithfulness_available
            or cap.values["Reasoning_Faithfulness"].isna().all()
        ),
        "ANP supermatrix is column stochastic": bool(np.allclose(W.sum(axis=0), 1.0, atol=1e-10)),
        "capability equilibrium is finite": bool(np.isfinite(xstar).all()),
        "capability equilibrium remains in [0,1]": bool(np.all((xstar >= 0) & (xstar <= 1))),
        "capability fixed-point residual < 1e-8": bool(cap_fp_resid < 1e-8),
        "regime final transition matrix is row stochastic": bool(np.allclose(P_final.sum(axis=1), 1.0, atol=1e-10)),
        "regime final prediction sums to one": bool(np.isclose(final_pred.sum(), 1.0, atol=1e-10)),
        "regime probabilities are finite": bool(np.isfinite(P_final).all()),
    }

    diagnostics = {
        "input": {
            "csv": str(csv_path),
            "n_models": int(len(df)),
            "period": args.period,
            "last_period_incomplete": bool(incomplete_last),
            "included_incomplete_last_period": bool(args.include_incomplete_last_period),
        },
        "measurement": {
            "active_capabilities": cap.active_names,
            "proxy_weighting": "equal_de_nested",
            "proxy_construction_weights": cap.proxy_weights.to_dict(orient="records"),
            "reasoning_inspectability_directly_measured": bool(cap.direct_reasoning_inspectability_available),
            "reasoning_faithfulness_directly_measured": bool(cap.direct_reasoning_faithfulness_available),
            "reasoning_inspectability_used_in_core_dynamics": False,
            "reasoning_centered_regime_is_equivalent_to_xai": False,
            "regime_model_counts": {r: int(counts[r]) for r in REGIME_NAMES},
            "warning": (
                "Reasoning inspectability is an auxiliary observable/proxy and is not a faithful explanation measure. "
                "Visible or verbalized reasoning must not be equated with mechanistic XAI."
            ),
        },
        "capability_dynamics": cap_diag,
        "capability_jacobian_eigenvalues": [
            {"real": float(np.real(z)), "imag": float(np.imag(z))} for z in cap_eig
        ],
        "regime_markov": reg_diag,
        "reviewer_validation": {
            "anp_ablation": anp_comp.to_dict(orient="records"),
            "markov_baselines": markov_comp.to_dict(orient="records"),
            "threshold_break_sensitivity": threshold_sens.to_dict(orient="records"),
            "aggregation_break_sensitivity": aggregation_sens.to_dict(orient="records"),
            "provenance_break_sensitivity": provenance_sens.to_dict(orient="records"),
            "manuscript_ready_summary": reviewer_summary.to_dict(orient="records"),
            "denested_proxy_robustness": denested_summary.to_dict(orient="records"),
        },
        "structural_break": break_summary,
        "verification": checks,
    }
    with open(outdir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    make_plots(cap_agg_all, cap.active_names, xstar, regime_agg_all, P_final, break_table, outdir)
    make_manuscript_validation_plots(
        anp_comp, markov_comp, threshold_sens, aggregation_sens, provenance_sens, boot, outdir
    )

    for name, ok in checks.items():
        print(f"      {'PASS' if ok else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise RuntimeError("One or more verification checks failed; inspect diagnostics.json.")

    print("\nReasoning inspectability boundary")
    print(f"  direct inspectability measurement available = {cap.direct_reasoning_inspectability_available}")
    print(f"  direct faithfulness measurement available = {cap.direct_reasoning_faithfulness_available}")
    print("  reasoning-centered regime is NOT treated as equivalent to XAI")

    print("\nContinuous-capability model")
    print(f"  in-sample one-step reconstruction RMSE = {cap_diag['rmse_one_step']:.6f}")
    print(f"  in-sample one-step reconstruction MAE  = {cap_diag['mae_one_step']:.6f}")
    print(f"  fixed-point residual = {cap_fp_resid:.3e}")
    print(f"  Jacobian spectral radius = {cap_rho:.6f}")
    print("  conditional equilibrium:")
    for c, v in zip(cap.active_names, xstar):
        print(f"    {c}: {v:.6f}")

    print("\nReviewer validation: Why ANP?")
    print(anp_comp.to_string(index=False))

    print("\nReviewer validation: Why nonlinear/non-homogeneous Markov?")
    print(markov_comp.to_string(index=False))

    if len(denested_summary):
        print("\nReviewer robustness: de-nested capability proxies (baseline regime labels fixed)")
        print(denested_summary.to_string(index=False))

    print("\nReviewer-ready conclusions")
    print(reviewer_summary.to_string(index=False))

    print("\nRegime Markov model")
    print(f"  in-sample one-step reconstruction RMSE = {reg_diag['rmse_one_step']:.6f}")
    print(f"  in-sample one-step reconstruction MAE  = {reg_diag['mae_one_step']:.6f}")
    if break_summary.get("available"):
        print(f"  best structural break after: {break_summary['best_break_after_period']}")
        print(f"  delta BIC (two-regime - homogeneous) = {break_summary['best_delta_bic_two_minus_homogeneous']:.6f}")
    print(f"\nAll results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
