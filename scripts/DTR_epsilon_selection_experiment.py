"""
Experiment plan for studying the position of epsilon-selection in backward Q-learning.

This module is designed to be placed at the root of the repository:

    Multiple_Policies_Qlearning/
        epsilon_selection_experiment.py
        scripts/
            DTR_Cancer_DataGeneration.py
            NearEquivalentQlearning.py
        results/

The notebook should import this module, run the experiment plan, and call the
plotting functions defined below.
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# ============================================================
# Experiment configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """Container for all experiment parameters."""

    n_train: int = 2000
    n_test: int = 1000
    n_replications: int = 10
    epsilon_selection: float = 0.3
    t_values: List[int] = field(default_factory=lambda: [5, 7, 8, 9, 10, 12])
    c0_death: float = -12.0
    possible_treatments: np.ndarray = field(
        default_factory=lambda: np.round(np.arange(0.1, 1.1, 0.1), 1)
    )
    seed: int = 42
    scripts_dir: str = "scripts"
    results_dir: str = "results"


# ============================================================
# Setup utilities
# ============================================================

def set_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""

    random.seed(seed)
    np.random.seed(seed)


def import_project_functions(scripts_dir: str = "scripts"):
    """Import project-specific simulation and Q-learning functions."""

    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)

    from DTR_Cancer_DataGeneration import (  # pylint: disable=import-error
        generate_dataset,
        calculate_rewards,
        create_data_stage_frames,
    )
    from NearEquivalentQlearning import (  # pylint: disable=import-error
        classicalQlearning,
        train_last_stage,
        select_values_within_epsilon,
    )

    return {
        "generate_dataset": generate_dataset,
        "calculate_rewards": calculate_rewards,
        "create_data_stage_frames": create_data_stage_frames,
        "classicalQlearning": classicalQlearning,
        "train_last_stage": train_last_stage,
        "select_values_within_epsilon": select_values_within_epsilon,
    }


def horizon_results_dir(base_results_dir: str, t_final: int, c0_death: float) -> str:
    """Build a stable directory name for a given horizon and death intercept."""

    c0_label = str(c0_death).replace("-", "m").replace(".", "p")
    return os.path.join(base_results_dir, f"T{t_final}_c0{c0_label}")


# ============================================================
# Core numerical helpers
# ============================================================

def summarize_selected_values(selected_values_array: Iterable[Iterable[float]],
                              total_n_treatments: int) -> Tuple[Dict[int, float], Dict[str, float]]:
    """Summarize admissible-set cardinalities for one stage."""

    n_admissible = np.array([len(x) for x in selected_values_array])
    total_patients = len(n_admissible)

    counts = Counter(n_admissible)
    percentages = {k: 100 * v / total_patients for k, v in sorted(counts.items())}

    summary = {
        "mean_n_admissible": float(np.mean(n_admissible)),
        "sd_n_admissible": float(np.std(n_admissible, ddof=1)) if total_patients > 1 else 0.0,
        "median_n_admissible": float(np.median(n_admissible)),
        "min_n_admissible": int(np.min(n_admissible)),
        "max_n_admissible": int(np.max(n_admissible)),
        "pct_1_treatment": 100 * np.mean(n_admissible == 1),
        "pct_2plus_treatments": 100 * np.mean(n_admissible >= 2),
        "pct_all_treatments": 100 * np.mean(n_admissible == total_n_treatments),
    }

    return percentages, summary


def build_predicted_matrix_for_stage(stage_idx: int,
                                     q_models: List,
                                     data_stages: Dict[str, pd.DataFrame],
                                     possible_treatments: np.ndarray) -> np.ndarray:
    """Build a treatments-by-patients matrix of predicted Q-values for one stage."""

    data_key = f"Data_Stage_{stage_idx}"
    dosage_column = f"Dosage_{stage_idx}"
    model = q_models[stage_idx]

    n_patients = len(data_stages[data_key])
    pred_mat = np.zeros((len(possible_treatments), n_patients))

    for j, treatment in enumerate(possible_treatments):
        stage_data = data_stages[data_key].copy()
        stage_data[dosage_column] = treatment
        pred_mat[j, :] = model.predict(stage_data)

    return pred_mat


def build_predicted_matrix_last_stage(last_stage_model,
                                      data_stages: Dict[str, pd.DataFrame],
                                      possible_treatments: np.ndarray,
                                      last_stage_idx: int) -> np.ndarray:
    """Build a treatments-by-patients matrix of predicted Q-values for the last stage."""

    data_key = f"Data_Stage_{last_stage_idx}"
    dosage_column = f"Dosage_{last_stage_idx}"

    n_patients = len(data_stages[data_key])
    pred_mat = np.zeros((len(possible_treatments), n_patients))

    for j, treatment in enumerate(possible_treatments):
        stage_data = data_stages[data_key].copy()
        stage_data[dosage_column] = treatment
        pred_mat[j, :] = last_stage_model.predict(stage_data)

    return pred_mat


def summarize_q_value_separation(pred_mat: np.ndarray) -> Dict[str, float]:
    """Compute stage-specific Q-value separation diagnostics."""

    q_sorted = np.sort(pred_mat, axis=0)
    q_range = pred_mat.max(axis=0) - pred_mat.min(axis=0)
    q_gap_top2 = q_sorted[-1, :] - q_sorted[-2, :]

    return {
        "mean_q_range": float(np.mean(q_range)),
        "median_q_range": float(np.median(q_range)),
        "sd_q_range": float(np.std(q_range, ddof=1)),
        "mean_top2_gap": float(np.mean(q_gap_top2)),
        "median_top2_gap": float(np.median(q_gap_top2)),
        "sd_top2_gap": float(np.std(q_gap_top2, ddof=1)),
        "mean_q_max": float(np.mean(pred_mat.max(axis=0))),
        "mean_q_min": float(np.mean(pred_mat.min(axis=0))),
        "mean_q_sd_across_treatments": float(np.mean(np.std(pred_mat, axis=0, ddof=1))),
    }


# ============================================================
# One replication and one horizon
# ============================================================

def run_one_replication(rep_id: int,
                        t_final: int,
                        config: ExperimentConfig,
                        project_functions: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one train/test replication for a fixed horizon."""

    number_of_decision = t_final - 1
    number_of_decision_rules = t_final - 1
    stages_all = list(range(t_final))

    generate_dataset = project_functions["generate_dataset"]
    calculate_rewards = project_functions["calculate_rewards"]
    create_data_stage_frames = project_functions["create_data_stage_frames"]
    train_last_stage = project_functions["train_last_stage"]
    classicalQlearning = project_functions["classicalQlearning"]
    select_values_within_epsilon = project_functions["select_values_within_epsilon"]

    # Training cohort.
    data_raw_train, remission_info_train, dead_info_train = generate_dataset(
        T_final=t_final,
        N=config.n_train,
        c0=config.c0_death,
    )

    rewards_train = calculate_rewards(
        data_raw_train,
        remission_info_train,
        dead_info_train,
        T_final=t_final,
    )

    data_stages_train = create_data_stage_frames(data_raw_train, number_of_decision)

    last_stage_model, _ = train_last_stage(
        number_of_decision_rules,
        data_stages_train,
        rewards_train,
        config.possible_treatments,
    )

    q_models = classicalQlearning(
        number_of_decision,
        data_stages_train,
        rewards_train,
        config.possible_treatments,
    )

    # Test cohort.
    data_raw_test, _, _ = generate_dataset(
        T_final=t_final,
        N=config.n_test,
        c0=config.c0_death,
    )
    data_stages_test = create_data_stage_frames(data_raw_test, number_of_decision)

    summary_rows = []
    pct_rows = []
    qscale_rows = []

    for stage_idx in stages_all:
        if stage_idx == number_of_decision:
            pred_mat = build_predicted_matrix_last_stage(
                last_stage_model,
                data_stages_test,
                config.possible_treatments,
                last_stage_idx=number_of_decision,
            )
        else:
            pred_mat = build_predicted_matrix_for_stage(
                stage_idx,
                q_models,
                data_stages_test,
                config.possible_treatments,
            )

        # Q-value separation diagnostics.
        qscale_rows.append({
            "T_final": t_final,
            "replication": rep_id,
            "n_train": config.n_train,
            "n_test": config.n_test,
            "c0_death": config.c0_death,
            "epsilon_selection": config.epsilon_selection,
            "stage": stage_idx,
            **summarize_q_value_separation(pred_mat),
        })

        # Epsilon-selection summaries.
        selected_values = select_values_within_epsilon(pred_mat, config.epsilon_selection)
        pct_stage, summary_stage = summarize_selected_values(
            selected_values,
            total_n_treatments=len(config.possible_treatments),
        )

        summary_rows.append({
            "T_final": t_final,
            "replication": rep_id,
            "n_train": config.n_train,
            "n_test": config.n_test,
            "c0_death": config.c0_death,
            "epsilon_selection": config.epsilon_selection,
            "stage": stage_idx,
            **summary_stage,
        })

        for n_adm, percentage in pct_stage.items():
            pct_rows.append({
                "T_final": t_final,
                "replication": rep_id,
                "n_train": config.n_train,
                "n_test": config.n_test,
                "c0_death": config.c0_death,
                "epsilon_selection": config.epsilon_selection,
                "stage": stage_idx,
                "n_admissible_treatments": n_adm,
                "percentage": percentage,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(pct_rows), pd.DataFrame(qscale_rows)


def aggregate_horizon_results(summary_by_rep: pd.DataFrame,
                              pct_by_rep: pd.DataFrame,
                              qscale_by_rep: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Aggregate replication-level outputs for one horizon."""

    group_keys = ["T_final", "c0_death", "epsilon_selection", "stage"]

    summary_agg = (
        summary_by_rep
        .groupby(group_keys, as_index=False)
        .agg(
            mean_n_admissible_mean=("mean_n_admissible", "mean"),
            mean_n_admissible_sd=("mean_n_admissible", "std"),
            mean_n_admissible_min=("mean_n_admissible", "min"),
            mean_n_admissible_max=("mean_n_admissible", "max"),
            sd_n_admissible_mean=("sd_n_admissible", "mean"),
            median_n_admissible_mean=("median_n_admissible", "mean"),
            min_n_admissible_mean=("min_n_admissible", "mean"),
            max_n_admissible_mean=("max_n_admissible", "mean"),
            pct_1_treatment_mean=("pct_1_treatment", "mean"),
            pct_1_treatment_sd=("pct_1_treatment", "std"),
            pct_2plus_treatments_mean=("pct_2plus_treatments", "mean"),
            pct_2plus_treatments_sd=("pct_2plus_treatments", "std"),
            pct_all_treatments_mean=("pct_all_treatments", "mean"),
            pct_all_treatments_sd=("pct_all_treatments", "std"),
        )
        .sort_values(["T_final", "stage"])
        .reset_index(drop=True)
    )

    pct_agg = (
        pct_by_rep
        .groupby(group_keys + ["n_admissible_treatments"], as_index=False)
        .agg(
            mean_percentage=("percentage", "mean"),
            sd_percentage=("percentage", "std"),
        )
        .sort_values(["T_final", "stage", "n_admissible_treatments"])
        .reset_index(drop=True)
    )

    qscale_agg = (
        qscale_by_rep
        .groupby(group_keys, as_index=False)
        .agg(
            mean_q_range_mean=("mean_q_range", "mean"),
            mean_q_range_sd=("mean_q_range", "std"),
            median_q_range_mean=("median_q_range", "mean"),
            median_q_range_sd=("median_q_range", "std"),
            mean_top2_gap_mean=("mean_top2_gap", "mean"),
            mean_top2_gap_sd=("mean_top2_gap", "std"),
            median_top2_gap_mean=("median_top2_gap", "mean"),
            median_top2_gap_sd=("median_top2_gap", "std"),
            mean_q_sd_across_treatments_mean=("mean_q_sd_across_treatments", "mean"),
            mean_q_sd_across_treatments_sd=("mean_q_sd_across_treatments", "std"),
            mean_q_max_mean=("mean_q_max", "mean"),
            mean_q_min_mean=("mean_q_min", "mean"),
        )
        .sort_values(["T_final", "stage"])
        .reset_index(drop=True)
    )

    return {
        "summary_agg": summary_agg,
        "pct_agg": pct_agg,
        "qscale_agg": qscale_agg,
    }


def run_horizon(t_final: int,
                config: ExperimentConfig,
                project_functions: Dict,
                verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run all replications for one horizon and save horizon-specific outputs."""

    out_dir = horizon_results_dir(config.results_dir, t_final, config.c0_death)
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print("=" * 80)
        print(
            f"Running T_final={t_final} | train={config.n_train} | "
            f"test={config.n_test} | replications={config.n_replications} | "
            f"epsilon={config.epsilon_selection} | c0={config.c0_death}"
        )
        print(f"Output directory: {out_dir}")
        print("=" * 80)

    all_summary = []
    all_pct = []
    all_qscale = []

    for rep_id in range(1, config.n_replications + 1):
        if verbose:
            print(f"Replication {rep_id}/{config.n_replications}")

        summary_rep, pct_rep, qscale_rep = run_one_replication(
            rep_id=rep_id,
            t_final=t_final,
            config=config,
            project_functions=project_functions,
        )

        all_summary.append(summary_rep)
        all_pct.append(pct_rep)
        all_qscale.append(qscale_rep)

    summary_by_rep = pd.concat(all_summary, ignore_index=True)
    pct_by_rep = pd.concat(all_pct, ignore_index=True)
    qscale_by_rep = pd.concat(all_qscale, ignore_index=True)

    aggregated = aggregate_horizon_results(summary_by_rep, pct_by_rep, qscale_by_rep)

    # Save replication-level and aggregated outputs.
    summary_by_rep.to_csv(os.path.join(out_dir, "summary_by_rep.csv"), index=False)
    pct_by_rep.to_csv(os.path.join(out_dir, "pct_by_rep.csv"), index=False)
    qscale_by_rep.to_csv(os.path.join(out_dir, "qscale_by_rep.csv"), index=False)
    aggregated["summary_agg"].to_csv(os.path.join(out_dir, "summary_aggregated.csv"), index=False)
    aggregated["pct_agg"].to_csv(os.path.join(out_dir, "pct_aggregated.csv"), index=False)
    aggregated["qscale_agg"].to_csv(os.path.join(out_dir, "qscale_aggregated.csv"), index=False)

    return {
        "T_final": t_final,
        "results_dir": out_dir,
        "summary_by_rep": summary_by_rep,
        "pct_by_rep": pct_by_rep,
        "qscale_by_rep": qscale_by_rep,
        **aggregated,
    }


def run_experiment_plan(config: ExperimentConfig, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run the complete experiment plan over all horizons."""

    os.makedirs(config.results_dir, exist_ok=True)

    set_seed(config.seed)
    project_functions = import_project_functions(config.scripts_dir)

    if verbose:
        print("\n" + "=" * 80)
        print("RUNNING COMPLETE EXPERIMENT PLAN")
        print("=" * 80)
        print(f"T horizons : {config.t_values}")
        print(f"Replications: {config.n_replications}")
        print(f"Results dir: {config.results_dir}")
        print("=" * 80)

    horizon_outputs = []

    for i, t_final in enumerate(config.t_values, start=1):

        if verbose:
            print(f"\n[{i}/{len(config.t_values)}] Running T_final = {t_final}")

        out = run_horizon(
            t_final=t_final,
            config=config,
            project_functions=project_functions,
            verbose=False,   # <- avoid spam from nested function
        )

        horizon_outputs.append(out)

        if verbose:
            final_stage = (
                out["summary_agg"]
                .loc[lambda d: d["stage"] == d["T_final"] - 1]
            )

            mean_adm = final_stage["mean_n_admissible_mean"].iloc[0]
            pct_single = final_stage["pct_1_treatment_mean"].iloc[0]

            print(
                f"   done | mean admissible = {mean_adm:.2f} "
                f"| singleton = {pct_single:.1f}%"
            )

    # ------------------------------------------------------------------
    # Aggregate outputs across horizons
    # ------------------------------------------------------------------

    comparison_summary = pd.concat(
        [out["summary_agg"] for out in horizon_outputs],
        ignore_index=True,
    )

    comparison_qscale = pd.concat(
        [out["qscale_agg"] for out in horizon_outputs],
        ignore_index=True,
    )

    comparison_pct = pd.concat(
        [out["pct_agg"] for out in horizon_outputs],
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Stage indexing
    # ------------------------------------------------------------------

    for df in [comparison_summary, comparison_qscale, comparison_pct]:

        df["stage"] = df["stage"] + 1

        df["relative_stage"] = (
            (df["stage"] - 1)
            / (df["T_final"] - 1)
        )

    # ------------------------------------------------------------------
    # Final-stage comparison table
    # ------------------------------------------------------------------

    final_stage_comparison = (
        comparison_summary
        .loc[
            comparison_summary["stage"] == comparison_summary["T_final"],
            [
                "T_final",
                "c0_death",
                "epsilon_selection",
                "stage",
                "mean_n_admissible_mean",
                "mean_n_admissible_sd",
                "pct_1_treatment_mean",
                "pct_all_treatments_mean",
            ],
        ]
        .sort_values("T_final")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    comparison_summary.to_csv(
        os.path.join(config.results_dir, "comparison_summary_by_T.csv"),
        index=False,
    )

    comparison_qscale.to_csv(
        os.path.join(config.results_dir, "comparison_qscale_by_T.csv"),
        index=False,
    )

    comparison_pct.to_csv(
        os.path.join(config.results_dir, "comparison_pct_by_T.csv"),
        index=False,
    )

    final_stage_comparison.to_csv(
        os.path.join(config.results_dir, "comparison_final_stage_by_T.csv"),
        index=False,
    )

    if verbose:
        print("\n" + "=" * 80)
        print("EXPERIMENT PLAN COMPLETED")
        print("=" * 80)
        print("Saved files:")
        print("  - comparison_summary_by_T.csv")
        print("  - comparison_qscale_by_T.csv")
        print("  - comparison_pct_by_T.csv")
        print("  - comparison_final_stage_by_T.csv")
        print("=" * 80)

    return {
        "horizon_outputs": horizon_outputs,
        "comparison_summary": comparison_summary,
        "comparison_qscale": comparison_qscale,
        "comparison_pct": comparison_pct,
        "final_stage_comparison": final_stage_comparison,
    }


# ============================================================
# Loading utilities
# ============================================================

def load_experiment_results(results_dir: str = "results") -> Dict[str, pd.DataFrame]:
    """Load cross-horizon experiment outputs from disk."""

    return {
        "comparison_summary": pd.read_csv(os.path.join(results_dir, "comparison_summary_by_T.csv")),
        "comparison_qscale": pd.read_csv(os.path.join(results_dir, "comparison_qscale_by_T.csv")),
        "comparison_pct": pd.read_csv(os.path.join(results_dir, "comparison_pct_by_T.csv")),
        "final_stage_comparison": pd.read_csv(os.path.join(results_dir, "comparison_final_stage_by_T.csv")),
    }


# ============================================================
# Plot styling and publication figures
# ============================================================

def set_publication_style() -> None:
    """Set a clean Matplotlib style for paper figures."""

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_figure(fig, output_path: Optional[str]) -> None:
    """Save a figure if an output path is provided."""

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")


def plot_main_results_side_by_side(comparison_summary: pd.DataFrame,
                                   comparison_qscale: pd.DataFrame,
                                   use_relative_stage: bool = False,
                                   output_path: Optional[str] = None):

    """
    Plot the two main paper figures side by side:
    - Mean admissible treatments
    - Mean top-2 Q-value gap
    """

    set_publication_style()

    x_col = "relative_stage" if use_relative_stage else "stage"
    x_label = (
        "Relative epsilon-selection stage"
        if use_relative_stage
        else "Epsilon-selection stage"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.2)
    )

    ax1, ax2 = axes

    # ========================================================
    # LEFT PANEL
    # Mean admissible treatments
    # ========================================================

    for t_final in sorted(comparison_summary["T_final"].unique()):

        tmp = (
            comparison_summary[
                comparison_summary["T_final"] == t_final
            ]
            .sort_values("stage")
        )

        ax1.plot(
            tmp[x_col].to_numpy(),
            tmp["mean_n_admissible_mean"].to_numpy(),
            marker="o",
            linewidth=2.3,
            markersize=5,
            label=f"T = {t_final}",
        )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Mean number of admissible treatments")

    if use_relative_stage:

        ax1.set_xlim(-0.02, 1.02)

    else:

        max_stage = int(comparison_summary["stage"].max())

        ax1.set_xticks(np.arange(1, max_stage + 1))

    ax1.set_ylim(
        0,
        len(np.round(np.arange(0.1, 1.1, 0.1), 1)) + 0.5
    )

    ax1.grid(alpha=0.2)

    # ========================================================
    # RIGHT PANEL
    # Mean top-2 gap
    # ========================================================

    for t_final in sorted(comparison_qscale["T_final"].unique()):

        tmp = (
            comparison_qscale[
                comparison_qscale["T_final"] == t_final
            ]
            .sort_values("stage")
        )

        ax2.plot(
            tmp[x_col].to_numpy(),
            tmp["mean_top2_gap_mean"].to_numpy(),
            marker="o",
            linewidth=2.3,
            markersize=5,
            label=f"T = {t_final}",
        )

    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Mean top-2 Q-value gap")
    ax2.set_xlabel("Stage")

    if use_relative_stage:

        ax2.set_xlim(-0.02, 1.02)

    else:

        max_stage = int(comparison_qscale["stage"].max())

        ax2.set_xticks(np.arange(1, max_stage + 1))

    ax2.grid(alpha=0.2)

    # ========================================================
    # Shared legend
    # ========================================================

    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    fig.tight_layout()

    _save_figure(fig, output_path)

    return fig, axes


def plot_horizon_effect_summary(
    epsilon_stage_comparison: pd.DataFrame,
    output_path: Optional[str] = None,
):
    """
    Summary figure showing:
    1. Stability of stage-1 admissible sets.
    2. Growth of final-stage admissible sets.
    3. Decrease of singleton recommendations at the final stage.
    """

    set_publication_style()

    df = epsilon_stage_comparison.copy()

    stage1 = (
        df[df["stage"] == 1]
        .sort_values("T_final")
        .copy()
    )

    stage_final = (
        df[df["stage"] == df["T_final"]]
        .sort_values("T_final")
        .copy()
    )

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_stage1 = "#1f77b4"
    color_final = "#d62728"
    color_singleton = "#2ca02c"

    ax1.plot(
        stage1["T_final"],
        stage1["mean_n_admissible_mean"],
        marker="o",
        linewidth=2.6,
        markersize=6,
        color=color_stage1,
        label="Stage 1",
    )

    ax1.plot(
        stage_final["T_final"],
        stage_final["mean_n_admissible_mean"],
        marker="s",
        linewidth=2.6,
        markersize=6,
        color=color_final,
        label="Final stage",
    )

    ax1.set_xlabel(r"Final horizon $T$")
    ax1.set_ylabel("Mean number of admissible treatments")
    ax1.set_xticks(sorted(df["T_final"].unique()))
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()

    ax2.plot(
        stage_final["T_final"],
        stage_final["pct_1_treatment_mean"],
        marker="^",
        linestyle="--",
        linewidth=2.4,
        markersize=6,
        color=color_singleton,
        label="% singleton (final stage)",
    )

    ax2.set_ylabel("Singleton recommendations (%)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    fig.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    _save_figure(fig, output_path)

    return fig, (ax1, ax2)