import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import random
random.seed(42)


# ==========================================================
# ODE SYSTEM (baseline fixed per patient)
# ==========================================================
def odes(x, t, Y0, X0, D, a1=0.1, a2=0.15, b1=1.2, b2=1.2, d1=0.5, d2=0.5):
    # State variables
    Y = x[0]  # tumor size
    X = x[1]  # toxicity

    # Dynamics (with absorbing boundary at Y=0 via indicator)
    dYdt = (a1 * max(X, X0) - b2 * (D - d1)) * (Y > 0)
    dXdt = a2 * max(Y, Y0) + b2 * (D - d2)

    return [dYdt, dXdt]


# ==========================================================
# FIXED-STEP INTEGRATOR (RK4) TO AVOID LSODA/odeint WARNINGS
# ==========================================================
def rk4_step(Y, X, Y0, X0, D, dt=1.0, n_sub=50,
             a1=0.1, a2=0.15, b1=1.2, b2=1.2, d1=0.5, d2=0.5):
    """Integrate the ODE over one stage of length dt using fixed-step RK4.

    The system includes discontinuities (max(·) and the indicator (Y > 0)).
    The tumor component is constrained to remain nonnegative via clamping.
    """
    h = dt / float(n_sub)
    y = np.array([float(Y), float(X)], dtype=float)

    def f(state):
        Yc, Xc = state
        dYdt = (a1 * max(Xc, X0) - b2 * (D - d1)) * (1.0 if Yc > 0 else 0.0)
        dXdt = a2 * max(Yc, Y0) + b2 * (D - d2)
        return np.array([dYdt, dXdt], dtype=float)

    for _ in range(int(n_sub)):
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Enforce Y >= 0 (absorbing boundary at 0)
        if y[0] < 0:
            y[0] = 0.0

        # Numerical safety check
        if not np.isfinite(y).all():
            return 0.0, float("nan")

    return float(y[0]), float(y[1])


def constant_dose_regimes(Dose, Y0, X0, T_final):
    # Stage grid (kept for consistency with previous versions)
    t = np.arange(0, 2)

    # Initial state (time 0)
    x0 = [Y0, X0]
    Y = [Y0]
    X = [X0]

    # Simulate T_final stages under a fixed dose
    for i in range(T_final):
        # One-stage transition using RK4
        Y_next, X_next = rk4_step(x0[0], x0[1], Y0, X0, Dose)
        Y.extend([Y_next])
        X.extend([X_next])
        x0 = [Y_next, X_next]

    Y = np.array(Y)
    X = np.array(X)

    return Y, X


# ==========================================================
# SINGLE-PATIENT TRAJECTORY UNDER Q-LEARNING POLICY
# (stage-wise standardisation)
# ==========================================================
def generate_patient_trajectory(Q0, Q1, Q2, Q3, Q4, Q5, Y0, X0, possibles_treatments):
    # Stage-specific Q-models (backward Q-learning)
    Qs = [Q0, Q1, Q2, Q3, Q4, Q5]

    # Stage -> scaler dictionary (injected externally, if used)
    stage_scalers = globals().get("STAGE_SCALERS", {})

    def predict_stage(Q, stage, df):
        # Align feature columns to training order if available
        if hasattr(Q, "feature_names_in_"):
            df = df.reindex(columns=list(Q.feature_names_in_))

        # Apply stage-wise scaling if provided
        sc = stage_scalers.get(stage, None)
        if sc is not None:
            df = pd.DataFrame(sc.transform(df), columns=df.columns)

        return float(Q.predict(df)[0])

    # ---------------- Stage 0 ----------------
    # Choose D0 = argmax_a Q0(H0, a)
    val = []
    for a in possibles_treatments:
        df = pd.DataFrame({'Tumor_0': [Y0], 'Toxicity_0': [X0], 'Dosage_0': [a]})
        val.append(predict_stage(Q0, 0, df))
    D0 = possibles_treatments[int(np.argmax(val))]

    # Simulate transition to stage 1 under D0
    t = np.arange(0, 2)
    x0 = [Y0, X0]
    Y1, X1 = rk4_step(x0[0], x0[1], Y0, X0, D0)

    # ---------------- Stage 1 ----------------
    # Choose D1 = argmax_a Q1(H1, a)
    val = []
    for a in possibles_treatments:
        df = pd.DataFrame({'Tumor_0': [Y0], 'Toxicity_0': [X0], 'Dosage_0': [D0],
                           'Tumor_1': [Y1], 'Toxicity_1': [X1], 'Dosage_1': [a]})
        val.append(predict_stage(Q1, 1, df))
    D1 = possibles_treatments[int(np.argmax(val))]

    # Simulate transition to stage 2 under D1
    x0 = [Y1, X1]
    Y2, X2 = rk4_step(x0[0], x0[1], Y1, X1, D1)

    # ---------------- Stage 2 ----------------
    # Choose D2 = argmax_a Q2(H2, a)
    val = []
    for a in possibles_treatments:
        df = pd.DataFrame({'Tumor_0': [Y0], 'Toxicity_0': [X0], 'Dosage_0': [D0],
                           'Tumor_1': [Y1], 'Toxicity_1': [X1], 'Dosage_1': [D1],
                           'Tumor_2': [Y2], 'Toxicity_2': [X2], 'Dosage_2': [a]})
        val.append(predict_stage(Q2, 2, df))
    D2 = possibles_treatments[int(np.argmax(val))]

    # Simulate transition to stage 3 under D2
    x0 = [Y2, X2]
    Y3, X3 = rk4_step(x0[0], x0[1], Y2, X2, D2)

    # ---------------- Stage 3 ----------------
    # Choose D3 = argmax_a Q3(H3, a)
    val = []
    for a in possibles_treatments:
        df = pd.DataFrame({'Tumor_0': [Y0], 'Toxicity_0': [X0], 'Dosage_0': [D0],
                           'Tumor_1': [Y1], 'Toxicity_1': [X1], 'Dosage_1': [D1],
                           'Tumor_2': [Y2], 'Toxicity_2': [X2], 'Dosage_2': [D2],
                           'Tumor_3': [Y3], 'Toxicity_3': [X3], 'Dosage_3': [a]})
        val.append(predict_stage(Q3, 3, df))
    D3 = possibles_treatments[int(np.argmax(val))]

    # Simulate transition to stage 4 under D3
    x0 = [Y3, X3]
    Y4, X4 = rk4_step(x0[0], x0[1], Y3, X3, D3)

    # ---------------- Stage 4 ----------------
    # Choose D4 = argmax_a Q4(H4, a)
    val = []
    for a in possibles_treatments:
        df = pd.DataFrame({'Tumor_0': [Y0], 'Toxicity_0': [X0], 'Dosage_0': [D0],
                           'Tumor_1': [Y1], 'Toxicity_1': [X1], 'Dosage_1': [D1],
                           'Tumor_2': [Y2], 'Toxicity_2': [X2], 'Dosage_2': [D2],
                           'Tumor_3': [Y3], 'Toxicity_3': [X3], 'Dosage_3': [D3],
                           'Tumor_4': [Y4], 'Toxicity_4': [X4], 'Dosage_4': [a]})
        val.append(predict_stage(Q4, 4, df))
    D4 = possibles_treatments[int(np.argmax(val))]

    # Simulate transition to stage 5 under D4
    x0 = [Y4, X4]
    Y5, X5 = rk4_step(x0[0], x0[1], Y4, X4, D4)

    # ---------------- Stage 5 ----------------
    # Choose D5 = argmax_a Q5(H5, a)
    val = []
    for a in possibles_treatments:
        df = pd.DataFrame({'Tumor_0': [Y0], 'Toxicity_0': [X0], 'Dosage_0': [D0],
                           'Tumor_1': [Y1], 'Toxicity_1': [X1], 'Dosage_1': [D1],
                           'Tumor_2': [Y2], 'Toxicity_2': [X2], 'Dosage_2': [D2],
                           'Tumor_3': [Y3], 'Toxicity_3': [X3], 'Dosage_3': [D3],
                           'Tumor_4': [Y4], 'Toxicity_4': [X4], 'Dosage_4': [D4],
                           'Tumor_5': [Y5], 'Toxicity_5': [X5], 'Dosage_5': [a]})
        val.append(predict_stage(Q5, 5, df))
    D5 = possibles_treatments[int(np.argmax(val))]

    # Simulate transition to stage 6 under D5
    x0 = [Y5, X5]
    Y6, X6 = rk4_step(x0[0], x0[1], Y5, X5, D5)

    return [Y0, Y1, Y2, Y3, Y4, Y5, Y6], [X0, X1, X2, X3, X4, X5, X6]


def generate_dataset(T_final=6, N=1000,
                     c0=-4.0, c1=1.0, c2=1.0,
                     a1=0.15, a2=0.1, b1=1.2, b2=1.2, d1=0.5, d2=0.5):
    """
    Simulate N patient trajectories over T_final stages.

    Outputs:
        - Data: longitudinal tumor/toxicity/dose history
        - Remission_Informations: (Patient, Stage) at first remission
        - Dead_Informations: (Patient, Stage) at death time
    """

    # Column names: (Tumor_t, Toxicity_t, Dosage_t) for t = 0..T_final
    column_names = []
    for i in range(T_final + 1):
        column_names.append(f"Tumor_{i}")
        column_names.append(f"Toxicity_{i}")
        column_names.append(f"Dosage_{i}")

    Data = pd.DataFrame(columns=column_names)

    # Death information (patient index, stage)
    list_patient_dead = []
    list_stop_stage = []

    # Remission information (patient index, stage)
    list_patient_remission = []
    list_remission_stage = []

    for k in range(N):
        # Baseline initial conditions
        t = np.arange(0, 2)  # time grid (kept for consistency)
        Y0 = np.random.uniform(0, 2)
        X0 = np.random.uniform(0, 2)
        x0 = [Y0, X0]

        Tumor = [Y0]
        Toxicity = [X0]
        D = [round(np.random.uniform(0.5, 1), 1)]

        # Last observed values prior to death (used for padding)
        last_valid_tumor = Y0
        last_valid_toxicity = X0

        patient_dead = False
        patient_in_remission = False

        for i in range(T_final):

            # After death: freeze state and set dose to 0
            if patient_dead:
                Tumor.append(last_valid_tumor)
                Toxicity.append(last_valid_toxicity)
                D.append(0)
                continue

            # After remission: set state and dose to 0
            if patient_in_remission:
                Tumor.append(0)
                Toxicity.append(0)
                D.append(0)
                continue

            # Remission check based on current tumor value
            if Tumor[i] == 0 or np.isnan(Tumor[i]):
                Tumor.append(0)
                Toxicity.append(0)
                D.append(0)
                patient_in_remission = True
                list_patient_remission.append(k)
                list_remission_stage.append(i + 1)
                continue

            # Death process at stage i
            hazard_rate = np.exp(c0 + c1 * Tumor[i] + c2 * Toxicity[i])
            P_death = 1 - np.exp(-hazard_rate)

            if bernoulli.rvs(P_death) == 1:
                last_valid_tumor = Tumor[i]
                last_valid_toxicity = Toxicity[i]
                Tumor.append(last_valid_tumor)
                Toxicity.append(last_valid_toxicity)
                D.append(0)
                patient_dead = True
                list_patient_dead.append(k)
                list_stop_stage.append(i + 1)
                break
            else:
                # One-stage forward simulation under current dose D[i]
                Y_next, X_next = rk4_step(
                    x0[0], x0[1], Y0, X0, D[i],
                    a1=a1, a2=a2, b1=b1, b2=b2, d1=d1, d2=d2
                )

                # Remission check based on next tumor value
                if Y_next < 0 or Y_next == 0:
                    Tumor.append(0)
                    Toxicity.append(X_next)
                    D.append(0)
                    patient_in_remission = True
                    list_patient_remission.append(k)
                    list_remission_stage.append(i + 1)
                    break
                else:
                    Tumor.append(Y_next)
                    Toxicity.append(X_next)

                    x0 = [Y_next, X_next]
                    last_valid_tumor = Y_next
                    last_valid_toxicity = X_next
                    D.append(round(np.random.uniform(0.1, 1), 1))

        # Assemble patient row (pad if early stop)
        vec = []
        for j in range(T_final + 1):
            vec.append(Tumor[j] if len(Tumor) > j else last_valid_tumor)
            vec.append(Toxicity[j] if len(Toxicity) > j else last_valid_toxicity)
            vec.append(D[j] if len(D) > j else 0)

        Data.loc[k] = vec

    # Drop the final dosage column (trial ends at T_final)
    if f"Dosage_{T_final}" in Data.columns:
        del Data[f"Dosage_{T_final}"]

    Dead_Informations = pd.DataFrame({
        "Patient": list_patient_dead,
        "Stage": list_stop_stage
    })

    Remission_Informations = pd.DataFrame({
        "Patient": list_patient_remission,
        "Stage": list_remission_stage
    })

    return Data, Remission_Informations, Dead_Informations


# ==========================================================
# REWARD CONSTRUCTION (same logic as your current script)
# ==========================================================
def calculate_rewards(Data, Remission_Informations, Dead_Informations, T_final):
    # Output columns: patient id + stage-wise rewards
    rewards_columns = ['Patient'] + ['Reward_%s' % t for t in range(T_final)]
    Rewards = pd.DataFrame(columns=rewards_columns)
    N = Data.shape[0]

    # Fast lookup: patient -> stage
    remission_dict = dict(zip(Remission_Informations['Patient'], Remission_Informations['Stage']))
    death_dict = dict(zip(Dead_Informations['Patient'], Dead_Informations['Stage']))

    for i in range(N):
        rewards = [i]

        stop_dead = death_dict.get(i, None)
        stop_rem = remission_dict.get(i, None)

        for t in range(T_final):
            tumor = Data.loc[i, f'Tumor_{t}']
            tox = Data.loc[i, f'Toxicity_{t}']

            # Death: reward is 0 from death stage onward
            if stop_dead is not None and t >= (stop_dead - 1):
                rewards.append(0.0)
                continue

            # Remission: reward is 1 from remission stage onward
            if stop_rem is not None and t >= (stop_rem - 1):
                rewards.append(1.0)
                continue

            # Otherwise: stage-wise utility
            rewards.append(float(np.exp(-tumor - tox)))

        Rewards.loc[i] = rewards

    return Rewards


# ==========================================================
# CREATE STAGE-WISE DATASETS (for backward Q-learning)
# ==========================================================
def create_data_stage_frames(Data, T_final):
    # For each stage i, keep the history up to i:
    # (Tumor_0..Tumor_i, Toxicity_0..Toxicity_i, Dosage_0..Dosage_i)
    data_stages = {}
    for i in range(T_final + 1):
        cols = []
        for t in range(i + 1):
            cols.extend([f"Tumor_{t}", f"Toxicity_{t}", f"Dosage_{t}"])
        existing = [c for c in cols if c in Data.columns]
        data_stages[f"Data_Stage_{i}"] = Data.loc[:, existing]
    return data_stages