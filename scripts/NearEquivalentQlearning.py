import numpy as np
import pandas as pd 
from sklearn.svm import SVR
from collections import Counter
import random
random.seed(42)

# Global rounding function
# This function rounds all values to 4 decimal places to ensure consistency across all calculations.
def r(x):
    try:
        return np.round(x, 4)
    except Exception:
        try:
            return round(float(x), 4)
        except Exception:
            return x

# --------------------------------------------------
# Trains an SVR model for the specified stage and computes prediction matrix
# for each possible treatment.
# --------------------------------------------------
def train_last_stage(final_stage, data_stages, Rewards_Zhao, possibles_treatments, c=1.0, epsilon=0.1):
    """
    Trains an SVR model for the specified stage and computes predictions
    for each possible treatment.

    Parameters:
        data_stages : dict containing DataFrames for each stage.
        Rewards_Zhao : dict containing reward vectors per stage.
        possibles_treatments : list of treatment values.
        final_stage : int indicating the final stage.
        c : SVR parameter C (default: 1.0).
        epsilon : SVR epsilon parameter (default: 0.1).

    Returns:
        svr_model : trained SVR model for the final stage.
        predicted_matrix : prediction matrix of shape (num_treatments, num_samples).
    """
    data_key = f"Data_Stage_{final_stage}"
    reward_key = f"Reward_{final_stage}"
    dosage_column = f"Dosage_{final_stage}"

    svr_model = SVR(kernel='rbf', C=c, epsilon=epsilon)
    svr_model.fit(data_stages[data_key], Rewards_Zhao[reward_key])

    n_samples = data_stages[data_key].shape[0]
    predicted_matrix = np.zeros((len(possibles_treatments), n_samples))

    for idx, treatment in enumerate(possibles_treatments):
        Data_Stage_update = data_stages[data_key].copy()
        Data_Stage_update[dosage_column] = treatment
        predicted_values = r(svr_model.predict(Data_Stage_update))
        predicted_matrix[idx, :] = predicted_values

    return svr_model, r(predicted_matrix)

# --------------------------------------------------
# Selects, for each patient (column), the values within epsilon margin
# of the column max, sorted in descending order.
# --------------------------------------------------
def select_values_within_epsilon(predicted_matrix, epsilon):
    num_treatments, num_patients = predicted_matrix.shape
    selected_values_per_column = []

    for col in range(num_patients):
        values = predicted_matrix[:, col].astype(float)
        qmax = np.max(values)
        threshold = qmax - epsilon * abs(qmax)

        selected = values[values >= threshold]
        if selected.size == 0:              # sécurité (NaN/edge cases)
            selected = np.array([qmax])

        selected_values_per_column.append(np.sort(selected)[::-1].tolist())

    return np.array(selected_values_per_column, dtype=object)

# --------------------------------------------------
# Constructs pseudo-outcomes:
# - completes each row if needed with the most frequent value
# - adds corresponding rewards to each row
# --------------------------------------------------
def pseudo_outcome_construct(selected_values_array, nb_treatments, rewards):
    """
    Constructs a full matrix of q-values per patient by filling missing values
    and adding reward values.
    """
    completed_lists = []

    for sublist in selected_values_array:
        if len(sublist) >= nb_treatments:
            completed_lists.append(sublist[:nb_treatments])
            continue
        counter = Counter(sublist)
        most_common_value = max(counter.items(), key=lambda x: (x[1], x[0]))[0]
        completed_list = sublist + [most_common_value] * (nb_treatments - len(sublist))
        completed_lists.append(completed_list)

    matrix = np.array(completed_lists)

    for i in range(len(matrix)):
        matrix[i] = r(matrix[i].astype(float) + rewards[i])

    return r(matrix)

# --------------------------------------------------
# Computes the Y-tilde matrix per model, using data and reward for a given stage
# --------------------------------------------------
def compute_ytilde_matrix(models, stage_number, possibles_treatment, data_stages, rewards_zhao):
    """
    Computes Y-tilde matrix = reward + max predicted q-value per sample.

    Returns:
        np.ndarray of shape (num_samples, num_models)
    """
    data_key = f"Data_Stage_{stage_number}"
    reward_key = f"Reward_{stage_number - 1}"

    n_samples = data_stages[data_key].shape[0]
    n_models = len(models)
    Ytilde_matrix = np.zeros((n_samples, n_models))

    dosage_column = f"Dosage_{stage_number}"

    for i, model in enumerate(models):
        predicted_matrix = np.zeros((len(possibles_treatment), n_samples))

        for idx, treatment in enumerate(possibles_treatment):
            Data_Stage_update = data_stages[data_key].copy()
            Data_Stage_update[dosage_column] = treatment
            predicted_matrix[idx, :] = r(model.predict(Data_Stage_update))

        max_values = np.max(predicted_matrix, axis=0)
        Ytilde_matrix[:, i] = r(rewards_zhao[reward_key] + max_values)

    return r(Ytilde_matrix)

# --------------------------------------------------
# Trains a list of SVR models from each column in a pseudo-outcome matrix
# --------------------------------------------------
def train_svr_models(pseudo_outcome_matrix, data_stage, C=1.0, epsilon=0.1):
    """
    Trains one SVR model per column in the pseudo-outcome matrix.
    """
    models = []
    for i in range(pseudo_outcome_matrix.shape[1]):
        svr_model = SVR(kernel='rbf', C=C, epsilon=epsilon)
        svr_model.fit(data_stage, pseudo_outcome_matrix[:, i])
        models.append(svr_model)
    return models

# --------------------------------------------------
# Computes models for all stages from last stage to stage 0
# --------------------------------------------------
def compute_all_stages_models(last_stage_models, penultimate_stage_models, number_of_decision_rules, possibles_treatments, data_stages, rewards_zhao):
    """
    Iteratively computes SVR models from last stage down to stage 0.

    Returns:
        dict mapping stage -> list of models.
    """
    models_by_stage = {}
    last_stage = number_of_decision_rules
    penultimate_stage = number_of_decision_rules - 1

    models_by_stage[last_stage] = last_stage_models
    models_by_stage[penultimate_stage] = penultimate_stage_models

    for s in range(penultimate_stage, 0, -1):
        pseudo_outcome_matrix = compute_ytilde_matrix(models_by_stage[s], s, possibles_treatments, data_stages, rewards_zhao)
        models_by_stage[s - 1] = train_svr_models(pseudo_outcome_matrix, data_stages[f"Data_Stage_{s-1}"])

    return models_by_stage

# --------------------------------------------------
# Full pipeline for training models from last stage to stage 0 using Near-Equivalent Q-learning
# --------------------------------------------------
def NearEquivalentQlearning(number_of_decision_rules, data_stages, rewards_zhao, possibles_treatments, c=1.0, epsilon_model=0.1, epsilon_selection=0.1):
    """
    Full pipeline from final stage to stage 0 using near-equivalent q-value selection.

    Returns:
        models_by_stage : dict from stage index to list of trained models.
    """
    last_stage_model, predicted_matrix = train_last_stage(number_of_decision_rules, data_stages, rewards_zhao, possibles_treatments)
    selected_values_array = select_values_within_epsilon(predicted_matrix, epsilon_selection)

    penultimate_stage = number_of_decision_rules - 1
    stage_penultimate_pseudo_outcome_matrix = pseudo_outcome_construct(selected_values_array, len(possibles_treatments), rewards_zhao[f"Reward_{penultimate_stage}"])

    penultimate_stage_models = train_svr_models(stage_penultimate_pseudo_outcome_matrix, data_stages[f"Data_Stage_{penultimate_stage}"])

    models_by_stage = compute_all_stages_models(last_stage_model, penultimate_stage_models, number_of_decision_rules, possibles_treatments, data_stages, rewards_zhao)

    return models_by_stage

# --------------------------------------------------
# Classical Q-learning pipeline (no near-equivalent selection)
# --------------------------------------------------
def classicalQlearning(number_of_decision_rules, data_stages, rewards_zhao, possibles_treatments, c=1.0, epsilon_model=0.1):
    """
    Implements classical Q-learning: predicts best future value at each step.
    """
    svr_models = []

    for stage in range(number_of_decision_rules, -1, -1):
        svr_model = SVR(kernel='rbf', C=c, epsilon=epsilon_model)

        if stage == number_of_decision_rules:
            Ytilde = rewards_zhao[f"Reward_{stage}"]
        else:
            Ytilde = r(rewards_zhao[f"Reward_{stage}"] + max_values)

        svr_model.fit(data_stages[f'Data_Stage_{stage}'], Ytilde)

        predicted_matrix = np.zeros((len(possibles_treatments), data_stages[f'Data_Stage_{stage}'].shape[0]))

        for idx, treatment in enumerate(possibles_treatments):
            Data_Stage_update = data_stages[f'Data_Stage_{stage}'].copy()
            Data_Stage_update[f'Dosage_{stage}'] = treatment
            predicted_values = r(svr_model.predict(Data_Stage_update))
            predicted_matrix[idx, :] = predicted_values

        max_values = np.max(predicted_matrix, axis=0)
        max_values = r(max_values)
        svr_models.append(svr_model)

    svr_models.reverse()
    return tuple(svr_models)
