import pandas as pd 
import numpy as np 

def data_generation_train(n, resp_type, p=10): 
    df = pd.DataFrame(columns=['X_' + str(i) for i in range(0, p)] + ['A', 'Resp'])

    for i in range(n):
        # Covariables
        X = np.random.uniform(-1, 1, p)
        
        # Traitement
        A = np.random.choice([-1, 1])
        
        # Response 
        if resp_type == "Scenario1":
            response = np.random.normal(1 + 2 * X[0] + X[1] + 0.5 * X[2] + (A * (X[0] + X[1])))
        elif resp_type == "Scenario2": 
            response = np.random.normal(1 + 2 * X[0] + X[1] + 0.5 * X[2] + (A * 0.442 *(1 - X[0] - X[1])))
        else:
            raise ValueError("Unknown response type")
        
        # Patient
        patient = np.append(X, [A, response])
        
        # Add patient
        df.loc[len(df)] = patient
    
    # Rescale response to be strictly positive
    df['Resp'] = df['Resp'] + (np.abs(df['Resp'].min()) + 1)
    return df


def data_generation_test(n, resp_type, p=10): 
    df = pd.DataFrame(columns=['X_' + str(i) for i in range(0, p)])

    for i in range(n):
        # Génération des covariables
        patient = np.random.uniform(-1, 1, p)
        
        # Ajout de la ligne au DataFrame
        df.loc[len(df)] = patient
    
    df_test = df[df.columns[df.columns.str.startswith('X_')]]
    df_test.insert(0, 'Intercept', 1)
    
    if resp_type == "Scenario1":
        y_true = np.sign(df_test['X_0'] + df_test['X_1'])
       
    elif resp_type == "Scenario2": 
         y_true = np.sign(1 - df_test['X_0'] - df_test['X_1'] )
    else:
        raise ValueError("Unknown response type")
    return df_test, y_true