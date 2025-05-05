

from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from typing import List

COUNTRIES =['US','UNK','CAN','MEX','PR']
NUM_ROUNDS = 3
NUM_CLIENTS = len(COUNTRIES)
CLIENT = 380680241


def set_params(model: IsolationForest, params: List[np.ndarray]) -> IsolationForest:

    model.max_features = int(params[0])
    model.n_estimators = max(1,int(params[1]))
    model.random_state = int(params[2])
    model.n_jobs = max(1,int(params[3]))
    model.verbose = int(params[4])

    return model

def get_params(model :IsolationForest) -> List[np.ndarray]:

    params = [
        model.max_features,
        model.n_estimators,
        model.random_state,
        model.n_jobs,
        model.verbose
    ]

    return params


def create_model():

    return IsolationForest(random_state=42, n_jobs=1)

def load_data(country_id:str):

    data = pd.read_pickle('data.pkl')

    country_data = data.loc[data.acqCountry==country_id].drop('acqCountry',axis=1).copy()

    X = country_data.drop('isFraud',axis = 1)
    y = country_data['isFraud']

    return X, y

def load_data_client(country_id:str, client:int):

    data = pd.read_pickle('data.pkl')

    country_data = data.loc[(data.acqCountry==country_id) & (data.accountNumber==client)].drop('acqCountry',axis=1).copy()

    X = country_data.drop('isFraud',axis = 1)
    y = country_data['isFraud']

    return X, y
