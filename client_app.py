from flwr.client import NumPyClient
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
import flwr as fl

from task import get_params, set_params,create_model, COUNTRIES, load_data, load_data_client, CLIENT


import sys

class FlowerClient(NumPyClient):

    def __init__(self, model, X,y, country):
        self.model = model
        self.X = X
        self.y = y
        self.country = country

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        self.model.fit(self.X)

        return get_params(self.model), len(self.X), {}
        

    def evaluate(self, parameters, config):

        set_params(self.model, parameters)
        preds = self.model.predict(self.X)
        preds_bool = preds==-1
        y_bool = self.y.values == 1

        loss  =log_loss(y_bool,preds_bool, labels=[False,True])

        acc = accuracy_score(y_bool,preds_bool)
        precision = precision_score(y_bool,preds_bool, average='weighted')
        recall = recall_score(y_bool,preds_bool, average='weighted')
        f1 = f1_score(y_bool,preds_bool, average='weighted')

        print(self.country)
        print(f'loss: {loss} acc: {acc}')
        print('*'* 20)

        return loss, len(self.X), {'accuracy':acc, 'precision':precision, 'recall':recall, 'f1':f1 }

    def get_parameters(self, config):
        return get_params(self.model)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError('Country ID required')
        sys.exit(1)
    else:
        country_id = sys.argv[1]

    if country_id not in COUNTRIES:
        raise ValueError(f'{country_id} not a recognised country choices are {COUNTRIES}')

    print(f'Client country: {country_id}')

    # if running for full population uncomment this line and comment the load_data_client function
    # X,y = load_data(country_id)

    X,y = load_data_client(country_id,CLIENT)
        

    print(f'Data for {country_id} - X: {X.shape[0]}, y: {y.shape[0]}')

    model = create_model()

    model.fit(X,y)

    fl.client.start_client(server_address='0.0.0.0:8080', client=FlowerClient(model, X,y, country_id).to_client())
