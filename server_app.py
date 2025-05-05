import flwr as fl
from task import NUM_CLIENTS, NUM_ROUNDS



def metrics_aggregate(results):
    if not results:
        return {}
    else:
        total_samples = 0  # Number of samples in the dataset

        # Collecting metrics
        aggregated_metrics = {
            "accuracy": 0,
            'precision':0, 
            'recall':0,
            'f1':0
        }

        # Extracting values from the results
        for samples, metrics in results:
            for key, value in metrics.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = 0
                else:
                    aggregated_metrics[key] += (value * samples)
            total_samples += samples

        # Compute the weighted average for each metric
        for key in aggregated_metrics.keys():
            aggregated_metrics[key] = round(aggregated_metrics[key] / total_samples, 6)

        return aggregated_metrics

if __name__ == "__main__":

    print('Server')

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_fit_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=metrics_aggregate,
    )

    fl.server.start_server(
        config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        server_address='0.0.0.0:8080'
    )