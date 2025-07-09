#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def FedWeightAvg(central_model, client_models):
    num_clients = len(client_models)

    weights = [1/num_clients for i in range(num_clients)]
    print(weights, num_clients)
    all_clients_params = [client_models[idx].state_dict() for idx in range(0, num_clients)]
   
    w_glob = central_model.state_dict()
    
    keys = w_glob.keys()
    noise_level = 0.
    for key in keys:
        temp = torch.zeros_like(w_glob[key])
        for idx in range(0, num_clients):
            if noise_level > 0 and 'bias' not in key:
                noise = noise_level * torch.empty(all_clients_params[idx][key].size()).normal_(mean=0,std=all_clients_params[idx][key].reshape(-1).float().std())
                temp = temp + weights[idx] * all_clients_params[idx][key] + noise.to(device)
            else:
                temp = temp + weights[idx] * all_clients_params[idx][key]

        w_glob[key] = temp
    return w_glob 


def FedProx(central_model, client_models, mu=0.1):
    num_clients = len(client_models)

    # Uniform weights for clients
    weights = [1 / num_clients for i in range(num_clients)]
    print(weights, num_clients)

    # Collect state_dict (model parameters) from all clients
    all_clients_params = [client_models[idx].state_dict() for idx in range(num_clients)]

    # Get the global model's state_dict
    w_glob = central_model.state_dict()

    # Iterate over each parameter in the model
    keys = w_glob.keys()
    noise_level = 0.  # If needed, you can add noise as in FedAvg

    for key in keys:
        temp = torch.zeros_like(w_glob[key])

        for idx in range(num_clients):
            # If noise is required (optional)
            if noise_level > 0 and 'bias' not in key:
                noise = noise_level * torch.empty(all_clients_params[idx][key].size()).normal_(
                    mean=0, std=all_clients_params[idx][key].reshape(-1).float().std())
                temp = temp + weights[idx] * (all_clients_params[idx][key] + noise.to(device))
            else:
                # Apply the FedProx proximal term here
                # Proximal term penalizes the deviation from the global model
                proximal_term = mu * (all_clients_params[idx][key] - w_glob[key])
                temp = temp + weights[idx] * (all_clients_params[idx][key] - proximal_term)

      
        w_glob[key] = temp

    return w_glob




def FedWeightAvgBatchNorm(central_model, client_models):
    num_clients = len(client_models)

    weights = [1/num_clients for i in range(num_clients)]
    print(weights, num_clients)
    all_clients_params = [client_models[idx].state_dict() for idx in range(0, num_clients)]
   
    w_glob = central_model.state_dict()
    
    keys = w_glob.keys()

    noise_level = 0.
    for key in keys:
        temp = torch.zeros_like(w_glob[key])
        if "running_mean" in key.lower() or "running_var" in key.lower() or "num_batches_tracked" in key.lower():
            # Just take the parameters from the first client, assuming they are identical
            temp = all_clients_params[0][key]
        else:
            for idx in range(0, num_clients):
                if noise_level > 0 and 'bias' not in key:
                    noise = noise_level * torch.empty(all_clients_params[idx][key].size()).normal_(mean=0, std=all_clients_params[idx][key].reshape(-1).float().std())
                    temp = temp + weights[idx] * all_clients_params[idx][key] + noise.to(device)
                else:
                    temp = temp + weights[idx] * all_clients_params[idx][key]
        
        w_glob[key] = temp

    return w_glob

   