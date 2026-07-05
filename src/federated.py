import torch
import torch.nn as nn
import copy
import numpy as np
from defense import apply_dp_flat, apply_dp_adaptive, compute_skew_score


def local_train(model, dataloader, epochs=1, lr=0.01, device='cpu'):
    """
    Train a copy of the global model on one client's local data.
    Returns the updated model's state_dict (weights).
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def federated_average(state_dicts):
    """
    Average a list of model state_dicts (weights) into one global state_dict.
    """
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state.keys():
        for i in range(1, len(state_dicts)):
            avg_state[key] += state_dicts[i][key]
        avg_state[key] = avg_state[key] / len(state_dicts)
    return avg_state


def evaluate(model, dataloader, device='cpu'):
    """
    Evaluate model accuracy on a dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def federated_train_with_defense(global_model, train_dataset, client_data, train_labels,
                                   num_clients, num_rounds, defense_mode='none',
                                   epochs=1, lr=0.01, batch_size=32, device='cpu'):
    """
    Run FedAvg training with an optional defense applied to each client's update.
    defense_mode: 'none', 'flat', or 'adaptive'
    """
    from torch.utils.data import DataLoader, Subset

    global_counts = np.array([np.sum(np.array(train_labels) == c) for c in range(10)])
    global_dist = global_counts / global_counts.sum()

    for round_num in range(num_rounds):
        client_state_dicts = []

        for client_id in range(num_clients):
            subset = Subset(train_dataset, client_data[client_id])
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

            current_global_state = copy.deepcopy(global_model.state_dict())
            state_dict = local_train(global_model, dataloader, epochs=epochs, lr=lr, device=device)

            if defense_mode == 'flat':
                state_dict = apply_dp_flat(state_dict, current_global_state, clip_norm=5.0, noise_scale=0.01)
            elif defense_mode == 'adaptive':
                client_label_subset = [train_labels[i] for i in client_data[client_id]]
                skew = compute_skew_score(client_label_subset, global_dist)
                state_dict = apply_dp_adaptive(state_dict, current_global_state, skew_score=skew, base_clip=5.0, base_noise=0.01)
            

            client_state_dicts.append(state_dict)

        new_global_state = federated_average(client_state_dicts)
        global_model.load_state_dict(new_global_state)

    return global_model