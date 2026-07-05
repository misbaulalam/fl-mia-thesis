import torch
import copy
import numpy as np

def apply_dp_flat(state_dict, global_state_dict, clip_norm=5.0, noise_scale=0.01):
    """
    Apply flat Differential Privacy to a client's model UPDATE (not raw weights).
    This clips the change from the global model, then adds noise, then applies it back.
    """
    clipped_state = copy.deepcopy(state_dict)

    for key in clipped_state.keys():
        param = clipped_state[key]
        global_param = global_state_dict[key]

        if param.dtype != torch.float32:
            continue

        # compute the update (how much this client changed the weights)
        update = param - global_param

        # clip the update's norm
        update_norm = torch.norm(update)
        if update_norm > clip_norm:
            update = update * (clip_norm / update_norm)

        # add noise to the update
        noise = torch.normal(mean=0, std=noise_scale, size=update.shape)
        update = update + noise

        # apply the noisy, clipped update back on top of the global weights
        clipped_state[key] = global_param + update

    return clipped_state

def compute_skew_score(client_labels, global_label_dist, num_classes=10):
    """
    Measure how much a client's label distribution deviates from the global one,
    using KL divergence. Higher score = more skewed/heterogeneous client.
    """
    client_labels = np.array(client_labels)
    client_counts = np.array([np.sum(client_labels == c) for c in range(num_classes)])
    client_dist = client_counts / client_counts.sum()

    epsilon = 1e-8
    client_dist = client_dist + epsilon
    global_dist = global_label_dist + epsilon

    kl_div = np.sum(client_dist * np.log(client_dist / global_dist))
    return kl_div


def apply_dp_adaptive(state_dict, global_state_dict, skew_score, base_clip=5.0, base_noise=0.01, 
                       skew_scale=0.5, max_skew_for_scaling=2.0):
    """
    Apply Differential Privacy with strength scaled to the client's skew score.
    """
    normalized_skew = min(skew_score / max_skew_for_scaling, 1.0)
    
    adaptive_clip = base_clip * (1 - skew_scale * normalized_skew)
    adaptive_noise = base_noise * (1 + skew_scale * normalized_skew)
    
    return apply_dp_flat(state_dict, global_state_dict, clip_norm=adaptive_clip, noise_scale=adaptive_noise)
    