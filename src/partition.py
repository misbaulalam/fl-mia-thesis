import numpy as np

def dirichlet_partition(labels, num_clients, alpha, num_classes=10, seed=42):
    """
    Split dataset indices among clients using a Dirichlet distribution.
    
    labels: array of labels for the full dataset (e.g., all 50000 CIFAR-10 train labels)
    num_clients: how many clients to split data into
    alpha: concentration parameter. Lower = more skewed/Non-IID, higher = more IID
    num_classes: number of classes in the dataset (10 for CIFAR-10)
    
    
    Returns: a dict {client_id: list of sample indices}
    """
    np.random.seed(seed)
    labels = np.array(labels)
    client_indices = {i: [] for i in range(num_clients)}

    for class_id in range(num_classes):
        # get all indices belonging to this class
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)

        # sample proportions for how this class's samples split across clients
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # convert proportions into actual split points
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, split_points)

        for client_id in range(num_clients):
            client_indices[client_id].extend(split_indices[client_id].tolist())

    return client_indices