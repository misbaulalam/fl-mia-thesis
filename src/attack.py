import torch
import numpy as np

def get_confidence_and_loss(model, dataloader, device='cpu'):
    """
    Run the model on a dataset and extract, for each sample:
    - the model's confidence (max softmax probability)
    - the loss value
    These are the features the attack model will use.
    """
    model.eval()
    confidences = []
    losses = []
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            confidences.extend(confidence.cpu().numpy())
            
            loss = criterion(outputs, labels)
            losses.extend(loss.cpu().numpy())

    return np.array(confidences), np.array(losses)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def build_attack_dataset(member_conf, member_loss, nonmember_conf, nonmember_loss):
    """
    Combine member and non-member features into one labeled dataset for the attack model.
    Label 1 = member, Label 0 = non-member.
    """
    X_member = np.stack([member_conf, member_loss], axis=1)
    X_nonmember = np.stack([nonmember_conf, nonmember_loss], axis=1)

    X = np.concatenate([X_member, X_nonmember], axis=0)
    y = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_nonmember))])

    return X, y


def run_mia_attack(X, y, test_size=0.3, seed=42):
    """
    Train a simple attack classifier (logistic regression) to distinguish
    members from non-members, and report attack performance metrics.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    attack_model = LogisticRegression()
    attack_model.fit(X_train, y_train)

    y_pred = attack_model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    return metrics