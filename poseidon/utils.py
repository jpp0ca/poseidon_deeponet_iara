import torch
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(labels, device):
    """
    Calculate class weights for the training dataset to handle class imbalance.
    
    Args:
        train_dataset_fold (TimeSeriesPairDataset): The training dataset.
        
    Returns:
        np.ndarray: Class weights for each class.
    """
    # print ("pegando os labels do zip(*train_dataset_fold)")
    # _, _, labels = zip(*train_dataset_fold)
    
    print ("tranformando os labels em tensor")
    labels = torch.tensor(labels, device=device)
    
    print("transformando labels em np array")
    labels = np.array([label.cpu().numpy() for label in labels])
    
    print("calculando class_weight")
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return class_weights