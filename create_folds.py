## 5 fold cross validation
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

def create_folds(data, n_splits=5, random_state=42):
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Create a new column 'fold' and initialize it with -1
    data['fold'] = -1
    
    # Iterate through each fold
    for fold, (train_index, val_index) in enumerate(skf.split(data, data['Survived'])):
        data.loc[val_index, 'fold'] = fold
    
    return data
