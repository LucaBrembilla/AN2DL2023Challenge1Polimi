import numpy as np

def preprocess_data(X, y):
    # Clean dataset
    shrek = X[58]
    troll = X[2150]
    shrek_indexes = []
    troll_indexes = []
    
    for i in range(len(X)):
        shrekDiff = np.mean(np.abs(shrek - X[i]))
        trollDiff = np.mean(np.abs(troll - X[i]))
        if shrekDiff == 0.0:
            shrek_indexes.append(i)
        elif trollDiff == 0.0:
            troll_indexes.append(i)

    rm_indexes = np.concatenate((shrek_indexes, troll_indexes))

    X_clean = np.delete(X, rm_indexes, axis=0)
    y_clean = np.delete(y, rm_indexes, axis=0)

    # Change labels to {0, 1}
    y_clean = (np.array(y_clean) == 'unhealthy').astype(int)

    return X_clean, y_clean
