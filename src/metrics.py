import numpy as np


def hinge_loss(y, y_pred):
    """
    Výpočet hinge straty.
    
    Args:
        y (numpy.ndarray): Pole pravdivých hodnôt.
        y_pred (numpy.ndarray): Pole predpovedaných hodnôt.
        
    Returns:
        numpy.ndarray: Hodnoty strát pre jednotlivé príklady.
    """
    return np.max([np.zeros(y.shape[0]), 1 - y * y_pred])

def accuracy(y, y_pred):
    """
    Vypočíta presnosť klasifikácie.
    
    Args:
        y (numpy.ndarray): Pole pravdivých hodnôt.
        y_pred (numpy.ndarray): Pole predpovedaných hodnôt.
        
    Returns:
        float: Presnosť klasifikácie.
    """
    # Indexy pozitívnych tried
    idx = np.where(y_pred == 1)
    # Počet správne klasifikovaných pozitívnych príkladov
    true_positives = np.sum(y_pred[idx] == y[idx])
    
    # Indexy negatívnych tried
    idx = np.where(y_pred == -1)
    # Počet nesprávne klasifikovaných negatívnych príkladov
    true_negatives = np.sum(y_pred[idx] == y[idx])
    
    # Celková presnosť
    return float(true_positives + true_negatives) / len(y)

def recall(y, y_pred):
    """
    Výpočet recallu.
    
    Args:
        y (numpy.ndarray): Pole pravdivých hodnôt.
        y_pred (numpy.ndarray): Pole predpovedaných hodnôt.
        
    Returns:
        float: recall.
    """
    # Počet správne klasifikovaných pozitívnych príkladov
    true_positives = np.sum((y_pred == 1) & (y == 1))
    # Počet nesprávne klasifikovaných pozitívnych príkladov
    false_negatives = np.sum((y_pred == -1) & (y == 1))
    # Výpočet recallu
    return true_positives / (true_positives + false_negatives)

def precision(y, y_pred):
    """
    Výpočet presnosti klasifikácie.
    
    Args:
        y (numpy.ndarray): Pole pravdivých hodnôt.
        y_pred (numpy.ndarray): Pole predpovedaných hodnôt.
        
    Returns:
        float: Presnosť klasifikácie.
    """
    # Počet správne klasifikovaných pozitívnych príkladov
    true_positives = np.sum((y_pred == 1) & (y == 1))
    # Počet nesprávne klasifikovaných negatívnych príkladov
    false_positives = np.sum((y_pred == 1) & (y == -1))
    # Výpočet presnosti
    return true_positives / (true_positives + false_positives)

def f1_score(y, y_pred):
    """
    Výpočet F1 skóre ako harmonického priemeru presnosti a odvolania.
    
    Args:
        y (numpy.ndarray): Pole pravdivých hodnôt.
        y_pred (numpy.ndarray): Pole predpovedaných hodnôt.
        
    Returns:
        float: F1 skóre.
    """
    # Výpočet presnosti a odvolania
    prec = precision(y, y_pred)
    rec = recall(y, y_pred)
    # Výpočet F1 skóre
    return 2 * (prec * rec) / (prec + rec)
