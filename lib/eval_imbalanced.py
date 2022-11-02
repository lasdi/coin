import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score

def eval_imbalanced(y_test, y_score, classes, do_plot=True):
    
    """
    Sensitivity (TPR) and specificity (TNR) calculation
    per class for scikit-learn machine learning algorithms.
    
    -------
    cm : ndarray
        Confusion matrix obtained with `sklearn.metrics.confusion_matrix`
        method.
    
    Returns
    -------
    sensitivities : ndarray
        Array of sensitivity values per each class.
    
    specificities : ndarray
        Array of specificity values per each class.
    """
    
    cm = confusion_matrix(y_test, y_score)
    
    # Sensitivity = TP/(TP + FN)
    # TP of a class is a diagonal element
    # Sum of all values in a row is TP + FN
    # So, we can vectorize it this way:
    sensitivities = np.diag(cm) / np.sum(cm, axis=1)

    # Specificity = TN/(TN + FP)
    # FP is the sum of all values in a column excluding TP (diagonal element)
    # TN of a class is the sum of all cols and rows excluding this class' col and row
    # A bit harder case...
    # TN + FP
    cm_sp = np.tile(cm, (cm.shape[0], 1, 1))
    z = np.zeros(cm_sp.shape)
    ids = np.arange(cm_sp.shape[0])

    # Placing a row mask
    # That will be our TN + FP vectorized calculation
    z[ids, ids, :] = 1
    tnfp = np.ma.array(cm_sp, mask=z).sum(axis=(1, 2))

    # TN
    # Now adding a column mask
    z[ids, :, ids] = 1
    tn = np.ma.array(cm_sp, mask=z).sum(axis=(1, 2))

    # Finally, calculating specificities per each class
    specificities = (tn / tnfp).filled()
    
    # Printing
    print('Confusion Matriz:')
    print(cm)

    print ('\nClass\tSens.\t\tSpec.')
    for c in range(len(classes)):
        print ('%s\t\t%.02f%%\t\t%.02f%%' %(classes[c], sensitivities[c]*100,specificities[c]*100))
    
    accuracy = accuracy_score(y_test, y_score)
    print('> Accuracy: %.02f%%' % (accuracy*100))

    auc = roc_auc_score(y_test, y_score)
    print('> auc: %.04f%%' % (auc))    
    

    if do_plot==True:        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()
    
    return sensitivities, specificities, accuracy
