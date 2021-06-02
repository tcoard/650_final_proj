#Code adapted from https://github.com/facebookresearch/esm/blob/master/examples/variant_prediction.ipynb
# & https://github.com/nafizh/TRAC/blob/master/utils.py

# This file runs 2 classifiers using the embeddings from the Facebook AI paper: ESM
# The classifiers are sent through a grid search for the best model parameters and then 
# passed through a 10-fold cross-validator with fold data printed out to the terminal

# CJ

import os
import torch
import numpy as np
import pandas as pd

import esm

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import matthews_corrcoef, make_scorer, accuracy_score, recall_score, precision_score, f1_score, multilabel_confusion_matrix

# Functions for explicitly calculating specificity, fall out, and miss rate
def mcm_specificity(clf, X, y):
    y_pred = clf.predict(X)
    mcm = multilabel_confusion_matrix(y, y_pred)
    metric = mcm[:, 0, 0] / (mcm[:, 0, 0] + mcm[:, 0, 1])
    # Be rid of NANs
    cleaned_metric = [x for x in metric if np.isnan(x) == False]
    #Return average for all classes
    if not cleaned_metric:
        return 1
    else:
        return sum(cleaned_metric)/len(cleaned_metric)

def mcm_fpr(clf, X, y):
    y_pred = clf.predict(X)
    mcm = multilabel_confusion_matrix(y, y_pred)
    # Or the fall out
    metric = mcm[:, 0, 1] / (mcm[:, 0, 1] + mcm[:, 1, 0])
    # Be rid of NANs
    cleaned_metric = [x for x in metric if np.isnan(x) == False]
    #Return average for all classes
    if not cleaned_metric:
        return 1
    else:
        return sum(cleaned_metric)/len(cleaned_metric)

def mcm_fnr(clf, X, y):
    y_pred = clf.predict(X)
    mcm = multilabel_confusion_matrix(y, y_pred)
    # Or the miss rate
    metric = mcm[:, 1, 0] / (mcm[:, 1, 0] + mcm[:, 1, 1])
    # Be rid of NANs
    cleaned_metric = [x for x in metric if np.isnan(x) == False]
    #Return average for all classes
    if not cleaned_metric:
        return 1
    else:
        return sum(cleaned_metric)/len(cleaned_metric)

# Function to run cross-validation (10 folds) and print out various metrics
def get_scores(model, x, y):
    cv_scoring = {'accuracy': make_scorer(accuracy_score, normalize=True),
        'false_positive_rate': mcm_fpr,
        'false_negative_rate': mcm_fnr,
        'recall': make_scorer(recall_score, average='macro'),
        'precision': make_scorer(precision_score, average='macro'),
        'specificity': mcm_specificity,
        'mcc': make_scorer(matthews_corrcoef),
        'f1': make_scorer(f1_score, average='macro'),
        }
    folds = 10
    cross_val_scores = cross_validate(model, x, y, cv=folds, scoring=cv_scoring)
    for i in range(folds):
        print(f'Model Number {i+1}')
        print(f'Accuracy: {cross_val_scores["test_accuracy"][i]}')
        print(f'False Positives: {cross_val_scores["test_false_positive_rate"][i]}')
        print(f'False Negatives: {cross_val_scores["test_false_negative_rate"][i]}')
        print(f'Recall: {cross_val_scores["test_recall"][i]}')
        print(f'Precision: {cross_val_scores["test_precision"][i]}')
        print(f'Specificity: {cross_val_scores["test_specificity"][i]}')
        print(f'MCC: {cross_val_scores["test_mcc"][i]}')
        print(f'F1 Score: {cross_val_scores["test_f1"][i]}')

# Antibiotic dicitonary. Lables are converted to indices, and a soft-voting classification strategy is employed
ab_res_class_dict = {0:'BETA-LACTAM', 1:'AMINOGLYCOSIDE', 2:'TETRACYCLINE', 3:'GLYCOPEPTIDE', 
                     4:'PHENICOL', 5:'FOLATE-SYNTHESIS-INHABITOR', 6:'RIFAMYCIN', 7:'TRIMETHOPRIM', 
                     8:'SULFONAMIDE', 9:'MACROLIDE', 10:'FOSFOMYCIN', 11:'QUINOLONE', 
                     12:'STREPTOGRAMIN', 13:'MACROLIDE/LINCOSAMIDE/STREPTOGRAMIN', 
                     14: 'MULTIDRUG', 15: 'BACITRACIN'}

# Set paths for the 
FASTA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "esm40.fa")
EMB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeds", "coala_40_reprs")
EMB_LAYER = 34

ys = []
Xs = []
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    scaled_effect = header.split('|')[-1]
    found = False
    for key, value in ab_res_class_dict.items():
        if scaled_effect == value:
            ys.append(key)
            found = True
            break
    if not found:
        print("Could not find dictionary entry for label: "+scaled_effect)
        continue
    file_name = header[1:].replace('|', '_')
    fn = f'{EMB_PATH}/{file_name}.pt'
    embs = torch.load(fn)
    Xs.append(embs['mean_representations'][EMB_LAYER])
Xs = torch.stack(Xs, dim=0).numpy()
print(len(ys))
print(Xs.shape)

train_size = 0.8
Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, train_size=train_size, random_state=42)

pca = PCA(60)
Xs_train_pca = pca.fit_transform(Xs_train)

knn_grid = {
    'n_neighbors': [5, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size' : [15, 30],
    'p' : [1, 2],
}

svm_grid = {
    'C' : [0.1, 1.0, 10.0],
    'kernel' :['linear', 'poly', 'rbf', 'sigmoid'],
    'degree' : [3],
    'gamma': ['scale'],
}

cls_list = [KNeighborsClassifier, SVC]
param_grid_list = [knn_grid, svm_grid]


result_list = []
grid_list = []
for cls_name, param_grid in zip(cls_list, param_grid_list):
    print(cls_name)
    grid = GridSearchCV(
        estimator = cls_name(), 
        param_grid = param_grid,
        scoring = 'r2',
        verbose = 1,
        n_jobs = -1,
        cv = 10
    )
    grid.fit(Xs_train_pca, ys_train)
    result_list.append(pd.DataFrame.from_dict(grid.cv_results_))
    grid_list.append(grid)

result_list[0].sort_values('rank_test_score')[:5]
result_list[1].sort_values('rank_test_score')

Xs_test_pca = pca.transform(Xs_test)
for grid in grid_list:
    print(grid.best_estimator_)
    print()
    preds = grid.predict(Xs_test_pca)
    get_scores(grid, Xs_test_pca, ys_test)