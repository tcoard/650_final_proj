# Code adapted from https://github.com/agemagician/ProtTrans/blob/master/Embedding/PyTorch/Basic/ProtBert.ipynb
# & https://github.com/nafizh/TRAC/blob/master/utils.py

# This file runs 2 classifiers using the embeddings from ProtBERT
# The classifiers are sent through a grid search for the best model parameters and then
# passed through a 10-fold cross-validator with fold data printed out to the terminal

# CJ, edited for ProtBERT by Thomas

import os
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    matthews_corrcoef,
    make_scorer,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    multilabel_confusion_matrix,
)

# Antibiotic dicitonary. Lables are converted to indices, and a soft-voting classification strategy is employed
# Set paths for the
DATASET = 100
FASTA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fasta", f"pbert{DATASET}.fa")
EMB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeds", f"protbert_mean_embeddings{DATASET}.pkl")
NUM_FOLDS = 10
MAX_SEQ_LEN = 1024
MIN_DRUG_RES = 50

# Functions for explicitly calculating specificity, fall out, and miss rate
def mcm_specificity(clf, X, y):
    y_pred = clf.predict(X)
    mcm = multilabel_confusion_matrix(y, y_pred)
    metric = mcm[:, 0, 0] / (mcm[:, 0, 0] + mcm[:, 0, 1])
    # Be rid of NANs
    cleaned_metric = [x for x in metric if np.isnan(x) == False]
    # Return average for all classes
    if not cleaned_metric:
        return 1
    else:
        return sum(cleaned_metric) / len(cleaned_metric)


def mcm_fpr(clf, X, y):
    y_pred = clf.predict(X)
    mcm = multilabel_confusion_matrix(y, y_pred)
    # Or the fall out
    metric = mcm[:, 0, 1] / (mcm[:, 0, 1] + mcm[:, 1, 0])
    # Be rid of NANs
    cleaned_metric = [x for x in metric if np.isnan(x) == False]
    # Return average for all classes
    if not cleaned_metric:
        return 1
    else:
        return sum(cleaned_metric) / len(cleaned_metric)


def mcm_fnr(clf, X, y):
    y_pred = clf.predict(X)
    mcm = multilabel_confusion_matrix(y, y_pred)
    # Or the miss rate
    metric = mcm[:, 1, 0] / (mcm[:, 1, 0] + mcm[:, 1, 1])
    # Be rid of NANs
    cleaned_metric = [x for x in metric if np.isnan(x) == False]
    # Return average for all classes
    if not cleaned_metric:
        return 1
    else:
        return sum(cleaned_metric) / len(cleaned_metric)


# Function to run cross-validation (10 folds) and print out various metrics
def get_scores(model, x, y):
    cv_scoring = {
        "accuracy": make_scorer(accuracy_score, normalize=True),
        "false_positive_rate": mcm_fpr,
        "false_negative_rate": mcm_fnr,
        "recall": make_scorer(recall_score, average="macro"),
        "precision": make_scorer(precision_score, average="macro"),
        "specificity": mcm_specificity,
        "mcc": make_scorer(matthews_corrcoef),
        "f1": make_scorer(f1_score, average="macro"),
    }
    folds = NUM_FOLDS
    cross_val_scores = cross_validate(model, x, y, cv=folds, scoring=cv_scoring)
    for i in range(folds):
        print(f"Model Number {i+1}")
        print(f'Accuracy: {cross_val_scores["test_accuracy"][i]}')
        print(f'False Positives: {cross_val_scores["test_false_positive_rate"][i]}')
        print(f'False Negatives: {cross_val_scores["test_false_negative_rate"][i]}')
        print(f'Recall: {cross_val_scores["test_recall"][i]}')
        print(f'Precision: {cross_val_scores["test_precision"][i]}')
        print(f'Specificity: {cross_val_scores["test_specificity"][i]}')
        print(f'MCC: {cross_val_scores["test_mcc"][i]}')
        print(f'F1 Score: {cross_val_scores["test_f1"][i]}')


def create_embeddings():
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert")
    fe = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=0)
    header_emb = dict()
    with open(FASTA_PATH, "r") as fasta:
        header = ""
        for line in fasta:
            line = line.strip()
            if line.startswith(">"):
                header = line
            else:
                if len(line) <= MAX_SEQ_LEN:
                    embedding = fe(" ".join(line))
                    seq_emb = embedding[0][1 : len(line) + 1]
                    avg_emb = np.ma.average(seq_emb, axis=0)
                    final_emb = np.empty((MAX_SEQ_LEN), dtype=float)
                    for i, val in enumerate(avg_emb):
                        final_emb[i] = val
                    header_emb[header] = final_emb
                header = ""

    pickle.dump(header_emb, open(EMB_PATH, "wb"))
    return header_emb


def filter_embeddings(header_emb):
    res_count = dict()
    for header in header_emb:
        res = header.split("|")[-1]
        if res_count.get(res) is None:
            res_count[res] = 1
        else:
            res_count[res] += 1

    res_to_keep = sorted(filter(lambda res: res_count[res] > MIN_DRUG_RES, list(res_count)))
    for header in list(header_emb):
        res = header.split("|")[-1]
        if res not in res_to_keep:
            del header_emb[header]
    return header_emb, res_to_keep


def main():
    header_emb = dict()
    if os.path.isfile(EMB_PATH):
        header_emb = pickle.load(open(EMB_PATH, 'rb'))
    else:
        header_emb = create_embeddings()
    header_emb, resistances = filter_embeddings(header_emb)

    ys = []
    Xs = []
    for header in header_emb:
        Xs.append(header_emb[header])
        ys.append(resistances.index(header.split("|")[-1]))

    train_size = 0.8
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, train_size=train_size, random_state=42)

    pca = PCA(60)
    Xs_train_pca = pca.fit_transform(Xs_train)

    knn_grid = {
        "n_neighbors": [5, 10],
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
        "leaf_size": [15, 30],
        "p": [1, 2],
    }

    svm_grid = {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [3],
        "gamma": ["scale"],
    }

    cls_list = [KNeighborsClassifier, SVC]
    param_grid_list = [knn_grid, svm_grid]

    grid_list = []
    for cls_name, param_grid in zip(cls_list, param_grid_list):
        print(cls_name)
        grid = GridSearchCV(
            estimator=cls_name(), param_grid=param_grid, scoring="r2", verbose=1, n_jobs=-1, cv=NUM_FOLDS
        )
        grid.fit(Xs_train_pca, ys_train)
        grid_list.append(grid)

    Xs_test_pca = pca.transform(Xs_test)
    for grid in grid_list:
        print(grid.best_estimator_)
        print()
        get_scores(grid, Xs_test_pca, ys_test)


if __name__ == "__main__":
    main()
