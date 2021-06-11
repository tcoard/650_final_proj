# https://romanorac.github.io/machine/learning/2019/12/02/identifying-hate-speech-with-bert-and-cnn.html
import random
import glob
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    multilabel_confusion_matrix,
)

from torch.autograd import Variable


#MODEL = "/ifs/groups/rosenMRIGrp/tgc37/650_proj/esm/model.pkl"
#PRED_FILE = "/ifs/groups/rosenMRIGrp/tgc37/test_pred_labels_subsamp100.pkl"



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = 40
#FASTA = f"/home/tcoard/w/650_proj/data/esm{DATASET}.fa"
EMB_PATH = f"/ifs/groups/rosenMRIGrp/tgc37/650_proj/esm/per_tok_{DATASET}"
#MODEL = f"/home/tcoard/w/650_proj/data/esm_cnn{DATASET}"
EMB_PATH = f"/ifs/groups/rosenMRIGrp/tgc37/650_proj/esm/per_tok_{DATASET}"
SEQ_SUBSET = 100
BATCH_SIZE = 10
NUM_FOLDS = 5

# MAX_LEN = 1024
MAX_LEN = 1024

np.random.seed(0)

def compute_performance(preds, labels):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    tpr_all = list()
    fpr_all = list()
    for t in range(0, 100):
        threshold = t / 100.0
        predictions = (preds >= threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        tpr = 0
        fpr = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            tn = len(predictions[i, :]) - (tp + fp + fn)

            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
                tpr += tp / (1.0 * (tp + fn))
                fpr += fp / (1.0 * (fp + tn))
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        tpr_all.append(tpr / total)
        fpr_all.append(fpr / total)
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max, tpr_all, fpr_all


def mcm_specificity(y, y_pred):
    mcm = multilabel_confusion_matrix(y, y_pred)
    metric = mcm[:, 0, 0] / (mcm[:, 0, 0] + mcm[:, 0, 1])
    # Be rid of NANs
    cleaned_metric = [x for x in metric if np.isnan(x) == False]
    # Return average for all classes
    if not cleaned_metric:
        return 1
    else:
        return sum(cleaned_metric) / len(cleaned_metric)


def mcm_fpr(y, y_pred):
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


def mcm_fnr(y, y_pred):
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


class KimCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(KimCNN, self).__init__()

        V = embed_num
        D = embed_dim
        C = class_num
        Co = kernel_num
        Ks = kernel_sizes

        self.static = static
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks]).to(DEVICE)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C).to(DEVICE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        output = self.sigmoid(logit)
        return output


def one_hot_encode(resistance, resistance_categories):
    one_hot = [0] * len(resistance_categories)
    one_hot[resistance_categories.index(resistance)] = 1

    return np.array(one_hot, dtype="int32")


def make_trial_data():

    drug_emb = dict()
    for f in glob.glob(f"{EMB_PATH}/*.pt"):
        entry = torch.load(f)
        seq = entry["representations"][34]
        if len(seq) <= MAX_LEN:
            drug = entry["label"].split("|")[-1].strip()
            if not drug_emb.get(drug):
                drug_emb[drug] = [seq.numpy()]
            else:
                drug_emb[drug].append(seq.numpy())

    sequences = list()
    resistances = list()
    for drug in drug_emb:
        if len(drug_emb[drug]) >= SEQ_SUBSET:
        #    sequences.extend(random.sample(drug_emb[drug], SEQ_SUBSET))
        #    resistances.extend([drug] * SEQ_SUBSET)  # len(drug_emb[drug]))
            sequences.extend(drug_emb[drug])
            resistances.extend([drug] * len(drug_emb[drug]))

            #sequences.extend(drug_emb[drug])
            #resistances.extend([drug] * len(drug_emb[drug]))

    #            sequences.append(seq.numpy())
    #            #if not np.any(np.isnan(mean_rep)) and np.all(np.isfinite(mean_rep)):
    #            #headers.append(entry["label"])
    #            resistances.append(entry["label"].split("|")[-1].strip())

    # get counts of resistences and remove those with count < 100
    # for resistences left, choose 100 randomly
    # random(len(

    # with open(FASTA, "r") as f:
    #     header = ""
    #     for line in f:
    #         line = line.strip()
    #         if line.startswith(">"):
    #             header = line
    #         else:
    #             if len(line) <= MAX_LEN:
    #                 headers.append(header)
    #                 resistances.append(header.split("|")[-1])
    #                 sequences.append(" ".join(line))
    #                 header = ""

    resistance_categories = sorted(set(resistances))

    # this seems inefficient. I have note used dataframes a ton before
    columns = ["sequence", "resistances"]
    columns.extend(resistance_categories)
    df = pd.DataFrame(columns=columns)
    for res, seq in zip(resistances, sequences):
        row = [seq, res]
        row.extend(one_hot_encode(res, resistance_categories))
        df_row = pd.DataFrame([row], columns=columns)
        df = df.append(df_row)
    print(df.shape)

    # _, df, _ = np.split(df.sample(frac=1, random_state=42), [int(0.90 * len(df)), int(0.95 * len(df))])

    # _, df, _ = np.split(df.sample(frac=1, random_state=42), [int(0.00 * len(df)), int(0.50 * len(df))])
    # print(df.shape)
    # df = df.groupby("").sample(n=1, random_state=1)

    return df, resistance_categories


def tokenize_text(df, tokenizer):
    return [tokenizer.encode(text, add_special_tokens=True)[:MAX_LEN] for text in df.sequence.values]


def pad_text(tokenized_text):
    return torch.tensor(
        [list(el) + ([[0] * 1280] * (MAX_LEN - len(el))) for el in tokenized_text], dtype=torch.float32
    ).to(DEVICE)

    # return np.array([list(el) + ([[0]*1280] * (MAX_LEN - len(el))) for el in tokenized_text], dtype="int32")
    # return np.array([el + [0] * (MAX_LEN - len(el)) for el in tokenized_text], dtype="int32")


def tokenize_and_pad_text(df, tokenizer):
    tokenized_text = tokenize_text(df, tokenizer)
    padded_text = pad_text(tokenized_text)
    return torch.tensor(padded_text, dtype=torch.int32)


def targets_to_tensor(df, target_columns):
    return torch.tensor(np.array(df[target_columns].values, dtype="int32"), dtype=torch.int32)


def generate_batch_data(x, y, batch_size):
    for batch, i in enumerate(range(0, len(x) - batch_size, batch_size), 1):
        x_batch = x[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        yield x_batch, y_batch, batch
    if i + batch_size < len(x):
        yield x[i + batch_size :], y[i + batch_size :], batch + 1
    if batch == 0:
        yield x, y, 1


def print_gpu_mem():
    # t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print(f"total: {f}, reserved: {r}, allocated: {a}, free: {f}")


def model_batch(indices, model):
    x = list()
    l = len(indices)
    for ndx in range(0, l, BATCH_SIZE):
        x.extend(model(indices[ndx : min(ndx + BATCH_SIZE, l)])[0])
    x = torch.stack(x)  # .to(DEVICE)
    # x = torch.tensor(x).to(DEVICE)
    return x

def run_one_fold(fold_num, train, df_test, target_columns):

    #df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(0.6 * len(df)), int(0.8 * len(df))])
    df_train, df_val, _ = np.split(train.sample(frac=1, random_state=42), [int(0.8 * len(train)), len(train)+1])
    with torch.no_grad():
        x_train = pad_text(df_train.sequence.values)
        x_val = pad_text(df_val.sequence.values)
        x_test = pad_text(df_test.sequence.values)
    y_train = targets_to_tensor(df_train, target_columns)
    y_val = targets_to_tensor(df_val, target_columns)
    y_test = targets_to_tensor(df_test, target_columns)

    #####
    embed_num = x_train.shape[1]
    embed_dim = x_train.shape[2]
    class_num = y_train.shape[1]
    kernel_num = 3
    kernel_sizes = [2, 3, 4]
    dropout = 0.5
    static = True

    model = KimCNN(
        embed_num=embed_num,
        embed_dim=embed_dim,
        class_num=class_num,
        kernel_num=kernel_num,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        static=static,
    )

    n_epochs = 50
    batch_size = 10
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss().to(DEVICE)

    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0

        model.train(True)
        for x_batch, y_batch, batch in generate_batch_data(x_train, y_train, batch_size):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_pred = model(x_batch).to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(y_pred.float(), y_batch.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= batch
        train_losses.append(train_loss)
        elapsed = time.time() - start_time

        model.eval()  # disable dropout for deterministic output
        with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
            val_loss, batch = 0, 1
            for x_batch, y_batch, batch in generate_batch_data(x_val, y_val, batch_size):
                y_pred = model(x_batch).to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                loss = loss_fn(y_pred.float(), y_batch.float())
                val_loss += loss.item()
            val_loss /= batch
            val_losses.append(val_loss)

        # print(
        #     "Epoch %d Train loss: %.2f. Validation loss: %.2f. Elapsed time: %.2fs."
        #     % (epoch + 1, train_losses[-1], val_losses[-1], elapsed)
        # )
    #torch.save(model.state_dict(), f"{MODEL}_fold{fold_num+1}.pkl")
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig("cnn_losses.png")

    # model = train_model()
    # model = torch.load(MODEL)

    model.eval()  # disable dropout for deterministic output
    with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
        y_preds = []
        batch = 0
        for x_batch, y_batch, batch in generate_batch_data(x_test, y_test, batch_size):
            y_pred = model(x_batch)
            y_preds.extend(y_pred.cpu().numpy().tolist())
        y_preds_np = np.array(y_preds)
    y_test_np = df_test[target_columns].values.astype("float")
    idx = np.argwhere(np.all(y_test_np[..., :] == 0, axis=0))
    #pickle.dump([y_test_np, y_preds_np, target_columns], open("test_pred_labels_before_removal.pkl", "wb"))
    y_test_np = np.delete(y_test_np, idx, axis=1)
    y_preds_np = np.delete(y_preds_np, idx, axis=1)
    for i in sorted(idx, reverse=True):
        i = i[0]
        target_columns.remove(target_columns[i])
    #pickle.dump([y_test_np, y_preds_np, target_columns], open("test_pred_labels_new.pkl", "wb"))
    auc_scores = roc_auc_score(y_test_np, y_preds_np, average=None)
    df_accuracy = pd.DataFrame({"label": target_columns, "auc": auc_scores})
    print(df_accuracy.sort_values("auc")[::-1])


    fmax =0
    thres =0
    acc = 0
    fpr = 0
    fnr = 0
    rec = 0
    pre = 0
    spec =0
    mat = 0
    f_max, p_max, r_max, t_max, predictions_max, tpr_all, fpr_all = compute_performance(y_preds_np, y_test_np)
    print(f"{f_max=}{p_max=}{r_max=}{t_max=}{predictions_max=}")

    for t in range(0, 100):
        threshold = t / 100.0
        #predictions = (preds >= threshold).astype(np.int32)
        y_preds_thres = (y_preds_np >= threshold).astype(np.int32)

        f1 = f1_score(y_test_np, y_preds_thres, average='macro')
        if f1>fmax:
            fmax = f1
            thres = t_max
            acc = accuracy_score(y_test_np, y_preds_thres)
            fpr = mcm_fpr(y_test_np, y_preds_thres)
            fnr = mcm_fnr(y_test_np, y_preds_thres)
            rec = recall_score(y_test_np, y_preds_thres, average='macro')
            pre = precision_score(y_test_np, y_preds_thres, average='macro')
            spec = mcm_specificity(y_test_np, y_preds_thres)
            mat = matthews_corrcoef(y_test_np.argmax(axis=1), y_preds_thres.argmax(axis=1))




    print(f"Model Number {fold_num+1}")
    print(f"Accuracy: {acc}")
    print(f"False Positives: {fpr}")
    print(f"False Negatives: {fnr}")
    print(f"Recall: {rec}")
    print(f"Precision: {pre}")
    print(f"Specificity: {spec}")
    print(f"MCC: {mat}")
    print(f"F1 Score: {fmax}")
    print(f"Threshold: {thres}")

def main():
    df, target_columns = make_trial_data()
    skf = StratifiedKFold(n_splits=NUM_FOLDS)
    t = df.resistances
    del df["resistances"]
    for fold_num, (train_index, test_index) in enumerate(skf.split(np.zeros(len(t)), t)):
        train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        run_one_fold(fold_num, train, df_test, target_columns)

if __name__ == "__main__":
    main()
