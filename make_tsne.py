"""
from keras.datasets import mnist
https://towardsdatascience.com/t-distributed-stochastic-neighbor-embedding-t-sne-bb60ff109561
https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-4
"""

import glob
import random
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import torch

EMBEDDER = "protbert"
# EMBEDDER = "esm"
COALA_DATASET = 40
EMB_PATH = f"../{EMBEDDER}/650_{COALA_DATASET}/"
SEQ_SUBSET = 100


def parse_embedder():
    num_nan_or_inf = 0
    drug_emb = dict()
    if EMBEDDER == "esm":
        for f in glob.glob(f"{EMB_PATH}*.pt"):
            entry = torch.load(f)
            mean_rep = entry["mean_representations"][34].numpy()
            if not np.any(np.isnan(mean_rep)) and np.all(np.isfinite(mean_rep)):
                drug = entry["label"].split("|")[-1].strip()
                if drug_emb.get(drug):
                    drug_emb[drug].append(mean_rep)
                else:
                    drug_emb[drug] = [mean_rep]
            else:
                num_nan_or_inf += 1
    elif EMBEDDER == "protbert":
        with open("pbert40.fa") as f:
            # entry = torch.load(f"{EMB_PATH}all.pt")
            entry = np.load(f"{EMB_PATH}all.npy")
            i = 0
            for line in f:
                if line.startswith(">"):
                    # mean_rep is not the correct term
                    # look into [0]
                    mean_rep = entry[i]
                    if not np.any(np.isnan(mean_rep)) and np.all(np.isfinite(mean_rep)):
                        drug = line.split("|")[-1].strip()
                        if drug_emb.get(drug):
                            drug_emb[drug].append(mean_rep)
                        else:
                            drug_emb[drug] = [mean_rep]
                    else:
                        num_nan_or_inf += 1
                    i += 1

    print(f"Lost {num_nan_or_inf} sequences because of Nan or Inf values in embedding")
    return drug_emb

def make_tsne(sampling_type):
    X = list()
    y = list()
    num_nan_or_inf = 0
    drug_emb = parse_embedder()
    for drug in drug_emb:
        if (
            sampling_type not in (f"greater_than_{SEQ_SUBSET}", f"{SEQ_SUBSET}_seq_subset")
            or len(drug_emb[drug]) >= SEQ_SUBSET
        ):
            if sampling_type == f"{SEQ_SUBSET}_seq_subset":
                X.extend(random.sample(drug_emb[drug], SEQ_SUBSET))
                y.extend([drug] * SEQ_SUBSET)
            else:
                X.extend(drug_emb[drug])
                y.extend([drug] * len(drug_emb[drug]))
    # breakpoint()
    sns.set(rc={"figure.figsize": (35, 25)})
    palette = sns.color_palette("hls", len(set(y)))
    X = np.asarray(X)
    y = np.asarray(y)

    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    sns_plot = sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend="full", palette=palette).set_title(
        f"coala{COALA_DATASET}_{sampling_type}_{EMBEDDER}", fontsize=20
    )
    # sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full')
    plt.legend(bbox_to_anchor=(1.005, 1.0), loc=2, borderaxespad=0.0)
    sns_plot.figure.savefig(f"test/coala{COALA_DATASET}_{sampling_type}_{EMBEDDER}_layer1.png")
    plt.clf()


def main():
    make_tsne("normal")
    make_tsne(f"greater_than_{SEQ_SUBSET}")
    make_tsne(f"{SEQ_SUBSET}_seq_subset")


if __name__ == "__main__":
    main()
