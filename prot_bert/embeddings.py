import gc
import pickle
import objgraph
import resource
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm
from large_list import LargeList

embeddings_file = "embeddings_100.pkl"

def main():
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert")
    fe = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=0)

    chunk_size = 100
    curr_chunk = list()
    embeddings = list()
    with open("/home/tcoard/w/650_proj/data/pbert100.fa", "r") as f, LargeList(embeddings_file, "wb") as emb_f:
        for i, line in enumerate(f):
            if ">" not in line:
                seq = line.strip()
                embedding = fe(" ".join(seq)) # this can do many at a time, but I don't want to write the chunk logic
                if len(embedding) > 1:
                    breakpoint()
                seq_emd = embedding[0][1:len(seq) + 1]
                emb_f.write(seq_emd)


if __name__ == "__main__":
    main()
