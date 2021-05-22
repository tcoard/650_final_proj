from large_list import LargeList
import numpy as np

file_name = "embeddings.pkl"
length = 0
max_emb_length = 0
with LargeList(file_name, "rb") as embs:
    for emb in embs.read():
        length += 1
        if len(emb) > max_emb_length:
            max_emb_length = len(emb)

with LargeList(file_name, "rb") as embs, open("np_emb.npy", "wb") as out:
    #my_array = np.empty((length, max_emb_length, 1024), dtype=float)
    my_array = np.empty((length, 1024), dtype=float)
    for i, emb_arr in enumerate(embs.read()):
        emb = np.ma.average(emb_arr, axis=0)
        #emb_arr.extend([[0] * (max_emb_length - len(emb_arr))] * 1024)
        #emb_arr.extend([[0] * (max_emb_length - len(emb_arr))] * 1024)
        #for ii, emb in enumerate(emb_arr):
        for ii, val in enumerate(emb):
            my_array[i][ii] = val
            # for iii, val in enumerate(emb):
            #     my_array[i][ii][iii] = val
    np.save(out, my_array)
