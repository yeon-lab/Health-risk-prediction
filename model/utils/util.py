import numpy as np

def padding(seqs, input_dim):
    lengths = np.array([len(seq) for seq in seqs]).astype("int32")
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros([maxlen, n_samples, input_dim]).astype(np.float32)
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:, idx, :], seq):
            xvec[subseq] = 1.
    return x, lengths
