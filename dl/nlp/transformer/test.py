import numpy as np


def generate_src_mask(src, src_pad_idx):
    # src_mask = (src != src_pad_idx).astype(int)
    return np.where(src != src_pad_idx, 0, -1e20).astype(int)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


if __name__ == "__main__":
    np.random.seed(0)
    src = np.random.randint(0, 8, (3, 10))
    src_mask = generate_src_mask(src, 0)
    print("src:", "\n", src)
    print("src + src_mask:", "\n", src + src_mask)
    print("src_mask:", "\n", src_mask)
    print("-----------------")
    print("softmax(src):", "\n", softmax(src))
    print("softmax(src + src_mask):", "\n", softmax(src + src_mask))
