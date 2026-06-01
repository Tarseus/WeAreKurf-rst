#!/usr/bin/env python
import os

from submission_det_strong import model as strong_det


def binary_auc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) * 0.5
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    pos_ranks = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = len(labels) - n_pos
    return (pos_ranks - n_pos * (n_pos + 1) * 0.5) / (n_pos * n_neg)


if __name__ == "__main__":
    os.environ.setdefault("DFGC_BATCH_SIZE", "1")
    det_model = strong_det.Model()
    img_names, prediction = det_model.run(
        os.path.join("DFGC_Detection", "sample_imgs"),
        os.path.join("DFGC_Detection", "sample_meta.json"),
    )
    labels = [0] * 5 + [1] * 5
    print("images:", len(img_names))
    print("predictions:", [round(float(p), 6) for p in prediction])
    print("AUC:", "%.6f" % binary_auc(labels, prediction))
