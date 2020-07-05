import numpy as np
import tensorflow as tf
import copy

assert int(tf.__version__.split(".")[0]) >= 2


def distortion_loss(dist, m):
    """dist - predicted, m - true"""
    diag = tf.eye(*dist.shape, dtype=tf.float64)
    m_with_diag = m + diag
    delta = tf.abs(dist / m_with_diag - 1)
    delta = (1 - diag) * delta  # fix diagonal
    mean_correction = np.prod(dist.shape) / (np.prod(dist.shape) - min(dist.shape))  # corrent denominator in mean
    loss = tf.reduce_mean(delta) * mean_correction
    return loss

def squared_distortion_loss(dist, m):
    """dist - predicted, m - true"""
    diag = tf.eye(*dist.shape, dtype=tf.float64)
    m_with_diag = m + diag
    delta = tf.abs((dist / m_with_diag) ** 2 - 1)
    delta = (1 - diag) * delta  # fix diagonal
    mean_correction = np.prod(dist.shape) / (np.prod(dist.shape) - min(dist.shape))  # corrent denominator in mean
    loss = tf.reduce_mean(delta) * mean_correction
    return loss

def softmax_proba_loss(d, r):
    """mean softmax-probability of relevant (r > 1) items * (-1)"""
    r = tf.cast(r, d.dtype)
    logsoftmax = tf.math.log_softmax(-d, axis=1)
    return -tf.reduce_sum(tf.math.multiply(logsoftmax, r)) / tf.reduce_sum(r)

def softmax_proba_inverted_dist_loss(d, r):
    """softmax_proba_loss with 1/dist instead of -dist coversion"""
    r = tf.cast(r, d.dtype)
    logsoftmax = tf.math.log_softmax(1. / tf.math.maximum(d, 1e-3), axis=1)
    return -tf.reduce_sum(tf.math.multiply(logsoftmax, r)) / tf.reduce_sum(r)

def proba_inverted_dist_loss(d, r):
    r = tf.cast(r, d.dtype)
    invdist = 1. / tf.math.maximum(d, 1e-3)
    inv_dist_denom = tf.reduce_sum(invdist, axis=1)
    logprob = tf.math.log(invdist / inv_dist_denom)
    return -tf.reduce_sum(tf.math.multiply(logprob, r)) / tf.reduce_sum(r)


def AP(r):
    # print(r)
    r = np.array(r)
    p_k = r.cumsum() / np.arange(1, len(r) + 1) 
    p_k = p_k * r
    # print(p_k)
    count = r.sum()
    # print(count)
    return 1. if count == 0 else np.sum(p_k) / count


def mAP(d, r, ignore_diagonal=False):
    """d - distances matrix (n, k)
    r - relevance (n, k) - 0 or 1"""
    if ignore_diagonal:
        d = np.array(d)
        mask = tf.eye(*d.shape).numpy()
        d = d + mask * (d.max() + 1)
        r = r * (1 - mask)
        
    d_sorted = np.argsort(d, axis=1)
    r = np.array(r)
    
    aps = [
        AP(r[i][d_sorted[i]])
        for i in range(d_sorted.shape[0])
    ]
    # print(aps)
    return np.mean(aps)


def DCG(r):
    arr = np.arange(len(r))
    return np.sum((2 ** (r) - 1) / np.log2(arr + 2))

def IDCG(r):
    r_best = np.sort(r)[::-1]
    return DCG(r_best)

def NDCG(r):
    return DCG(r) / IDCG(r)

def NDCG_graph_v1(true_d, pred_d):
    d_sorted = np.argsort(true_d, axis=1)
    pred_d = np.array(pred_d)
    
    return np.mean([
        NDCG(1. / (1 + pred_d[i][d_sorted[i]]))
        for i in range(d_sorted.shape[0])
    ])


def self_tests():
    """test for all file.
    todo: convert to pytest or smth"""

    #######################################
    ### mAP unittest #1 ###
    print("mAP unittest #1...")
    assert AP([1, 0, 0]) == 1
    assert AP([1, 0, 0, 1]) == (1 + 0 + 0 + 2/4) / 2
    assert AP([0, 0, 1, 1]) == (0 + 0 + 1/3 + 2/4) / 2
    assert mAP([
        [1, 2, 3, 4],
        [2, 3, 4, 1],
        [3, 4, 1, 2]
    ], [
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ]) == (1 + (1 + 0 + 0 + 2/4) / 2 + (0 + 0 + 1/3 + 2/4) / 2) / 3
