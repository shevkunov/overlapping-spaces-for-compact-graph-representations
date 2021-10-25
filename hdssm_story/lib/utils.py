import os
import networkx as nx
import numpy as np
import tensorflow as tf
import tqdm
import metrics
import collections

from matplotlib import pyplot as plt
from IPython.display import clear_output
from time import gmtime, strftime

assert int(tf.__version__.split(".")[0]) >= 2


class LRScheduler:
    def __init__(self, lr=0.1, alpha=None, gamma=None, window=10, minimum=0., maximum=1e9,
                 constant=False, constant_after=None):
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.window = window
        
        if constant:
            minimum = maximum = lr

        self.minimum = minimum
        self.maximum = maximum
        
        self.constant_after = constant_after
        if constant_after is not None:
            assert len(constant_after) == 2, "expected constant_after = (iters, lr)"
            assert type(constant_after[0]) == int
       
        self.loss_story = list()
        self.lr_story = list()
        
    def push_loss(self, x):
        self.loss_story.append(x)
        self.lr_story.append(self.lr)
        
        if (self.alpha is not None) and len(self.loss_story) > 1:
            if np.all(np.array(self.loss_story[-self.window:-1]) > self.loss_story[-1]):
                self.lr *= self.alpha
            else:
                self.lr /= self.alpha
             
        if self.gamma is not None:
            self.lr *= self.gamma
            
        if self.constant_after is not None:
            if self.constant_after[0] <= len(self.loss_story):
                self.lr = self.constant_after[1]

        self.lr = max(min(self.lr, self.maximum), self.minimum)
    
    def get_lr(self):
        return self.lr
    
def expand_to_hyperboloid(x):
    """(n, d) -> (n, d+1), s.t. x[:, 0]**2 - np.sum(x[:, 1:]**2) == 1"""
    #x64 = tf.cast(x, dtype=tf.float64)
    add = tf.expand_dims(tf.math.sqrt(1. + tf.reduce_sum(x ** 2, axis=1)), axis=1)
    #add = tf.cast(add, dtype=tf.float32)
    return tf.concat([add, x], axis=1)

def expand_to_hypersphere(v):
    cv = tf.math.cos(v)
    sv = tf.math.sin(v)

    m = 1.
    l = list()
    for i in range(v.shape[1]):
        l.append(sv[:, i] * m)  # sin_i * cos_(i-1) * ... * cos_0
        m = m * cv[:, i]
    l.append(m)  # cos_i * ... * cos_0
    ev = tf.transpose(tf.stack(l))
    if not (tf.abs(tf.norm(ev, axis=1) - 1.) < 1e-5).numpy().all():
        print("WARN: expand_to_hypersphere assert failed at e-5 prec.")
    return ev

def load_usca312(path):
    usca312 = np.eye(312) - np.ones((312, 312))
    
    with open(path) as f:
        print(f"Loading from {path}")
        for line in f:
            s, e, d = line.split()
            s = int(s)
            e = int(e)
            d = float(d)

            usca312[s, e] = d
            usca312[e, s] = d
    
    assert np.all(usca312 >= 0)
    return usca312

def get_dataset(fname="CSPhDs", distances_matrix=False, edges_matrix=False):
    if not fname.endswith(".edges"):
        fname += ".edges"
   
    path = None
    for r in ["", "datasets"]:
        for l in (["./"] +  ["../" * i for i in range(1, 5)]):
            if os.path.isdir(l + r) and (fname in os.listdir(l + r)):
                path = l + r + "/" + fname
           
    assert path is not None, "Not Found"
    
    if path.endswith("usca312.edges"):
        return None, load_usca312(path)
    
    G = nx.Graph()
    with open(path) as f:
        print(f"Loading from {path}")
        for line in f:
            s, e = map(int, line.strip().split())
            # print(s, e)
            G.add_edge(s, e)

    print(f"|V| = {len(G.nodes())}, |E| = {len(G.edges())}")
    
    R = [G]
    
    if distances_matrix or edges_matrix:
        graph_dists = dict(nx.shortest_path_length(G))

    if distances_matrix:
        R.append(
            np.array([[graph_dists[x][y] for y in G.nodes()] for x in G.nodes()])
        )
        
    if edges_matrix:
        R.append(
            1 * np.array([[graph_dists[x][y] == 1 for y in G.nodes()] for x in G.nodes()])
        )
    
    if len(R) == 1:
        return R[-1]
    else:
        return tuple(R)

def estimate_signatures_softmax(make_model, qs, distances_matrix, r_matrix, dists, iters=50, learning_rate=0.1, loss_eval_interval=5, draw_interval=20, prev_total_loss_story=None, different_d_sum=False, print_results=False, loss_name="softmax"):
    total_loss_story = list() if prev_total_loss_story is None else prev_total_loss_story
    for dist in dists:
        d = (make_model(dist.d_sum) if different_d_sum else make_model())
        loss_story = list()
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        for i in tqdm.tqdm_notebook(range(iters)):
            loss = lambda : d.get_matrix_loss(
                qs,
                None,
                r_matrix,
                distance=dist,
                loss=loss_name,
                symmetric=True
            )

            distortion = loss().numpy()
            print(f"{i}:{distortion}")
            m = opt.minimize(
                loss,
                var_list=d.get_weights() + dist.get_weights()
            )

            if (i % loss_eval_interval == 0) or (i + 1 == iters):
                pred = d.get_matrix_loss(
                    qs,
                    qs,
                    None,
                    distance=dist,
                    loss=None,
                    symmetric=True
                )
                
                loss_story.append((
                    round(distortion, 7),
                    round(metrics.mAP(pred, r_matrix, True), 7),
                    round(metrics.NDCG_graph_v1(pred, distances_matrix), 7)
                ))
            
            if (i % draw_interval == 0) or (i + 1 == iters):
                plt.figure(figsize=(10, 5))
                plt.title(f"|V| = {qs.shape[0]}")
                for i, prev_loss_story in enumerate(total_loss_story):
                    plt.plot([x[-2] for x in prev_loss_story], label=f"{dists[i]}:{prev_loss_story[-1]}")
                plt.plot([x[-2] for x in loss_story], label=f"{dist}:{loss_story[-1]}")
                plt.legend()
                plt.grid()
                clear_output()
                plt.savefig(strftime(f"./pngs/softmax-V-{qs.shape[0]}--%Y-%m-%d-%Hh.png", gmtime()))
                plt.show()
                # print(np.mean(loss_story[-20:]))
        total_loss_story.append(loss_story)

    if print_results:
        for d, l in zip(dists, total_loss_story):
            print(f"{l[-1][1]}\t{d}")

    return total_loss_story


def estimate_signatures_distortion(make_model, qs, distances_matrix, r_matrix, dists, iters=50, learning_rate=0.1, loss_eval_interval=5, draw_interval=20, prev_total_loss_story=None, calc_ranking=False, different_d_sum=False, print_results=False):
    total_loss_story = list() if prev_total_loss_story is None else prev_total_loss_story
    for dist in dists:
        d = (make_model(dist.d_sum) if different_d_sum else make_model())
        loss_story = list()
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        for i in tqdm.tqdm_notebook(range(iters)):
            loss = lambda : d.get_matrix_loss(
                qs,
                None,
                distances_matrix,
                distance=dist,
                loss="distortion",
                symmetric=True
            )

            distortion = loss().numpy()
            print(f"{i}:{distortion}")
            m = opt.minimize(
                loss,
                var_list=d.get_weights() + dist.get_weights()
            )

            if (i % loss_eval_interval == 0) or (i + 1 == iters):
                if calc_ranking:
                    pred = d.get_matrix_loss(
                        qs,
                        None,
                        None,
                        distance=dist,
                        loss=None,
                        symmetric=True
                    )
                    loss_story.append((
                        round(distortion, 7),
                        round(metrics.mAP(pred, r_matrix, True), 7),
                        round(metrics.NDCG_graph_v1(pred, distances_matrix), 7)
                    ))
                else:
                    loss_story.append((round(distortion, 7),))
            
            if (i % draw_interval == 0) or (i + 1 == iters):
                plt.figure(figsize=(10, 5))
                plt.title(f"|V| = {qs.shape[0]}")
                for i, prev_loss_story in enumerate(total_loss_story):
                    plt.plot([x[0] for x in prev_loss_story], label=f"{dists[i]}:{prev_loss_story[-1]}")
                plt.plot([x[0] for x in loss_story], label=f"{dist}:{loss_story[-1]}")
                plt.legend()
                plt.grid()
                clear_output()
                plt.savefig(strftime(f"./pngs/distortion-V-{qs.shape[0]}--%Y-%m-%d-%Hh.png", gmtime()))
                plt.show()
                # print(np.mean(loss_story[-20:]))
        total_loss_story.append(loss_story)
        
    if print_results:
        for d, l in zip(dists, total_loss_story):
            print(f"{l[-1][0]}\t{d}")

    return total_loss_story

def calc_ci_distortion(*args, **kwargs):
    total_loss_story = estimate_signatures_distortion(*args, **kwargs)
    
    dist = kwargs["dists"]
    cc = collections.defaultdict(list)
    for dist, ls  in zip(dist, total_loss_story):
        cc[str(dist)].append(ls[-1][0])

    print(f"\n\nkey\tval\tstd\truns")
    for key, val in cc.items():
        print(f"{key}\t{np.mean(val)}\t{np.std(val)}\t{len(val)}")
        
    return cc

def self_tests():
    """test for all file.
    todo: convert to pytest or smth"""
    
    ####################################
    ### expand_to_hyperboloid unittest #1 ###
    print("expand_to_hyperboloid unittest #1")
    x = expand_to_hyperboloid(
        tf.constant([
            [1, 2, 3],
            [1, -5660, -1]
        ], dtype=tf.float64
    ))
    for i in range(2):
        y = x.numpy()[i]
        doth = y[0] **2 - y[1]**2- y[2]**2- y[3]**2
        assert np.abs(doth - 1) < 1e-6, doth
