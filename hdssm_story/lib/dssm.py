import copy
import numpy as np
import tensorflow as tf
from importlib import reload 

assert int(tf.__version__.split(".")[0]) >= 2

import metrics
import dssm_trainer as trainer
trainer = reload(trainer)

from distances import Distances
from collections import defaultdict


class DSSM_V2:
    cfg = {
        "batch_size": 64,
        "wh_buckets": 13000,
        "neg_samples_count": 49,
        "hid_size": 300,
        "activation": "relu",
        "emb_size": 100,
        "hash_type": "poly_107"
    }

    def __init__(self, **kwargs):
        self.cfg = copy.deepcopy(DSSM_V2.cfg)
        self.cfg.update(kwargs)
        self.init_layers()

    def init_layers(self):
        self.dense_in_1 = tf.keras.layers.Dense(self.cfg["hid_size"], activation=self.cfg["activation"], dtype=tf.float64)
        # dense_in_2 = tf.keras.layers.Dense(self.cfg["hid_size"], activation=self.cfg["activation"])(dense_in_1)
        self.dense_in_3 = tf.keras.layers.Dense(self.cfg["emb_size"], dtype=tf.float64)

        self.dense_out_1 = tf.keras.layers.Dense(self.cfg["hid_size"], activation=self.cfg["activation"], dtype=tf.float64)
        # dense_out_2 = tf.keras.layers.Dense(self.cfg["hid_size"], activation=self.cfg["activation"])(dense_out_1)
        self.dense_out_3 = tf.keras.layers.Dense(self.cfg["emb_size"], dtype=tf.float64)
    
    def get_weights(self):
        return self.dense_in_1.weights + self.dense_in_3.weights + self.dense_out_1.weights + self.dense_out_3.weights 
    def call_query(self, q):
        return self.dense_in_3(self.dense_in_1(q))

    def call_doc(self, d):
        return self.dense_out_3(self.dense_out_1(d))

    def get_loss(self, q, d):
        dense_in_3 = self.call_query(q)
        dense_out_3 = self.call_doc(d)

        sm = tf.nn.softmax(dense_in_3 @ tf.transpose(dense_out_3))
        hit_prob = tf.slice(sm, [0, 0], [-1, 1])
        loss = -tf.reduce_sum(tf.math.log(hit_prob)) / q.shape[0]
        return loss

    def get_matrix_loss(self, q, d, m, distance="dot", loss="mse", symmetric=False):
        query_part = self.call_query(q)
        if not symmetric:
            doc_part = self.call_doc(d)
        else:
            doc_part = (
                query_part
                if (d is None) or (d is q) else
                self.call_query(d)
            )
    
        if isinstance(distance, str):
            dist = Distances(distance)(query_part, doc_part)
        elif hasattr(distance, "__call__"):
            if symmetric:
                assert query_part is doc_part # debug
            dist = distance(query_part, doc_part)
        else:
            raise RuntimeError("suspicious distance")

        if loss == "mse":
            delta = (m - dist)
            loss = tf.reduce_mean(delta ** 2)
        elif loss == "distortion":
            loss = metrics.distortion_loss(dist=dist, m=m)
        elif loss == "softmax":
            loss = metrics.softmax_proba_loss(dist, r=m)
        elif loss == "softmax_1/dist":
            loss = metrics.softmax_proba_inverted_dist_loss(dist, r=m)
        elif loss == "1/dist":
            loss = metrics.proba_inverted_dist_loss(dist, r=m)
        elif loss is None:
            return dist
        elif hasattr(metrics, loss + "_loss"):
            loss = getattr(metrics, loss + "_loss")(dist, m=m)
        else:
            raise NotImplementedError("Unknown loss")
        
        return loss

    def custom_hash(self, s):
        if self.cfg["hash_type"] == "python":
            return hash(s)
        elif self.cfg["hash_type"] == "poly_107":
            P = 107
            hs = 0
            for i, ch in enumerate(s):
                # assert ord(ch) < P
                hs += ord(ch) * (P ** i)
            return hs
        else:
            assert False, f"Unknown hash {self.cfg['hash_type']}"
        

    def pretokenize(self, strings):
        WH_BUCKETS = self.cfg["wh_buckets"] 
        
        mp = defaultdict(float)
        for row_i, s in enumerate(strings):
            s = s.lower()
            for word in s.split():
                word = "#" + word + "#"
                for i in range(3, len(word) + 1):
                    hs = self.custom_hash(word[i-3:i]) % WH_BUCKETS
                    mp[(row_i, hs)] += 1.

        args = list()
        values = list()
        for k, v in sorted(mp.items()):
            args.append(k)
            values.append(v)

        # return args, values
        return tf.sparse.SparseTensor(
            args,
            values,
            (len(strings), WH_BUCKETS)
        )

    def sparse_arange(self, n):
        WH_BUCKETS = self.cfg["wh_buckets"] 

        args = list()
        values = list()
        for i in range(n):
            args.append((i, i))
            values.append(1)
        return tf.sparse.SparseTensor(
            args,
            values,
            (n, WH_BUCKETS)
        )
    

class JustEmbedding(DSSM_V2):
    def __init__(self, tokens, **kwargs):
        kwargs["wh_buckets"] = tokens
        super(JustEmbedding, self).__init__(**kwargs)
        
    def init_layers(self):
        self.embedding_layer = tf.keras.layers.Dense(
            self.cfg["emb_size"],
            dtype=tf.float64,
            use_bias=False
        )
        
    def call_query(self, q):
        return self.embedding_layer(q)
    
    def call_doc(self, d):
        print("Warning: something goes wrong, you are using call_doc instead of call_query")
        return self.call_query(d)

    def get_weights(self):
        return self.embedding_layer.weights