import collections
import numpy as np
import tensorflow as tf
import tqdm

import matplotlib.pyplot as plt

from IPython.display import clear_output

assert int(tf.__version__.split(".")[0]) >= 2


class StringPretokenizer:
    def __init__(self, total_tokens, triplets_weight=20, pairs_weight=2, words_weight=8, hash_type="poly_107"):
        self.fitted = False
        total_weight = int(triplets_weight + pairs_weight + words_weight)
        assert total_weight > 0 
        self.total_tokens = int(total_tokens)
        self.triplets = int(total_tokens * triplets_weight) // total_weight
        self.pairs = int(total_tokens * pairs_weight) // total_weight
        self.words = self.total_tokens - self.triplets - self.pairs
        self.hash_type = hash_type

        if self.hash_type == "python":
            self.custom_hash = hash
        elif self.hash_type == "poly_107":
            def poly_107(s):
                P = 107
                hs = 0
                for i, ch in enumerate(s):
                    # assert ord(ch) < P
                    hs += ord(ch) * (P ** i)
                return hs
            self.custom_hash = poly_107
        else:
            assert False, f"Unknown hash {self.cfg['hash_type']}"        
        
    def fit(self, rows):
        if self.fitted:
            print("WARN:String pretokenizer already fitted, ignoring...")
            return

        tokens = collections.Counter()
        for name in tqdm.tqdm(rows):
            if isinstance(name, str):
                name = name.lower().replace("-", " ")
                for t in name.split(' '):
                    if t:
                        tokens[t] += 1

        self.word_tokens = dict()
        for i, (t, c) in enumerate(tokens.most_common(self.words)):
            self.word_tokens[t] = i
        self.fitted = True
        
    def __call__(self, *args, **kwargs):
        return self.pretokenize(*args, **kwargs)
        
    def fit_pretokenize(self, rows, *args, **kwargs):
        self.fit(rows)
        return self.pretokenize(rows, *args, **kwargs)

    def from_raw(self, batch):
        args = list()
        values = list()
            
        for row_i, row in enumerate(batch):
            for hs, v in row:
                k = (row_i, hs)
                args.append(k)
                values.append(v)
                
        return tf.sparse.SparseTensor(
            args,
            values,
            (len(batch), self.total_tokens)
        )
        
                
    def pretokenize(self, rows, raw=False):
        assert self.fitted
        
        mp = collections.defaultdict(float)
        for row_i, s in enumerate(rows):
            if not isinstance(s, str):
                continue
            s = s.lower()
            for word in s.split():
                if word in self.word_tokens:
                    hs = self.word_tokens[word]
                    # print(word, "->", hs)
                    mp[(row_i, hs + self.triplets + self.pairs)] += 1.

                word = "#" + word + "#"
                if self.triplets:
                    for i in range(3, len(word) + 1):
                        hs = self.custom_hash(word[i-3:i]) % self.triplets
                        # print(word[i-3:i], "->", hs)
                        mp[(row_i, hs)] += 1.

                if self.pairs:
                    for i in range(2, len(word) + 1):
                        hs = self.custom_hash(word[i-2:i]) % self.pairs
                        # print(word[i-2:i], "->", hs)
                        mp[(row_i, hs + self.triplets)] += 1.
                    
        if raw:
            raw_result = collections.defaultdict(list)
            for k, v in sorted(mp.items()):
                row_i, hs = k
                raw_result[row_i].append((hs, v))
            return raw_result

        args = list()
        values = list()
        for k, v in sorted(mp.items()):
            args.append(k)
            values.append(v)

        return tf.sparse.SparseTensor(
            args,
            values,
            (len(rows), self.total_tokens)
        )
    

class DssmTrainerWithCustomTokenizer:
    def __init__(self, model, qs, train_size=0.9, val_offset=1000, batch_size=4096,
                 doc_tokenizer=None, query_tokenizer=None,
                 tokenizer_cfg={"total_tokens":30000}):
        self.train_data = qs
        self.batch_size = batch_size
        self.val_offset = val_offset
        
        self.X_test = self.train_data.values[int(self.train_data.shape[0] * train_size):]
        self.X_train = self.train_data.values[:int(self.train_data.shape[0] * train_size)]
        
        self.doc_tokenizer = StringPretokenizer(**tokenizer_cfg) if doc_tokenizer is None else doc_tokenizer
        self.query_tokenizer = StringPretokenizer(**tokenizer_cfg) if query_tokenizer is None else query_tokenizer
        
        self.train_query_tokens = self.query_tokenizer.fit_pretokenize(self.X_train[:, 0], raw=True)
        self.train_doc_tokens = self.doc_tokenizer.fit_pretokenize(self.X_train[:, 1], raw=True)
        
        self.all_doc_unique = np.unique(self.train_data.values[:, 1])
        self.all_doc_unique_tokens = self.doc_tokenizer(self.all_doc_unique)
        self.val_query_tokens = self.query_tokenizer(self.X_test[:val_offset, 0])
        self.test_query_tokens = self.query_tokenizer(self.X_test[:, 0])
                
        self.best_val_acc = -1
        self.best_model_weights = None
        self.best_dist_weights = None
        
    def get_batch(self, model, self_check=False):
        sub_set = sorted(np.random.choice(self.X_train.shape[0], size=self.batch_size, replace=False))
        
        d_t = self.doc_tokenizer.from_raw([self.train_doc_tokens[i] for i in sub_set])
        q_t = self.query_tokenizer.from_raw([self.train_query_tokens[i] for i in sub_set])

        if self_check:
            sub_set = self.X_train[sub_set]

            q_t_ = self.query_tokenizer(sub_set[:,0])
            d_t_ = self.doc_tokenizer(sub_set[:,1])
            
            d_t__, q_t__, d_t_, q_t_ = map(tf.sparse.to_dense, [d_t, q_t, d_t_, q_t_])
                                       
            assert np.sum((q_t__ - q_t_) ** 2) < 1e-9
            assert np.sum((d_t__ - d_t_) ** 2) < 1e-9 
        
        return q_t, d_t
    
    def get_loss(self, model, dist):
        if hasattr(model, "train"):
            model.train = True
        q_t, d_t = self.get_batch(model)
        return model.get_matrix_loss(
            q_t,
            d_t,
            np.eye(self.batch_size),
            distance=dist,
            loss="softmax",
            symmetric=False
        )
    
    def _create_metrics(self):
        return {
            "hits": 0,
            "total": 0,
        }
 
    def _update_metrics(self, metrics, distances, y, just_acc=False):
        metrics["total"] += 1
        
        if just_acc:           
            best = tf.argmin(distances)
        else:
            d_argsort = tf.argsort(distances)
            best = d_argsort[0]
            
            if "map100" not in metrics:
                metrics["map100"] = 0.
                
            for i in range(100):
                if self.all_doc_unique[d_argsort[i]] == y:
                    metrics["map100"] += 1. / (i + 1.)
                    break
            
        if self.all_doc_unique[best] == y:
            metrics["hits"] += 1
            
            
    def _finalize_metrics(self, metrics):
        metrics["acc"] = float(metrics["hits"]) / metrics["total"]
        if "map100" in metrics:
            metrics["map100"] =  metrics["map100"] / metrics["total"]
                
    def get_val_acc(self, model, dist, update_best=False, return_metrics=False):
        if hasattr(model, "train"):
            model.train = False
            
        all_emb = model.call_doc(self.all_doc_unique_tokens)
        q_embs = model.call_query(self.val_query_tokens)
        
        metrics = self._create_metrics()
        for q_emb, y in tqdm.tqdm_notebook(zip(q_embs, self.X_test[:self.val_offset, 1]), total=self.val_offset):
            q_emb_r = tf.reshape(q_emb, (1, -1))
            d = dist(q_emb_r, all_emb)[0]
            self._update_metrics(metrics, d, y, just_acc=(not return_metrics))
         
        self._finalize_metrics(metrics)
        acc = metrics["acc"]
        
        if update_best and (acc > self.best_val_acc):
            self.best_val_acc = acc
            self.best_model_weights = [w.numpy() for w in model.get_weights()]
            self.best_dist_weights = [w.numpy() for w in dist.get_weights()]

        return acc if (not return_metrics) else metrics

    def get_train_acc(self, model, dist, cut=10000, return_metrics=False):
        if hasattr(model, "train"):
            model.train = False

        all_emb = model.call_doc(self.all_doc_unique_tokens)
        q_t = self.query_tokenizer.from_raw([self.train_query_tokens[i] for i in range(cut)])
        q_embs = model.call_query(q_t)
        
        metrics = self._create_metrics()
        for q_emb, y in tqdm.tqdm_notebook(zip(q_embs, self.X_train[:cut, 1]), total=cut):
            q_emb_r = tf.reshape(q_emb, (1, -1))
            d = dist(q_emb_r, all_emb)[0]
            self._update_metrics(metrics, d, y)
            
        self._finalize_metrics(metrics)
        return metrics["acc"] if (not return_metrics) else metrics
    
    def get_test_acc(self, model, dist, return_metrics=False, bug_fixed=False):
        if hasattr(model, "train"):
            model.train = False
        all_emb = model.call_doc(self.all_doc_unique_tokens)
        q_embs = model.call_query(self.test_query_tokens)
        
        metrics = self._create_metrics()
        cut = 0 if (not bug_fixed) else self.val_offset
        for q_emb, y in tqdm.tqdm_notebook(zip(q_embs[cut:], self.X_test[cut:, 1]), total=self.X_test.shape[0] - cut):
            q_emb_r = tf.reshape(q_emb, (1, -1))
            d = dist(q_emb_r, all_emb)[0]
            self._update_metrics(metrics, d, y)
            
        self._finalize_metrics(metrics)
        return metrics["acc"] if (not return_metrics) else metrics
        
    def fit(self, model, dist, opt_train=None,
            loss_story=None, val_loss_story=None,
            iters=10000, redraw_interval=100):
        self.opt_train = (
            tf.keras.optimizers.Adam()
            if (opt_train is None) and (not hasattr(self, "opt_train")) else
            opt_train
        )
            
        self.loss_story = (
            list()
            if (loss_story is None) and (not hasattr(self, "loss_story")) else
            loss_story
        )
        
        self.val_loss_story = (
            list()
            if (val_loss_story is None) and (not hasattr(self, "val_loss_story")) else
            val_loss_story
        )
            
        def loss_train():
            while True:
                try:
                    l = self.get_loss(model, dist)
                    self.loss_story.append(l.numpy())
                    print(self.loss_story[-1])
                    return l
                except Exception as e:
                    print(f"E:{e}")
                    pass

        for iteration in tqdm.tqdm_notebook(range(iters)):
            m = self.opt_train.minimize(
                loss_train,
                var_list=model.get_weights() + dist.get_weights()
            )

            if iteration + 1 == iters or iteration % redraw_interval == 0:
                self.val_loss_story.append(
                    self.get_val_acc(model, dist, update_best=True)
                )
                plt.figure(figsize=(14, 7))
                plt.subplot(211)
                plt.plot(np.arange(len(self.loss_story)), self.loss_story,
                         "-o", label="loss", color="blue")
                plt.legend()
                plt.subplot(212)
                plt.plot(np.arange(len(self.val_loss_story)) * redraw_interval, self.val_loss_story,
                         "-o", label="val_acc", color="red")
                plt.legend()
                clear_output(wait=True)
                plt.show()
                print(f"val_acc[-1] = {self.val_loss_story[-1]}")
                
                
class DssmTrainerWithCustomTokenizerAndHards(DssmTrainerWithCustomTokenizer):
    def __init__(self, hard_neg_k=16, hard_every=10, *args, **kwargs):
        super(DssmTrainerWithCustomTokenizerAndHards, self).__init__(*args, **kwargs)
        self.hard_neg_k = hard_neg_k
        self.hard_every = hard_every
        self.batch_iter = 0
        self.train_doc_tokens_tf = self.doc_tokenizer(self.X_train[:, 1])

    def get_batch(self, model, dist, self_check=False):
        self.batch_iter += 1
        if self.batch_iter % self.hard_every:
            return super(DssmTrainerWithCustomTokenizerAndHards, self).get_batch(model, self_check=self_check)
        print("HardBatch!")
        sub_set = sorted(np.random.choice(self.X_train.shape[0], size=self.batch_size // self.hard_neg_k,
                                          replace=False))
        # print(sub_set)
        # self.doc_tokenizer.from_raw([self.train_doc_tokens[i] for i in range(self.X_train.shape[0])])
        q_t = self.query_tokenizer.from_raw([self.train_query_tokens[i] for i in sub_set])
        
        train_d_emb = model.call_doc(self.train_doc_tokens_tf)
        all_q_emb = model.call_query(q_t)
        
        for q_emb in all_q_emb:
            q_emb_r = tf.reshape(q_emb, (1, -1))
            distances = dist(q_emb_r, train_d_emb)
            # print(distances)
            top = tf.argsort(distances[0])[:self.hard_neg_k - 1]
            # print(top)
            for cand in top.numpy():
                sub_set.append(cand)
                
        sub_set = np.unique(sub_set)
        # print(len(sub_set))
        
        d_t = self.doc_tokenizer.from_raw([self.train_doc_tokens[i] for i in sub_set])
        q_t = self.query_tokenizer.from_raw([self.train_query_tokens[i] for i in sub_set])
        
        if self_check:
            sub_set = self.X_train[sub_set]

            q_t_ = self.query_tokenizer(sub_set[:,0])
            d_t_ = self.doc_tokenizer(sub_set[:,1])
            
            d_t__, q_t__, d_t_, q_t_ = map(tf.sparse.to_dense, [d_t, q_t, d_t_, q_t_])
                                       
            assert np.sum((q_t__ - q_t_) ** 2) < 1e-9
            assert np.sum((d_t__ - d_t_) ** 2) < 1e-9 
        
        return q_t, d_t
    
    def get_loss(self, model, dist):
        if hasattr(model, "train"):
            model.train = True
        q_t, d_t = self.get_batch(model, dist)
        return model.get_matrix_loss(
            q_t,
            d_t,
            np.eye(q_t.shape[0]),
            distance=dist,
            loss="softmax",
            symmetric=False
        )