import numpy as np
import tensorflow as tf
import copy

assert int(tf.__version__.split(".")[0]) >= 2

from utils import expand_to_hyperboloid
from layers import *


class Distances:
    """Given pair of (n, d), (m, d) tensors returns pairwise distances matrix (n, m)"""
    
    def __init__(self, distance, scalable=False, opt=None, custom=None, scale_gradient_rescale=None, dtype=tf.float64):
        if not hasattr(Distances, distance):
            raise NotImplementedError("Unknown distance")
        self.dtype = dtype
        self.distance = distance
        self.scalable = scalable
        self.opt = opt
        
        self.custom = custom
        assert (distance != "custom") or (custom is not None), "You forget to specify your custom funtion"
        if custom is not None:
            assert distance == "custom"
            assert hasattr(custom, "__call__"), "custom distance isn't callable"
        
        self.weights = list()
        self.custom_weights_initialized = False
        self.scalable = scalable
        self.scale_gradient_rescale = scale_gradient_rescale
        
        if scalable:
            self.scale = tf.Variable(1, dtype=self.dtype, trainable=True)

    def get_weights(self, add_scale=True):
        weights = list()
        if self.scalable and add_scale:
            weights.append(self.scale)
        weights += self.weights
        return weights

    def __call__(self, l, r=None):
        if r is None:
            r = l
        assert len(l.shape) == len(r.shape) == 2
        assert l.shape[1] == r.shape[1]
        distance =  getattr(self, self.distance)(l, r)
        if self.scalable:
            if self.scale_gradient_rescale is not None:
                return GradientScaleLayer(self.scale_gradient_rescale)(self.scale) * distance
            else:
                return self.scale * distance
        else:
            return distance

    def dot(self, l, r):
        return l @ tf.transpose(r)
    
    def inverted_dot(self, l, r):
        return 1. - self.dot(l, r)

    def exp_minus_dot(self, l, r):
        return tf.exp(-self.dot(l, r))
    
    def inverted_softmax_dot(self, l, r):
        return 1. - tf.nn.softmax(self.dot(l, r))
    
    def inverted_dot_hyp(self, l, r):
        return - self.dot_hyp(l, r)
    
    def custom(self, l, r):
        return self.custom(l, r)
    
    def cosine(self, l, r):
        return self.dot(
            tf.nn.l2_normalize(l, 1),
            tf.nn.l2_normalize(r, 1)
        )
    
    def inverted_cosine(self, l, r):
        return -self.cosine(l, r)

    def euclidian_slow(self, l, r):
        """(x - y)^2 = x^2 + y^2 - 2xy"""
        s1 = tf.reduce_sum(l ** 2, axis=1)
        s2 = tf.reduce_sum(r ** 2, axis=1)
        s3 = -2. * l @ tf.transpose(r)
        return tf.transpose(tf.transpose(s3 + s2) + s1)

    def euclidian(self, l, r):
        """(x - y)^2 = x^2 + y^2 - 2xy"""
        s1 = tf.reduce_sum(l ** 2, axis=1)
        s2 = tf.reduce_sum(r ** 2, axis=1)
        s3 = -2. * r @ tf.transpose(l)
        return tf.math.sqrt(tf.math.maximum(tf.transpose(s3 + s1) + s2, 0.0))
    
    def euclidian_corrected(self, l, r):
        """(x - y)^2 = x^2 + y^2 - 2xy"""
        s1 = tf.reduce_sum(l ** 2, axis=1)
        s2 = tf.reduce_sum(r ** 2, axis=1)
        s3 = -2. * r @ tf.transpose(l)
        return tf.math.sqrt(tf.math.maximum(tf.transpose(s3 + s1) + s2, 0.01))

    def dot_hyp(self, l, r):
        l_add = tf.expand_dims(l[:, 0], 1)
        r_add = tf.expand_dims(r[:, 0], 0)

        return - l_add @ r_add + l[:, 1:] @ tf.transpose(r[:, 1:])

    def expanded_dot_hyp(self, l, r):
        l_exp, r_exp = map(expand_to_hyperboloid, [l, r])
        # print(l_exp, r_exp)
        return self.dot_hyp(l_exp, r_exp)

    def expanded_hyp(self, l, r):
        dot_h = self.expanded_dot_hyp(l, r)
        # assert (dot_h +1e-4 > 1).numpy().all(), dot_h.numpy().min()
        dot_h_max = tf.reduce_max(dot_h)
        # print(dot_h)
        if dot_h_max > -1. + 1e-9:
            print(f"Bad hyperbolical dot = {dot_h_max}")

        minus_dot_h = tf.math.maximum(-dot_h, 1. + 1e-4)
        # def acosh(x):
        #       return tf.math.log(x + tf.math.sqrt(x**2-1))
        return tf.math.acosh(minus_dot_h)

    def spherical(self, l, r):
        cos = self.cosine(l, r)
        mn, mx = tf.reduce_min(cos), tf.reduce_max(cos)
        if (mn < -1 - 1e-9) or (mx > 1 + 1e-9):
            print(f"Bad spherical cosine = {mn}, {mx}")
        cos = tf.math.maximum(cos, -1. + 1e-4)
        cos = tf.math.minimum(cos, 1. - 1e-4)

        return tf.math.acos(cos)
  
    def triple_trainable(self, l, r): 
        return self.tripletriple_trainable_l0(l, r)

    def triple_trainable_l0(self, l, r, corrected=False):
        if not self.custom_weights_initialized:
            self.weights.append(
                tf.Variable([0, 0, 0], dtype=self.dtype, trainable=True,
                            name="triple_trainable_distance_weights")
            )
            self.custom_weights_initialized = True
        
        normed = tf.nn.softmax(self.weights[-1])
        return (
            normed[0] * (self.euclidian if not corrected else self.euclidian_corrected)(l, r)
            + normed[1] * self.expanded_hyp(l, r)
            + normed[2] * self.spherical(l, r)
        )
    
    def triple_trainable_l0_corrected(self, l, r):
        return self.triple_trainable_l0(l, r, corrected=True)
    
    def triple_trainable_l1(self, l, r, corrected=False, normed=True):
        assert l.shape[1] % 2 == 0
        
        if not self.custom_weights_initialized:
            self.weights.append(
                tf.Variable(np.zeros(9), dtype=self.dtype, trainable=True,
                            name="triple_trainable_distance_weights")
            )
            self.custom_weights_initialized = True
        
        def call(l, r, fr, to, weights):
            l_slice = l[:, fr:to]
            r_slice = r[:, fr:to]
            return (
                weights[0] * (self.euclidian if not corrected else self.euclidian_corrected)(l_slice, r_slice)
                + weights[1] * self.expanded_hyp(l_slice, r_slice)
                + weights[2] * self.spherical(l_slice, r_slice)
            )    
            
        normed = tf.nn.softmax(self.weights[-1]) if normed else tf.math.abs(self.weights[-1])
        fr, ce, to = 0, int(l.shape[1] / 2), l.shape[1]
        return (
            call(l, r, fr, to, normed[0:3])
            + call(l, r, fr, ce, normed[3:6])
            + call(l, r, ce, to, normed[6:9])
        )
    
    def triple_trainable_l1_sq(self, l, r, corrected=False):
        assert l.shape[1] % 2 == 0
        
        if not self.custom_weights_initialized:
            self.weights.append(
                tf.Variable(np.zeros(8), dtype=self.dtype, trainable=True,
                            name="triple_trainable_l1_sq_distance_weights")
            )
            self.custom_weights_initialized = True
        
        normed = tf.nn.softmax(self.weights[-1])
        fr, ce, to = 0, int(l.shape[1] / 2), l.shape[1]
        euclidian = self.euclidian if not corrected else self.euclidian_corrected
        return tf.math.sqrt(
            normed[0] * self.expanded_hyp(l, r) ** 2
            + normed[1] * self.spherical(l, r) ** 2
            
            + normed[2] * self.expanded_hyp(l[:, fr:ce], r[:, fr:ce]) ** 2
            + normed[3] * self.spherical(l[:, fr:ce], r[:, fr:ce]) ** 2
            + normed[4] * euclidian(l[:, fr:ce], r[:, fr:ce]) ** 2
            
            + normed[5] * self.expanded_hyp(l[:, ce:to], r[:, ce:to]) ** 2
            + normed[6] * self.spherical(l[:, ce:to], r[:, ce:to]) ** 2
            + normed[7] * euclidian(l[:, ce:to], r[:, ce:to]) ** 2
        )
    
    def triple_trainable_l1_corrected(self, l, r):
        return self.triple_trainable_l1(l, r, corrected=True)
    
    def triple_trainable_l1_sq_corrected(self, l, r):
        return self.triple_trainable_l1_sq(l, r, corrected=True)
    
    def triple_trainable_l1_not_normed(self, l, r):
        return self.triple_trainable_l1(l, r, normed=False)

    def symmetric_riemann_spherical(self, l, r):
        assert self.opt is not None
        assert tf.reduce_mean((r - l) ** 2) < 1e-9  #  just one gradient flow, but in ProductDist was copy, so we cannot use r is l

        # l = tf.nn.l2_normalize(l, 1)
        # r = tf.nn.l2_normalize(r, 1)

        l = SphericalExponentialMapLayer(self.opt)(l)

        cos = self.cosine(l, l)  # todo: fix to dot and assert
        mn, mx = tf.reduce_min(cos), tf.reduce_max(cos)
        if (mn < -1 - 1e-9) or (mx > 1 + 1e-9):
            print(f"Bad spherical cosine = {mn}, {mx}")
        cos = tf.math.maximum(cos, -1. + 1e-4)
        cos = tf.math.minimum(cos, 1. - 1e-4)

        return tf.math.acos(cos)

    def hyperbolical(self, l, r):
        def on_hyperboloid(x):
            dot_h = - x[:, 0] * x[:, 0] + tf.reduce_sum(x[:, 1:] * x[:, 1:], axis=1)
            return tf.reduce_mean((dot_h + 1) ** 2) < 1e-9
        
        if not on_hyperboloid(l):
            print("l is not on hyperboloid, pinning...")
            l = expand_to_hyperboloid(l[:, 1:])
        if not on_hyperboloid(r):
            print("r is not on hyperboloid, pinning...")
            r = expand_to_hyperboloid(r[:, 1:])

        dot_h = self.dot_hyp(l, r)
        dot_h_max = tf.reduce_max(dot_h)

        if dot_h_max > -1. + 1e-9:
            print(f"Bad hyperbolical dot = {dot_h_max}")

        minus_dot_h = tf.math.maximum(-dot_h, 1. + 1e-4)
        return tf.math.acosh(minus_dot_h)
    
    def hyperbolical_2(self, l, r):
        el = ExpandToHyperboloidLayer()
        # el = expand_to_hyperboloid
        l, r = el(l), el(r)
        dot_h = self.dot_hyp(l, r)
        dot_h_max = tf.reduce_max(dot_h)

        if dot_h_max > -1. + 1e-9:
            print(f"Bad hyperbolical dot = {dot_h_max}")

        minus_dot_h = tf.math.maximum(-dot_h, 1. + 1e-4)
        return tf.math.acosh(minus_dot_h)

    def symmetric_riemann_hyperbolical(self, l, r):
        assert self.opt is not None
        assert tf.reduce_mean((r - l) ** 2) < 1e-9 
        #  just one gradient flow, but in ProductDist was copy, so we cannot use r is l

        # we didn't pin to hyperboloid

        l = HyperbolicalExponentialMapLayer(self.opt)(l)

        return self.hyperbolical(l, l)
    
    def symmetric_riemann_hyperbolical_expanded(self, l, r):
        # aka symmetric_riemann_hyperbolical2
        assert self.opt is not None
        assert tf.reduce_mean((r - l) ** 2) < 1e-9 
        #  just one gradient flow, but in ProductDist was copy, so we cannot use r is l

        # we didn't pin to hyperboloid

        l = ExpandToHyperboloidLayer()(l)
        # l = expand_to_hyperboloid(l)
        l = HyperbolicalExponentialMapLayer(self.opt)(l)

        return self.hyperbolical(l, l)
    
    def __str__(self):
        return self.distance

class DistancesAggregator:
    def __init__(self, kind="l2"):
        if kind == "l2":
            self.convert_input = lambda d: d ** 2
            self.convert_output = lambda d: tf.math.sqrt(tf.math.maximum(d, 0.0))
            self.agg = lambda x, y: x + y
        elif kind == "sum":
            self.convert_input = self.convert_output = lambda d: d
            self.agg = lambda x, y: x + y
        else:
            raise NotImplementedError(f"Unknown aggregation kind: '{kind}'.")
        
        for x in ["convert_input", "convert_output", "agg"]:
            assert hasattr(self, x), "Selfcheck failure: no '{x}' method. What have you done?"
            method = getattr(self, x)
            assert hasattr(method, "__call__"), "Selfcheck failure: '{x}' isn't callable. What have you done?"
        
        self.kind = kind
        self.dag = None  # distanges aggregated
        
    def add(self, d):
        if self.dag is None:
            self.dag = self.convert_input(d)
        else:
            assert self.dag.shape == d.shape
            self.dag = self.agg(
                self.dag, 
                self.convert_input(d)
            )
    
    def get(self):
        result = self.dag
        self.clear()
        return self.convert_output(result)       
           
    def clear(self):
        self.dag = None


class ProductDistances:
    def __init__(self, distances, aggregation_kind="l2"):
        """Given [(Distances(...), d1), (Distances(...), d2), ...]
        or [(d1, Distances(...)), (d2, Distances(...)), ...]
        produces d(x, y) = sqrt(d_1(x[:, 0:d1]) ** 2 + d_2(x[:, d1:d1 + d2]) ** 2 + ...) 
        if aggregation_kind is "l2". See DistancesAgreggator for more aggregations, such as l1."""

        self.distances = list()
        self.d_sum = 0
        self.aggregation_kind = aggregation_kind
        self.dag = DistancesAggregator(aggregation_kind)
        
        for dist, d_i in distances:
            if isinstance(d_i, Distances):
                dist, d_i = d_i, dist  # eXtremly important functionality
            self.distances.append((dist, d_i))

            assert d_i > 0
            assert isinstance(dist, Distances)
            self.d_sum += d_i
    
    def __call__(self, l, r):
        assert len(l.shape) == len(r.shape) == 2
        assert l.shape[1] == r.shape[1] == self.d_sum

        d_last = 0
        self.dag.clear()
        for dist, d_i in self.distances:
            current_dist = dist(
                l[:, d_last:d_last + d_i],
                r[:, d_last:d_last + d_i]
            )
            
            self.dag.add(current_dist)
            d_last += d_i
            
        return self.dag.get()

    def get_weights(self):
        weights = list()
        for dist, _ in self.distances: 
            weights += dist.get_weights()
        return weights

    def __repr__(self):
        return "ProductDistances(" + str(self) + ")"

    def __str__(self):
        stringify_map = {
            "euclidian": "E",
            "spherical": "S",
            "expanded_hyp": "H",
            "symmetric_riemann_spherical": "Sr",
            "symmetric_riemann_hyperbolical_expanded": "Hr",
        }
        signatute = list()
        for dist, d_i in self.distances:
            if dist.distance not in stringify_map:
                # print("WARNING: Can't find string repr for '{dist.distance}' distance. You may add it.")
                if (dist.distance != "custom") or (not hasattr(dist.custom, "__str__")):
                    signatute.append(f"{str(dist.distance)}{d_i}")
                else:
                    signatute.append(f"{str(dist.custom)}{d_i}")
            else:
                signatute.append(f"{stringify_map[dist.distance]}{d_i}")
                
        signatute = ",".join(signatute)
        
        if self.aggregation_kind != "l2":
            signatute += f" ({self.aggregation_kind})"
        
        return signatute
    
    def fix_weights(self, w):
        """fix constraint on weights (n, d)"""
        w_i_list = list()
        d_last = 0
        for dist, d_i in self.distances:
            w_i = w[:, d_last:d_last + d_i]
            
            n = dist.distance 
            if n == "euclidian":
                pass  # R_n already in R_n
            elif n == "spherical":
                pass  # there is normalization in distance
            elif n in {"expanded_hyp", "symmetric_riemann_hyperbolical_expanded"}: 
                pass  # there is projection R_n -> H_{n+1}
            elif n == "symmetric_riemann_spherical":
                w_i = tf.nn.l2_normalize(w_i, 1)
                          
            w_i_list.append(w_i)
            d_last += d_i
        return tf.concat(w_i_list, axis=1)         
    

def self_tests():
    """test for all file.
    todo: convert to pytest or smth"""

    #######################################
    ### DistancesAggregator unittest #1 ###
    print("DistancesAggregator unittest #1...")
    dag = DistancesAggregator("sum")
    l = list()
    for _ in range(4):
        l.append(np.random.randn(4, 4) + 10)
        dag.add(l[-1])
    assert np.mean((dag.get() - sum(l)) ** 2) < 1e-9
    
    #######################################
    ### DistancesAggregator unittest #2 ###
    print("DistancesAggregator unittest #2...")
    dag = DistancesAggregator("l2")
    l = list()
    for _ in range(4):
        l.append(np.random.randn(4, 4) + 10)
        dag.add(l[-1])
    true_sol = np.sqrt(sum([x ** 2 for x in l]))
    assert np.mean((dag.get() - true_sol) ** 2) < 1e-9

    ####################################
    ### ProductDistances unittest #1 ###
    print("ProductDistances unittest #1...")
    a = tf.random.normal((1000, 5))
    b = tf.random.normal((1000, 5))
    
    mse = lambda x: np.mean(x ** 2)

    for kind in ["euclidian", "spherical", "expanded_hyp"]:
        pd = ProductDistances([
            (Distances(kind), 5)
        ])
        assert mse(pd(a, b) - Distances(kind)(a, b)) < 1e-9, mse(pd(a, b) - Distances(kind)(a, b))
        

    ####################################
    ### ProductDistances unittest #2 ###
    print("ProductDistances unittest #2...")
    pd = ProductDistances([
        (Distances("euclidian"), 2),
        (Distances("spherical"), 3),
    ])

    d1 = Distances("euclidian")(a[:, :2], b[:, :2])
    d2 = Distances("spherical")(a[:, 2:], b[:, 2:])
    assert mse(pd(a, b) - (d1 ** 2 + d2 ** 2) ** 0.5) < 1e-9
    
    
    ####################################
    ### ProductDistances unittest #3 ###
    print("ProductDistances unittest #3...")
    a = tf.random.normal((1000, 8))
    b = tf.random.normal((1000, 8))

    pd = ProductDistances([
        (2, Distances("euclidian")),
        (3, Distances("spherical")),
        (3, Distances("expanded_hyp")),
    ])

    d1 = Distances("euclidian")(a[:, :2], b[:, :2])
    d2 = Distances("spherical")(a[:, 2:5], b[:, 2:5])
    d3 = Distances("expanded_hyp")(a[:, 5:8], b[:, 5:8])
    assert mse(pd(a, b) - (d1 ** 2 + d2 ** 2 + d3 ** 2) ** 0.5) < 1e-9
    
    
    ####################################
    ### ProductDistances unittest #4 ###
    print("ProductDistances unittest #4...")
    pd = ProductDistances([
        (2, Distances("euclidian", scalable=True)),
        (3, Distances("spherical")),
        (3, Distances("expanded_hyp", scalable=True)),
    ])
    assert len(pd.get_weights()) == 2
    print(f"str(pd) = {pd}")