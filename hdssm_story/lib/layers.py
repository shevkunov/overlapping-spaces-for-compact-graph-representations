import numpy as np
import tensorflow as tf

assert int(tf.__version__.split(".")[0]) >= 2

from utils import expand_to_hyperboloid


class GradientScaleLayer:
    def __init__(self, scale=1.):
        self.scale = scale
        
    def __call__(self, x):
        def backward(dy):
            return self.scale * dy
        
        @tf.custom_gradient
        def forward(x):
            return x, backward
        
        return forward(x)


class ExpandToHyperboloidLayer:
    """(n, d) -> (n, d+1), s.t. x[:, 0]**2 - np.sum(x[:, 1:]**2) == 1"""
    def __call__(self, x):
        def cut_gradient(dy):
            return dy[:, 1:]
        
        @tf.custom_gradient
        def expand(x):
            add = tf.expand_dims(tf.math.sqrt(1. + tf.reduce_sum(x ** 2, axis=1)), axis=1)
            return tf.concat([add, x], axis=1), cut_gradient
    
        return expand(x)


class SphericalExponentialMapLayer:
    def __init__(self, opt):
        self.opt = opt
        # assert (opt is None) or isinstance(opt, tf.keras.optimizers.SGD), "Only native SGD is supported"
        if not ((opt is None) or isinstance(opt, tf.keras.optimizers.SGD)):
            print("SphericalExponentialMapLayer: WARNING! Only native SGD is supported, be carefull.")

    def __call__(self, p):
        @tf.custom_gradient
        def spherical_exponental_map_layer(p):
            """https://arxiv.org/pdf/1805.08207.pdf, page 3
            given (n, d) vector return itself on forward pass
            and applies exponential map on backward"""

            def project_onto_tangent_space(v, p):
                return v - tf.expand_dims(tf.reduce_sum(v * p, axis=1), 1) * p

            def exponential_map(v, p):
                n = tf.expand_dims(tf.norm(v, axis=1), 1)
                return tf.where(
                    n > 1e-7,  
                    tf.math.cos(n) * p + tf.math.sin(n) * v / n,
                    p  # sin(x) / x ~ 1 and v -> 0 at n -> 0, so ^ -> p
                )

            def grad(dp):
                """dp aka v in article"""
                opt = self.opt
                learing_rate = (opt.learning_rate if opt is not None else 1.)
                dp = - dp * learing_rate
                dp = project_onto_tangent_space(dp, p)
                new_p = exponential_map(dp, p)
                #print(new_p - p)
                return (p - new_p) / learing_rate

            return p, grad

        return spherical_exponental_map_layer(p)

                  
def dot_hyp(x, y):
    return - (x[:, 0] * y[:, 0]) + tf.reduce_sum(x[:, 1:] * y[:, 1:], axis=1)

                  
class HyperbolicalExponentialMapLayer:
    def __init__(self, opt):
        self.opt = opt
        # assert (opt is None) or isinstance(opt, tf.keras.optimizers.SGD), "Only native SGD is supported"
        if not ((opt is None) or isinstance(opt, tf.keras.optimizers.SGD)):
            print("HyperbolicalExponentialMapLayer: WARNING! Only native SGD is supported, be carefull.")
        
    def __call__(self, p):
        @tf.custom_gradient
        def hyperbolical_exponental_map_layer(p):
            """https://arxiv.org/pdf/1805.08207.pdf, page 4
            given (n, d) vector return itself on forward pass
            and applies exponential map on backward"""

            def project_onto_tangent_space(v, p):
                v = tf.concat([tf.expand_dims(-v[:, 0], 1), v[:, 1:]], axis=1)
                return v - tf.expand_dims(dot_hyp(v, p) / dot_hyp(p, p), 1) * p
                # spherical case, FUI
                # return v - tf.expand_dims(tf.reduce_sum(v * p, axis=1), 1) * p

            def exponential_map(v, p):
                n = dot_hyp(v, v)
                n = tf.math.maximum(n, 0)
                n = tf.math.sqrt(n)
                n = tf.math.minimum(n, 1)  # for stability (see ProductSpaces original code)
                # print("n = ", n)
                n = tf.expand_dims(n, 1)

                return tf.where(
                    n > 1e-7,  
                    tf.math.cosh(n) * p + tf.math.sinh(n) * v / n,
                    p  # sin(x) / x ~ 1 and v -> 0 at n -> 0, so ^ -> p
                )

            def grad(dp):
                # print(dp)
                """dp aka v in article"""
                opt = self.opt
                learing_rate = (opt.learning_rate if opt is not None else 1.)
                dp = - dp * learing_rate
                dp = project_onto_tangent_space(dp, p)
                # print("Projection: ", dp)
                new_p = exponential_map(dp, p)
                # print("new_p = ", new_p)
                new_p = expand_to_hyperboloid(new_p[:, 1:])  # fix after trick for stability
                # print("self_hyp_dot(new_p) = ", dot_hyp(new_p, new_p))
                #print(new_p - p)
                # assert np.mean((dot_hyp(new_p, new_p) + 1) ** 2) < 1e-9, "not on hyperboloid :("
                return (p - new_p) / learing_rate

            return p, grad

        return hyperbolical_exponental_map_layer(p)
    
    
def self_tests():
    #########################################
    ### SphericalExponentialMapLayer UT 1 ###
    print("SphericalExponentialMapLayer UT 1")
    
    v = tf.nn.l2_normalize(tf.Variable([[1., 1., 1.], [1, 0, 0], [4, 5, 2], [-5, 1, 0]], dtype=tf.float64), axis=1)
    with tf.GradientTape() as g:
        g.watch(v)
        loss = tf.reduce_mean((SphericalExponentialMapLayer(None)(v) - [1, 0, 0]) ** 2)

    gg = g.gradient(loss, v)
    tf.norm(v - gg, axis=1)  # minimization
    assert np.mean((tf.norm(v - gg, axis=1) - 1) ** 2)< 1e-9
    
    
    #########################################
    ### SphericalExponentialMapLayer UT 2 ###
    print("SphericalExponentialMapLayer UT 2")
    
    v0 = tf.Variable(
        tf.nn.l2_normalize([[1., 1., 1.], [1, 0, 0.5], [4, 5, 2], [-5, 1, 0]], axis=1).numpy(),
        dtype=tf.float64
    )

    opt = tf.keras.optimizers.SGD(learning_rate=tf.Variable(10., dtype=tf.float64))

    def get_loss():
        v2 = SphericalExponentialMapLayer(opt)(v0)
        c = tf.constant([1, 0, 0], dtype=tf.float64)
        # v1 = tf.nn.l2_normalize(v0, axis=1)
        loss = tf.reduce_mean((v2 - c) ** 2)
        return loss

    loss_story = list()

    for _ in range(10):
        v0.assign(v0 / tf.expand_dims(tf.norm(v0, axis=1), 1))
        opt.minimize(get_loss, v0)
        loss_story.append(get_loss())
        # print(loss_story[-1], tf.norm(v0, axis=1))
        assert (len(loss_story) < 2) or (loss_story[-1] < loss_story[-2])
        assert tf.reduce_mean((tf.norm(v0, axis=1) - 1) ** 2) < 1e-9
        
        
    ############################################
    ### HyperbolicalExponentialMapLayer UT 1 ###
    print("HyperbolicalExponentialMapLayer UT 1")

    v = expand_to_hyperboloid(tf.Variable([[1., 1., 1.], [1, 0, 0], [4, 5, 2], [-5, 1, 0]], dtype=tf.float64))
    with tf.GradientTape() as g:
        g.watch(v)
        loss = tf.reduce_mean((HyperbolicalExponentialMapLayer(None)(v) - [2, 1, 1, 1]) ** 2)

    gg = g.gradient(loss, v)
    # print(gg)
    assert np.mean(gg.numpy() ** 2) > 0.01  # nonzero gradients assert
    self_dot = dot_hyp(v - gg, v - gg)  # should be on hyperboloid, e.g. equal (-1)
    # print(self_dot)
    assert np.mean((self_dot + 1) ** 2) < 1e-9
    
    
    ############################################
    ### HyperbolicalExponentialMapLayer UT 2 ###
    print("HyperbolicalExponentialMapLayer UT 2")

    v0 = tf.Variable(expand_to_hyperboloid(np.array(
        [[1., 1., 1.], [1, 0, 0.5], [4, 5, 2], [-5, 1, 0]],
        dtype=np.float64
    )), dtype=tf.float64)

    opt = tf.keras.optimizers.SGD(learning_rate=tf.Variable(0.1, dtype=tf.float64))

    def get_loss():
        v2 = HyperbolicalExponentialMapLayer(opt)(v0)
        # print(v2)
        c = tf.constant([2, 1, 1, 1], dtype=tf.float64)
        # v1 = tf.nn.l2_normalize(v0, axis=1)
        loss = tf.reduce_mean((v2 - c) ** 2)
        # print("Loss = ", loss)
        return loss

    loss_story = list()

    for _ in range(100):
        # v0.assign(v0 / tf.expand_dims(tf.norm(v0, axis=1), 1))
        opt.minimize(get_loss, v0)
        loss_story.append(get_loss().numpy())
        self_dot = dot_hyp(v0, v0)
        # print(v0)
        # print(f"loss = {round(loss_story[-1], 5)}, self_dot = {self_dot}")
        assert (len(loss_story) < 2) or (loss_story[-1] < loss_story[-2])
        assert tf.reduce_mean((self_dot + 1) ** 2) < 1e-9