# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from thumt.layers.nn import linear


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4, name=None):
    """
    This function adds a bunch of sinusoids of different frequencies to a
    Tensor. See paper: `Attention is all you need'

    :param x: A tensor with shape [batch, length, channels]
    :param min_timescale: A floating point number
    :param max_timescale: A floating point number
    :param name: An optional string

    :returns: a Tensor the same shape as x.
    """

    with tf.name_scope(name, default_name="add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) *
                       tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def split_heads(inputs, num_heads, name=None):
    """ Split heads
    :param inputs: A tensor with shape [batch, ..., channels]
    :param num_heads: An integer
    :param name: An optional string
    :returns: A tensor with shape [batch, heads, ..., channels / heads]
    """

    with tf.name_scope(name, default_name="split_heads", values=[inputs]):
        x = inputs
        n = num_heads
        old_shape = x.get_shape().dims
        ndims = x.shape.ndims

        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        perm = [0, ndims - 1] + [i for i in range(1, ndims - 1)] + [ndims]
        return tf.transpose(ret, perm)


def combine_heads(inputs, name=None):
    """ Combine heads
    :param inputs: A tensor with shape [batch, heads, length, channels]
    :param name: An optional string
    :returns: A tensor with shape [batch, length, heads * channels]
    """

    with tf.name_scope(name, default_name="combine_heads", values=[inputs]):
        x = inputs
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        x.set_shape(new_shape)

        return x


def high_combine_heads(inputs, scope=None):
    """ Combine heads in high order (low rank bilinear)
    :param inputs: A tensor with shape [batch, heads, length, channels]
    :param queries: A tensor with shape [batch, length_q, key_size], already linear and scale
    :returns: A tensor with shape [batch, length, heads * channels]
    channels=64, key_size=512 for base model
    """

    with tf.variable_scope(scope, default_name="high_combine_heads", values=[inputs]):
        x = inputs
        heads = x.shape[1].value # 8
        channels = x.shape[3].value # 64
        x = tf.transpose(x, [0, 2, 1, 3]) #shape [batch, q_length, heads, channels]
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        c = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        c.set_shape(new_shape) #[batch, length, heads * channels]

        c1 = linear(c, 512, True, True, scope="c1_transform") #[batch, q_length, 32]
        c2 = linear(c, 512, True, True, scope="c2_transform") #[batch, q_length, 32]
        # c1 = tf.tanh(c1)
        # c2 = tf.tanh(c2)
        outputs = c1 * c2
        outputs = tf.concat([outputs, c], -1) #concact to consider first-order

        return outputs


def diff_outputs(inputs, name=None):
    """ Calculate the differences of all heads outputs
    :param inputs: A tensor with shape [batch, heads, q_length, channels]
    :param name: An optional string
    :returns: A tensor with shape [batch, q_length]
    """

    with tf.name_scope(name, default_name="diff_outputs", values=[inputs]):
        x = inputs
        x = tf.transpose(x, [0, 2, 1, 3])  #shape [batch, q_length, heads, channels]
        x = tf.nn.l2_normalize(x, dim=-1) #normalize the last dimension
        x1 = tf.expand_dims(x, 2)  #shape [batch, q_length, 1, heads, channels]
        x2 = tf.expand_dims(x, 3)  #shape [batch, q_length, heads, 1, channels]
        cos_diff = tf.reduce_sum(tf.multiply(x1, x2), axis=[-1]) #shape [batch, q_length, heads, heads], broadcasting

        cos_diff_square = tf.reduce_mean(tf.square(cos_diff), axis=[-2,-1])
        cos_diff = tf.reduce_mean(cos_diff, axis=[-2,-1]) + 1.0  #shape [batch, q_length]

        return cos_diff


def diff_subspaces(inputs, name=None):
    """ Calculate the differences of all heads subspaces
    :param inputs: A tensor with shape [batch, heads, length_kv, depth_v]
    :param name: An optional string
    :returns: A tensor with shape [batch, length_kv]
    """

    with tf.name_scope(name, default_name="diff_subspaces", values=[inputs]):
        x = inputs
        x = tf.transpose(x, [0, 2, 1, 3])  #shape [batch, length_kv, heads, depth_v]
        x = tf.nn.l2_normalize(x, dim=-1) #normalize the last dimension
        x1 = tf.expand_dims(x, 2)  #shape [batch, length_kv, 1, heads, depth_v]
        x2 = tf.expand_dims(x, 3)  #shape [batch, length_kv, heads, 1, depth_v]
        cos_diff = tf.reduce_sum(tf.multiply(x1, x2), axis=[-1]) #shape [batch, length_kv, heads, heads], broadcasting

        cos_diff_square = tf.reduce_mean(tf.square(cos_diff), axis=[-2,-1])
        cos_diff = tf.reduce_mean(cos_diff, axis=[-2,-1]) + 1.0  #shape [batch, length_kv]

        return cos_diff


def diff_positions(inputs, name=None):
    """ Calculate the differences of all heads alignment matrices (attention weights)
    :param inputs: A tensor with shape [batch, heads, length_q, length_kv]
    :param name: An optional string
    :returns: A tensor with shape [batch], alignment from sentence to sentence
    """

    with tf.name_scope(name, default_name="diff_positions", values=[inputs]):
        x = inputs
        heads = tf.cast(tf.shape(x)[1], tf.float32)
        x1 = tf.expand_dims(x, 1)  #shape [batch, 1, heads, length_q, length_kv]
        x2 = tf.expand_dims(x, 2)  #shape [batch, heads, 1, length_q, length_kv]

        sos_diff = tf.subtract(x1, x2) #shape [batch, heads, heads, length_q, length_kv], broadcasting
        sos_diff = tf.transpose(sos_diff, [0, 3, 1, 2, 4]) #shape [batch, length_q, heads, heads, length_kv]
        sos_diff = tf.reduce_sum(tf.square(sos_diff), axis=[-3,-2,-1]) / (heads*heads) #shape [batch, length_q]
        # sos_diff_log = tf.negative(tf.log(sos_diff))
        # sos_diff = tf.negative(sos_diff) + 1.0  # Query side needs mask, which is at outside

        mul_diff = tf.multiply(x1, x2) #shape [batch, heads, heads, length_q, length_kv]
        mul_diff = tf.transpose(mul_diff, [0, 3, 1, 2, 4]) #shape [batch, length_q, heads, heads, length_kv]
        mul_diff = tf.reduce_sum(mul_diff, axis=[-3,-2,-1]) / (heads*heads) #shape [batch, length_q]
        # mul_diff_log = tf.negative(tf.log(mul_diff))
        #mul_diff = tf.negative(mul_diff) + 1.0  # Query side needs mask, which is at outside

        cos_diff = tf.multiply(tf.nn.l2_normalize(x1, dim=-1), tf.nn.l2_normalize(x2, dim=-1))
        cos_diff = tf.transpose(cos_diff, [0, 3, 1, 2, 4]) #shape [batch, length_q, heads, heads, length_kv]
        cos_diff = tf.reduce_sum(cos_diff, axis=[-3,-2,-1]) / (heads*heads) #shape [batch, length_q], no need to plus one

        return mul_diff


def heads_classification(inputs, myMatrix, myBias,name=None):
    """ Calculate the cross_entropy of 8 heads classification
    :param inputs: A tensor with shape [batch, heads, len_q, channels]
    :param name: An optional string
    :returns: A tensor with shape [1], reduce_mean
    """

    with tf.name_scope(name, default_name="heads_classification", values=[inputs,myMatrix,myBias]):
        x = inputs
        batch = tf.shape(x)[0] # is None
        heads = x.shape[1].value # 8
        len_q = tf.shape(x)[2] # is None
        channels = x.shape[3].value # 64
        label = tf.range(heads) #shape [heads]
        x = tf.transpose(x, [0, 2, 1, 3])  #shape [batch, len_q, heads, channels]

        label_word = tf.tile(label, [batch*len_q]) #shape[batch*len_q*heads]
        x_word = tf.reshape(x, [-1, channels])
        logit_word =linear(x_word, heads, True, True, scope="head_class")  #shape [batch*len_q*heads, heads]
        ce_word = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_word,logits=logit_word) #shape[batch*len_q*heads]
        output_word = tf.reduce_mean(ce_word)

        label_senten = tf.tile(label, [batch]) #shape[batch*heads]
        x_senten = tf.reshape(tf.reduce_mean(x,axis=[1]), [-1, channels]) #shape [batch*heads, channels]
        logit_senten = linear(x_senten, heads, True, True, scope="heads_class")  #shape [batch*heads, heads]
        ce_senten = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_senten,logits=logit_senten) #shape[batch*heads]
        output_senten = tf.reduce_mean(ce_senten)

        return output_word

def attention_bias(inputs, mode, inf=-1e9, name=None):
    """ A bias tensor used in attention mechanism
    :param inputs: A tensor
    :param mode: one of "causal", "masking", "proximal" or "distance"
    :param inf: A floating value
    :param name: optional string
    :returns: A 4D tensor with shape [batch, heads, queries, memories]
    """

    with tf.name_scope(name, default_name="attention_bias", values=[inputs]):
        if mode == "causal":
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "proximal":
            length = inputs
            r = tf.to_float(tf.range(length))
            diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
            m = tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)
            return m
        elif mode == "distance":
            length, distance = inputs
            distance = tf.where(distance > length, 0, distance)
            distance = tf.cast(distance, tf.int64)
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            mask_triangle = 1.0 - tf.matrix_band_part(
                tf.ones([length, length]), distance - 1, 0
            )
            ret = inf * (1.0 - lower_triangle + mask_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        else:
            raise ValueError("Unknown mode %s" % mode)


def attention(query, memories, bias, hidden_size, cache=None, reuse=None,
              dtype=None, scope=None):
    """ Standard attention layer

    :param query: A tensor with shape [batch, key_size]
    :param memories: A tensor with shape [batch, memory_size, key_size]
    :param bias: A tensor with shape [batch, memory_size]
    :param hidden_size: An integer
    :param cache: A dictionary of precomputed value
    :param reuse: A boolean value, whether to reuse the scope
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string, the scope of this layer
    :return: A tensor with shape [batch, value_size] and
        a Tensor with shape [batch, memory_size]
    """

    with tf.variable_scope(scope or "attention", reuse=reuse,
                           values=[query, memories, bias], dtype=dtype):
        mem_shape = tf.shape(memories)
        key_size = memories.get_shape().as_list()[-1]

        if cache is None:
            k = tf.reshape(memories, [-1, key_size])
            k = linear(k, hidden_size, False, False, scope="k_transform")

            if query is None:
                return {"key": k}
        else:
            k = cache["key"]

        q = linear(query, hidden_size, False, False, scope="q_transform")
        k = tf.reshape(k, [mem_shape[0], mem_shape[1], hidden_size])

        hidden = tf.tanh(q[:, None, :] + k)
        hidden = tf.reshape(hidden, [-1, hidden_size])

        # Shape: [batch, mem_size, 1]
        logits = linear(hidden, 1, False, False, scope="logits")
        logits = tf.reshape(logits, [-1, mem_shape[1]])

        if bias is not None:
            logits = logits + bias

        alpha = tf.nn.softmax(logits)

        outputs = {
            "value": tf.reduce_sum(alpha[:, :, None] * memories, axis=1),
            "weight": alpha
        }

    return outputs


def additive_attention(queries, keys, values, bias, hidden_size, concat=False,
                       keep_prob=None, dtype=None, scope=None):
    """ Additive attention mechanism. This layer is implemented using a
        one layer feed forward neural network

    :param queries: A tensor with shape [batch, heads, length_q, depth_k]
    :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
    :param values: A tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: A tensor
    :param hidden_size: An integer
    :param concat: A boolean value. If ``concat'' is set to True, then
        the computation of attention mechanism is following $tanh(W[q, k])$.
        When ``concat'' is set to False, the computation is following
        $tanh(Wq + Vk)$
    :param keep_prob: a scalar in [0, 1]
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string, the scope of this layer

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, length_q]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    with tf.variable_scope(scope, default_name="additive_attention",
                           values=[queries, keys, values, bias], dtype=dtype):
        length_q = tf.shape(queries)[2]
        length_kv = tf.shape(keys)[2]
        q = tf.tile(tf.expand_dims(queries, 3), [1, 1, 1, length_kv, 1])
        k = tf.tile(tf.expand_dims(keys, 2), [1, 1, length_q, 1, 1])

        if concat:
            combined = tf.tanh(linear(tf.concat([q, k], axis=-1), hidden_size,
                                      True, True, name="qk_transform"))
        else:
            q = linear(queries, hidden_size, True, True, name="q_transform")
            k = linear(keys, hidden_size, True, True, name="key_transform")
            combined = tf.tanh(q + k)

        # shape: [batch, heads, length_q, length_kv]
        logits = tf.squeeze(linear(combined, 1, True, True, name="logits"),
                            axis=-1)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        if keep_prob or keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        outputs = tf.matmul(weights, values)

        return {"weights": weights, "outputs": outputs}


def multiplicative_attention(queries, keys, values, bias, keep_prob=None,
                             name=None):
    """ Multiplicative attention mechanism. This layer is implemented using
        dot-product operation.

    :param queries: A tensor with shape [batch, heads, length_q, depth_k]
    :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
    :param values: A tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: A tensor
    :param keep_prob: a scalar in (0, 1]
    :param name: the name of this operation

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_kv]
        outputs: A tensor with shape [batch, heads, length_q, depth_v]
    """

    with tf.name_scope(name, default_name="multiplicative_attention",
                       values=[queries, keys, values, bias]):
        # shape: [batch, heads, length_q, length_kv]
        logits = tf.matmul(queries, keys, transpose_b=True)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        if keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        outputs = tf.matmul(weights, values)

        return {"weights": weights, "outputs": outputs}


def multihead_attention(queries, memories, bias, num_heads, key_size,
                        value_size, output_size, params, keep_prob=None, myMatrix=None, myBias=None, output=True,
                        state=None, dtype=None, scope=None):
    """ Multi-head scaled-dot-product attention with input/output
        transformations.

    :param queries: A tensor with shape [batch, length_q, depth_q]
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param bias: A tensor (see attention_bias)
    :param num_heads: An integer dividing key_size and value_size
    :param key_size: An integer
    :param value_size: An integer
    :param output_size: An integer
    :param keep_prob: A floating point number in (0, 1]
    :param output: Whether to use output transformation
    :param state: An optional dictionary used for incremental decoding
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_kv]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_attention",
                           values=[queries, memories], dtype=dtype):
        next_state = {}

        if memories is None:
            # self attention
            size = key_size * 2 + value_size
            combined = linear(queries, size, True, True, scope="qkv_transform")
            q, k, v = tf.split(combined, [key_size, key_size, value_size],
                               axis=-1)

            if state is not None:
                k = tf.concat([state["key"], k], axis=1)
                v = tf.concat([state["value"], v], axis=1)
                next_state["key"] = k
                next_state["value"] = v
        else:
            q = linear(queries, key_size, True, True, scope="q_transform")
            combined = linear(memories, key_size + value_size, True,
                              scope="kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=-1)

        # split heads
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        diff_subspace = diff_subspaces(v) #shape [batch, length_kv]

        # scale query
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5

        # attention
        results = multiplicative_attention(q, k, v, bias, keep_prob)

        # combine heads
        weights = results["weights"]
        x = combine_heads(results["outputs"])

        # new combine heads
        # new_queries = linear(queries, key_size, True, True, scope="new_q_transform")
        # new_queries *= key_depth_per_head ** -0.5
        # x = high_combine_heads(results["outputs"])
        
        # if myBias is None:
        #     x = x0 # default use both enc and dec
            
        diff_output = diff_outputs(results["outputs"]) #shape [batch, q_length]
        diff_position = diff_positions(weights)
        # head_classification = heads_classification(results["outputs"], myMatrix, myBias) #shape []

        if params.disagreement == "outputs":
            diffheads = diff_output
        elif params.disagreement == "subspaces":
            diffheads = diff_subspace
        else:
            diffheads = diff_position

        output = False  # aggregate outside
        if output:
            outputs = linear(x, output_size, True, True,
                             scope="new_output_transform")
        else:
            outputs = x

        outputs = {"weights": weights, "outputs": outputs, "diffheads": diffheads}

        if state is not None:
            outputs["state"] = next_state

        return outputs
