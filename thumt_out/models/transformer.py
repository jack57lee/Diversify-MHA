# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
              dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output

def _ffn_layer_2(inputs, hidden_size, output_size, keep_prob=None,
              dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer_2", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output

def _ffn_layer_tanh(inputs, hidden_size, output_size, keep_prob=None,
              dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer_tanh", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.tanh(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)
            output = tf.nn.softmax(output)

        return output


def squash(vector):
    epsilon = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

def dynamic_routing(output_heads, params):   #[batch, length, heads * channels]
    with tf.variable_scope("dynamic_routing"):
        num_capsules = 512
        heads = 8
        channels = 64
        output_heads = tf.reshape(output_heads, [tf.shape(output_heads)[0], tf.shape(output_heads)[1], heads, channels])
        # [batch, length, heads, channels]
        combined_output = _ffn_layer_2(
            _layer_process(tf.reshape(output_heads, [tf.shape(output_heads)[0], tf.shape(output_heads)[1], channels * heads])
            , params.layer_preprocess),
            heads*channels,    #2048, how to set it?
            heads*channels*heads,  #512*8=4096
            1.0 - params.relu_dropout,
        )
        combined_output = tf.reshape(combined_output, [tf.shape(output_heads)[0], tf.shape(output_heads)[1], num_capsules, heads, 1])

        b = tf.zeros([tf.shape(output_heads)[0], tf.shape(output_heads)[1], num_capsules, heads])
        routing_iter = 3
        for i in range(routing_iter):
            c = tf.reshape(tf.nn.softmax(b), [tf.shape(b)[0], tf.shape(b)[1], num_capsules, heads, 1])
            s = tf.reduce_sum(c * combined_output, axis = 3)
            v = squash(s)# [batch, length, num_capsules, 1]
            temp_v = tf.reshape(v, [tf.shape(output_heads)[0], tf.shape(output_heads)[1], num_capsules, 1, 1])
            temp_v2 = tf.reduce_sum(temp_v * combined_output, axis = 4) # [batch, length, num_capsules, 8]
            b += temp_v2

        v = tf.reshape(v, [tf.shape(b)[0], tf.shape(b)[1], num_capsules])
        outputs = _layer_process(v, params.layer_postprocess) # if necessary?
        
        return outputs

def em_routing(output_heads, params):   #[batch, length, heads * channels]
    with tf.variable_scope("em_routing"):
        num_capsules = 512
        heads = 8
        channels = 64
        # output_heads = tf.reshape(output_heads, [tf.shape(output_heads)[0], tf.shape(output_heads)[1], heads, channels])
        # [batch, length, heads, channels]

        # v0 = layers.nn.linear(output_heads[:,:,0,:], num_capsules, True, True, scope="v0_transform") #[batch,len,numcaps]
        # v1 = layers.nn.linear(output_heads[:,:,1,:], num_capsules, True, True, scope="v1_transform")
        # v2 = layers.nn.linear(output_heads[:,:,2,:], num_capsules, True, True, scope="v2_transform")
        # v3 = layers.nn.linear(output_heads[:,:,3,:], num_capsules, True, True, scope="v3_transform")
        # v4 = layers.nn.linear(output_heads[:,:,4,:], num_capsules, True, True, scope="v4_transform")
        # v5 = layers.nn.linear(output_heads[:,:,5,:], num_capsules, True, True, scope="v5_transform")
        # v6 = layers.nn.linear(output_heads[:,:,6,:], num_capsules, True, True, scope="v6_transform")
        # v7 = layers.nn.linear(output_heads[:,:,7,:], num_capsules, True, True, scope="v7_transform")
        # vote_in = tf.stack([v0,v1,v2,v3,v4,v5,v6,v7], axis=2)
        # [batch, length, heads, numcaps]
        
        vote_in = _ffn_layer(
            _layer_process(output_heads
            , params.layer_preprocess),
            heads * channels,  # how to set?
            heads * channels * heads, #512*8=4096
            1.0 - params.relu_dropout,
        )
        vote_in = tf.reshape(vote_in, [tf.shape(output_heads)[0], tf.shape(output_heads)[1], heads, num_capsules, 1])

        r = tf.ones([tf.shape(output_heads)[0], tf.shape(output_heads)[1], heads, num_capsules]) / num_capsules

        initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)
        # beta_v = tf.get_variable(
        #           name='beta_v', shape=[1, 1, 1, num_capsules, 1], dtype=tf.float32, 
        #           initializer=initializer
        #         )
        # beta_a = tf.get_variable(
        #           name='beta_a', shape=[1, 1, 1, num_capsules, 1], dtype=tf.float32,
        #           initializer=initializer
        #         )
        
        routing_iter = 3
        epsilon = 1e-9
        it_min = 1.0
        it_max = min(routing_iter, 3.0)
        o_mean = None

        for i in range(routing_iter):
            #M step
            inverse_temperature = it_min + (it_max - it_min) * i / max(1.0, routing_iter - 1.0)

            r = tf.reshape(r, [tf.shape(r)[0], tf.shape(r)[1], heads, num_capsules, 1]) #[?,?,8,512,1]
            r_sum = tf.reduce_sum(r, axis = 2, keep_dims = True) #[?, ?, 1, 512, 1]
            # r_sum = tf.reshape(r_sum, [tf.shape(r)[0], tf.shape(r)[1], 1, num_capsules, 1]) #[?, ?, 1, 512, 1]

            o_mean = tf.reduce_sum(r * vote_in, axis = 2, keep_dims = True) / (r_sum + epsilon) #[?, ?, 1, 512, 1]

            o_stdv = (tf.reduce_sum(r * tf.square(vote_in - o_mean), axis = 2, keep_dims = True)) / (r_sum + epsilon) #[?, ?, 1, 512, 1]

            # o_cost_h = (beta_v + 0.5 * tf.log(o_stdv + epsilon)) * r_sum # [?, ?, 1, 512, 1]
            # o_cost = tf.reduce_sum(o_cost_h, axis = -1, keep_dims = True) #[?, ?, 1, 512, 1]

            # activation_out = tf.sigmoid(inverse_temperature * (beta_a - o_cost)) #[?, ?, 1, num, 1]
            if i < routing_iter - 1:
                #E step
                o_p_unit0 = - tf.reduce_sum(tf.square(vote_in - o_mean) / (2*o_stdv), axis = -1, keep_dims=True) #[?, ?, 8, num, 1]
                o_p_unit2 = - 0.5 * tf.reduce_sum(tf.log(o_stdv + epsilon), axis = -1, keep_dims=True) #[?,?,1,num,1]
                o_p = o_p_unit0 + o_p_unit2 #[?, ?, 8, num, 1]
                # zz = tf.log(activation_out + epsilon) + o_p #[?,?,8,num,1]
                zz = o_p
                r = tf.nn.softmax(zz, dim = 3) + epsilon#[?,?,8,num,1] 
                r = tf.reshape(r, [tf.shape(r)[0], tf.shape(r)[1], heads, num_capsules]) #[?,?,8,num]

        v = tf.reshape(o_mean, [tf.shape(r)[0], tf.shape(r)[1], params.hidden_size]) #[batch, length, 512]
        outputs = _layer_process(v, params.layer_postprocess) # if necessary?
        
        return outputs


def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
		# for visualization (Baoosong)
        enc_atts=[]		
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                	mask = _ffn_layer_tanh(_layer_process(x, params.layer_preprocess), params.hidden_size, 2)
                    # [batch, len, 2], after softmax
                    if mode == "train":
                        tau = 1 / (tf.nn.softplus(layers.nn.linear(x, 1, True, True, scope="tau_transform")) + 1)
                        mask = relaxed_onehot_categorical.RelaxedOneHotCategorical(temperature=tau, probs=mask).sample()
                        mask_hard = tf.cast(tf.one_hot(tf.argmax(mask,-1),2), mask.dtype)
                        mask = tf.stop_gradient(mask_hard - mask) + mask
                    else:
                        mask = tf.Print(mask, [layer, mask[:,0:10,0]], summarize=20)
                        mask = tf.cast(tf.greater(mask, 0.5), mask.dtype)
                    mask0 = tf.expand_dims(mask[:,:,0], -1)

                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    #for visulization (Baosong)
                    if params.outweights:
                    	mask_out = tf.expand_dims(mask0, 1)
                    	mask_out = tf.tile(mask_out, [1,8,1,1])
						enc_atts.append(mask_out)					
                    y = y["outputs"]
                    y1 = em_routing(y, params)
                    y2 = layers.nn.linear(y, 512, True, True, scope="out_transform")
                    y = mask0*y1 + (1-mask0)*y2
                    #### dynamically choose whether routing
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)
		
        if params.outweights: return outputs, tf.transpose(tf.stack(enc_atts),[1,0,2,3,4])
        else: return outputs	

def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
		# for visualization (Baoosong)
		dec_atts=[]
		encdec_atts=[]		
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state
                    )
                    # for visualization (Baoosong)					
                    if params.outweights:
                        dec_atts.append(y["weights"])
                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    y = layers.nn.linear(y, 512, True, True, scope="out_transform")
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )
					# for visualization (Baoosong)					
                    if params.outweights:
                        encdec_atts.append(y["weights"])						
                    y = y["outputs"]
                    y = layers.nn.linear(y, 512, True, True, scope="out_transform")
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
			# for visualization (Baoosong)		
            if params.outweights: return outputs, next_state, tf.transpose(tf.stack(dec_atts),[1,0,2,3,4]),tf.transpose(tf.stack(encdec_atts),[1,0,2,3,4])
            else: return outputs, next_state	
        elif params.outweights: return outputs,tf.transpose(tf.stack(dec_atts),[1,0,2,3,4]),tf.transpose(tf.stack(encdec_atts),[1,0,2,3,4])
        else: return outputs

def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("bias", [hidden_size])

    # id => embedding
    # src_seq: [batch, max_src_length]
    inputs = tf.gather(src_embedding, src_seq) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(src_mask, -1)

    # Preparing encoder
    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = layers.attention.add_timing_signal(encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

		
	# for visualization (Baoosong)
    if params.outweights:
        encoder_output,enc_atts = transformer_encoder(encoder_input, enc_attn_bias, params)
        return encoder_output, enc_atts		
    else:
        encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)
        return encoder_output


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    # id => embedding
    # tgt_seq: [batch, max_tgt_length]
    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)
    targets = targets * tf.expand_dims(tgt_mask, -1)

    # Preparing encoder and decoder input
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]
    #for visulization (Baosong)		
    if params.outweights: enc_atts=state["enc_atts"]
    if mode != "infer":
        decoder_output = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)											 
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        #for visulization (Baosong)		
        if params.outweights:        		
            decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])
            decoder_output, decoder_state, dec_atts, encdec_atts = decoder_outputs								  
        else:
            decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])		
            decoder_output, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)
		#for visulization (Baosong)	
        if params.outweights:
	    #oldshape=state["dec_atts"].shape.as_list()
	    oldshape = tf.concat([tf.shape(state["dec_atts"])[:-1],tf.fill([1,],1)],axis=-1)
	    #if oldshape[-1]!=None: 
		#tf.fill(oldshape[-1]+[1], 0.0)
	    #dec_atts = tf.cond(oldshape[-1]!=None,tf.concat([tf.concat([state["dec_atts"],tf.fill(oldshape[:-1]+[1],0.0)],axis=-1),dec_atts],axis=-2),dec_atts)
	    dec_atts = tf.concat([tf.concat([state["dec_atts"],tf.fill(oldshape,0.0)],axis=-1),dec_atts],axis=-2)
	    encdec_atts = tf.concat([state["encdec_atts"],encdec_atts],axis=-2) 
	    return log_prob, {"encoder": encoder_output, "decoder": decoder_state,"enc_atts":enc_atts, "dec_atts":dec_atts,"encdec_atts":encdec_atts} 
        else: return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

    # [batch, length, channel] => [batch * length, vocab_size]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)
    labels = features["target"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


def model_graph(features, mode, params):
    #for visulization (Baosong)	
    if params.outweights: 
        encoder_output,enc_atts = encoding_graph(features, mode, params)
        state = {
            "encoder": encoder_output,
            "enc_atts":enc_atts
            }
    else: 
        encoder_output = encoding_graph(features, mode, params)
        state = {
            "encoder": encoder_output
            }
    output = decoding_graph(features, state, mode, params)

    return output


class Transformer(interface.NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                #for visulization (Baosong)	
                if params.outweights: encoder_output,enc_atts = encoding_graph(features, "infer", params)
                else: encoder_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]
                #for visulization (Baosong)					
                if params.outweights:
                    state = {
                        "encoder": encoder_output,
                        "enc_atts": enc_atts,						
                        "decoder": {
							"layer_%d" % i: {
								"key": tf.zeros([batch, 0, params.hidden_size]),
								"value": tf.zeros([batch, 0, params.hidden_size])
							}
							for i in range(params.num_decoder_layers)
                        },
                        #"dec_atts": tf.zeros([batch,params.num_decoder_layers, params.num_heads, 0, 0]),
                        "dec_atts": tf.zeros([batch,params.num_decoder_layers, params.num_heads, 0, 0]),
			#"encdec_atts": tf.zeros([batch,params.num_decoder_layers, params.num_heads, 0, 0])
			"encdec_atts": tf.zeros([batch,params.num_decoder_layers, params.num_heads, 0, tf.shape(encoder_output)[-2]])						
                    }				
                else:state = {
                        "encoder": encoder_output,
                        "decoder": {
							"layer_%d" % i: {
								"key": tf.zeros([batch, 0, params.hidden_size]),
								"value": tf.zeros([batch, 0, params.hidden_size])
							}
							for i in range(params.num_decoder_layers)
                        }
                    }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            outweights = True,
	    disagreement = "outputs"
        )

        return params
