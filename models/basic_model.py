# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn,layers,crf
__author__='liuxiaoyu'

'''
Basic BiLSTM+CRF model with word segmentation information as additional features. 
We use word segmentation information when config.seg_dim>0.
'''

class BasicModel(object):
    def __init__(self,config,char_embeddings):

        #config
        self.config=config
        self.lr=config.lr
        self.l2_lamda=config.l2_lamda
        self.clip=config.clip

        self.char_dim=config.char_dim
        self.lstm_dim=config.lstm_dim
        self.seg_dim=config.seg_dim
        self.num_tags=config.num_tags
        self.num_chars=config.num_chars
        self.num_segs=config.num_segs

        #placeholder
        self.char_inputs=tf.placeholder(dtype=tf.int32,shape=[None,None],name='CharInputs')
        self.seg_inputs=tf.placeholder(dtype=tf.int32,shape=[None,None],name='SegInputs')
        self.tags=tf.placeholder(dtype=tf.int32,shape=[None,None],name='Tags')
        self.dropout_keep=tf.placeholder(dtype=tf.float32,name='Dropout_keep')

        #shape
        #[batch_size]
        self.lengths=tf.reduce_sum(tf.cast(tf.greater(self.char_inputs, tf.zeros_like(self.char_inputs)), tf.int32), 1)
        self.batch_size=tf.shape(self.char_inputs)[0]
        self.max_length=tf.shape(self.char_inputs)[1]

        #embedding_layer
        with tf.variable_scope("embedding_layer"):
            if char_embeddings is None:
                self.char_embeddings=tf.get_variable(
                    name='char_embeddings',
                    shape=[self.num_chars,self.char_dim],
                    dtype=tf.float32
                )
            else:
                self.char_embeddings=tf.Variable(
                    char_embeddings,
                    name='char_embeddings',
                    dtype=tf.float32
                )
            char_inputs=tf.nn.embedding_lookup(self.char_embeddings,self.char_inputs)

            if self.config.seg_dim>0:
                self.seg_embeddings=tf.get_variable(
                    name='seg_embeddings',
                    shape=[self.num_segs,self.seg_dim]
                )
                seg_inputs=tf.nn.embedding_lookup(self.seg_embeddings,self.seg_inputs)
                inputs=tf.concat([char_inputs,seg_inputs],axis=-1)
            else:
                inputs=char_inputs

        #dropout
        lstm_inputs=tf.nn.dropout(inputs,keep_prob=self.dropout_keep)

        #bilistm_layer
        with tf.variable_scope("bilstm_layer"):
            cell_fw = rnn.LSTMCell(num_units=self.lstm_dim)
            cell_bw = rnn.LSTMCell(num_units=self.lstm_dim)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=lstm_inputs,
                sequence_length=self.lengths,
                dtype=tf.float32
            )
            lstm_outputs = tf.concat([output_fw, output_bw], axis=2)

        #project_layer
        self.logits=layers.fully_connected(
            inputs=lstm_outputs,
            num_outputs=self.num_tags,
            activation_fn=None,
            scope='project_layer'
        )

        #crf_layer
        with tf.variable_scope("crf_layer"):
            log_likelihood, self.transition_params=crf.crf_log_likelihood(
                inputs=self.logits,
                tag_indices=self.tags,
                sequence_lengths=self.lengths
            )

        self.loss = tf.reduce_mean(-log_likelihood)
        #summary
        tf.summary.scalar("loss",self.loss)

        #train_op
        self.global_step = tf.Variable(0, trainable=False)
        optimizer=tf.train.AdamOptimizer(self.lr)
        tvars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,tvars),self.clip)
        self.train_op=optimizer.apply_gradients(zip(grads,tvars),self.global_step)

    # train
    def train_step(self,sess,data,dropout_keep):
        _,chars,segs,tags,merged=data
        feed_dict={
            self.char_inputs:chars,
            self.seg_inputs:segs,
            self.tags:tags,
            self.dropout_keep:dropout_keep
        }
        summary,global_step, loss, _ = sess.run(
            [merged, self.global_step, self.loss, self.train_op],
            feed_dict)
        return summary,global_step,loss

    # dev/test
    def evaluate_step(self,sess,data):
        strings,chars,segs,tags=data
        feed_dict={
            self.char_inputs:chars,
            self.seg_inputs:segs,
            self.dropout_keep:1.0
        }
        logits,lengths,transition_params=sess.run([self.logits,self.lengths,self.transition_params],feed_dict)
        new_strings,predicts,new_tags=[],[],[]
        for length,logit,string,tag in zip(lengths,logits,strings,tags):
            predict,_=crf.viterbi_decode(logit[:length],transition_params)
            predicts.append(predict)
            new_strings.append(string[:length])
            new_tags.append(tag[:length])
        return new_strings,predicts,new_tags
