# -*- coding: utf-8 -*-

class BasicModelConfig:

    def __init__(self,FLAGS,num_chars,num_tags,num_segs=4):
        #configurations for the model
        self.char_dim = FLAGS.char_dim
        self.seg_dim = FLAGS.seg_dim
        self.lstm_dim = FLAGS.lstm_dim
        self.num_chars = num_chars
        self.num_tags = num_tags
        self.num_segs = num_segs
        # configurations for training
        self.lr = FLAGS.lr
        self.clip = FLAGS.clip
        self.l2_lamda = FLAGS.l2_lamda


