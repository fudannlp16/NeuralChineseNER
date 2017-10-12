# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cPickle
import json
from config import BasicModelConfig
from models import BasicModel
from progressbar import ProgressBar
from data_utils import load_sentences,update_tag_scheme,char_mapping,tag_mapping,augment_with_pretrained,prepare_dataset
from data_utils import load_word2vec,BatchManager
from utils import *

tf.flags.DEFINE_boolean("train",       True,      "Whether train the model")

# configurations for the model
tf.flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
tf.flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
tf.flags.DEFINE_integer("seg_dim",     0,         "Embedding size for segmentation, 0 if not used")

# configurations for training
tf.flags.DEFINE_float("dropout_keep",  0.5,        "Dropout rate")
tf.flags.DEFINE_float("batch_size",    20,         "batch size")
tf.flags.DEFINE_float("clip",          5,          "Gradient clip")
tf.flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
tf.flags.DEFINE_float("l2_lamda",      0,          "lamda of l2_loss")

tf.flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")
tf.flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
tf.flags.DEFINE_boolean("pre_emb",     True,       "Whether use pre-trained embedding")
tf.flags.DEFINE_boolean("zeros",       False,      "Whether replace digits with zero")

tf.flags.DEFINE_integer("max_epoch",   100,        "max epoch")
tf.flags.DEFINE_float("memory_usage",  1.0,        "The maximum CPU memory usage")
tf.flags.DEFINE_string("checkpoints",  "checkpoints",  "The folder for saving trained models")
tf.flags.DEFINE_string("model_name",   "basic_model",   "The name of model")
tf.flags.DEFINE_string("summaries_dir", "summaries",    "The folder for saving tensorboard files")

tf.flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
tf.flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
tf.flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")
tf.flags.DEFINE_string("pre_emb_file", os.path.join("data", 'wiki_100.utf8'),  "Path for pre_trained embeddings")
tf.flags.DEFINE_string("map_file",     "data/maps.pkl",     "file for maps")

FLAGS=tf.flags.FLAGS

def train():
    # load data sets
    train_sentences=load_sentences(FLAGS.train_file,FLAGS.zeros)
    dev_sentences=load_sentences(FLAGS.dev_file,FLAGS.zeros)
    test_sentences=load_sentences(FLAGS.test_file,FLAGS.zeros)

    # appoint tagging scheme (IOB/IOBES)
    train_sentences=update_tag_scheme(train_sentences,FLAGS.tag_schema)
    dev_sentences=update_tag_scheme(dev_sentences,FLAGS.tag_schema)
    test_sentences=update_tag_scheme(test_sentences,FLAGS.tag_schema)

    #create maps if not exist
    if not os.path.exists(FLAGS.map_file):
        if FLAGS.pre_emb:
            char_to_id,_=char_mapping(train_sentences)
            char_to_id,id_to_char=augment_with_pretrained(char_to_id,'wiki_100.utf8')
        else:
            char_to_id, id_to_char=char_mapping(train_sentences)
        tag_to_id, id_to_tag=tag_mapping(train_sentences)
        with open(FLAGS.map_file,'wb') as f:
            cPickle.dump([char_to_id,id_to_char,tag_to_id,id_to_tag],f,cPickle.HIGHEST_PROTOCOL)
    else:
        with open(FLAGS.map_file,'rb') as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag=cPickle.load(f)

    # prepare data, get a collection of list containing index
    train_data=prepare_dataset(train_sentences,char_to_id,tag_to_id,True)
    dev_data=prepare_dataset(dev_sentences,char_to_id,tag_to_id,True)
    test_data=prepare_dataset(test_sentences,char_to_id,tag_to_id,True)
    print "%i %i %i sentences in train / dev / test." % (len(train_data),len(dev_data),len(test_data))

    if not FLAGS.pre_emb:
        pre_emb=None
    else:
        pre_emb=load_word2vec(FLAGS.pre_emb_file,char_to_id,FLAGS.char_dim)
        print "init embedding shape: (%d,%d)" %(pre_emb.shape[0],pre_emb.shape[1])

    train_manager=BatchManager(train_data,FLAGS.batch_size,True)
    dev_manager=BatchManager(dev_data,FLAGS.batch_size,False)
    test_manager=BatchManager(test_data,FLAGS.batch_size,False)

    config=BasicModelConfig(FLAGS,len(char_to_id),len(tag_to_id),4)
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_usage
    with tf.Session(config=tfConfig) as sess:
        print "Train started!"
        model=BasicModel(config,pre_emb)
        saver=tf.train.Saver()

        # tensorboard
        if not os.path.exists(FLAGS.summaries_dir):
            os.mkdir(FLAGS.summaries_dir)
        merged=tf.summary.merge_all()
        train_writer=tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir,FLAGS.model_name,"train"),sess.graph)
        test_writer=tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir,FLAGS.model_name,"test"),sess.graph)

        # load previous trained model or create a new model
        if not os.path.exists(FLAGS.checkpoints):
            os.mkdir(FLAGS.checkpoints)
        model_name=os.path.join(FLAGS.checkpoints,FLAGS.model_name)
        ckpt=tf.train.get_checkpoint_state(FLAGS.checkpoints)
        if ckpt and ckpt.model_checkpoint_path:
            print "restore from previous traied model: %s" % FLAGS.model_name
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        def evaluate(sess,model,manager):
            strings=[]
            predicts=[]
            goldens=[]
            bar = ProgressBar(max_value=manager.num_batch)
            for batch in bar(manager.iter_batch()):
                batch_string,batch_predict,batch_golden=model.evaluate_step(sess,batch)
                strings.extend(batch_string)
                predicts.extend(batch_predict)
                goldens.extend(batch_golden)
            return strings,predicts,goldens

        best_eval_f1=0
        noimpro_num=0
        for i in range(FLAGS.max_epoch):
            #train
            train_loss=[]
            bar = ProgressBar(max_value=train_manager.num_batch)
            for step,batch in bar(enumerate(train_manager.iter_batch())):
                batch.append(merged)
                summary,global_step,batch_loss=model.train_step(sess,batch,FLAGS.dropout_keep)
                #add summary to tensorboard
                train_writer.add_summary(summary,global_step)
                train_loss.append(batch_loss)
            print "Epoch %d Train loss is %.4f" % (i+1,np.mean(train_loss))

            #dev
            strings,predicts,goldens=evaluate(sess,model,dev_manager)
            eval_f1=report_results(strings,predicts,goldens,id_to_char,id_to_tag,'outputs/dev')
            if eval_f1>best_eval_f1:
                best_eval_f1=eval_f1
                noimpro_num=0
                saver.save(sess,model_name)
            else:
                noimpro_num+=1
            print "Epoch %d Best eval f1:%.6f" % (i+1,best_eval_f1)

            #test
            strings,predicts,goldens=evaluate(sess,model,test_manager)
            test_f1=report_results(strings,predicts,goldens,id_to_char,id_to_tag,'outputs/test',True)
            #early_stop
            if noimpro_num>=3:
                print "Early stop! Final F1 scores on test data is :%.6f" % test_f1
                break
            print

def evaluate_line():
    with open(FLAGS.map_file, 'rb') as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = cPickle.load(f)
    config=BasicModelConfig(FLAGS,len(char_to_id),len(tag_to_id),4)
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_usage
    with tf.Session(config=tfConfig) as sess:
        model = BasicModel(config, None)
        saver = tf.train.Saver()
        model_name = os.path.join(FLAGS.checkpoints, FLAGS.model_name)
        saver.restore(sess, model_name)
        while True:
            line = raw_input('\n请输入测试句子(0:Exit):\n'.encode('utf-8')).decode('utf-8')
            if line=='0':
                exit()
            strings, predicts, _ = model.evaluate_step(sess, input_from_line(line, char_to_id))
            string=strings[0]
            tags=[id_to_tag[t] for t in predicts[0]]
            assert len(string)==len(tags)
            result_json=result_to_json(string,tags)
            print_result_json(result_json)

if __name__ == '__main__':
    if FLAGS.train:
        train()
    else:
        evaluate_line()





