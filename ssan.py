# -*- coding: utf-8 -*-
"""
@author: Erting Pan
"""

from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

import indices
data = indices.ReadDatasets()

num_classes    = 9

#************ Network Parameters ***************
window_size     = 27
num_components = 4 
num_input      = 1
timesteps      = 102                        
num_hidden     = 512 

batch_size     = 128
learning_rate  = 0.005
dropout        = 0.6
training_steps = 10000
display_step   = 100

spe_X     = tf.placeholder("float", [None, timesteps, num_input], name='spe_X') 
spa_X     = tf.placeholder(tf.float32, shape=[None, window_size, window_size, num_components],name='spa_X')
Y         = tf.placeholder("float", [None, num_classes], name='Y')
keep_prob = tf.placeholder(tf.float32,name='keep_prob') 

#************ Define weights & bias *************
spe_weights = tf.Variable(tf.random_normal([num_hidden*2, num_classes]))
spe_biases =  tf.Variable(tf.random_normal([num_classes]))

spa_weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 4, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}
spa_biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}  

m_weights = {
    'wf1' : tf.Variable(tf.random_normal([num_classes, 256])),
    'wf2' : tf.Variable(tf.random_normal([512, 1024])),
    'mout': tf.Variable(tf.random_normal([1024, num_classes]))
}
m_biases = {
    'bf1' : tf.Variable(tf.random_normal([256])),
    'bf2' : tf.Variable(tf.random_normal([1024])),
    'mout': tf.Variable(tf.random_normal([num_classes]))
}  

'''////////////////////////////
       Spectral branch
///////////////////////////'''

ATTENTION_SIZE= 32
def SpectralAttention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)
    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value    
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    
    v      = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
    vu     = tf.tensordot(v, u_omega, axes=1, name='vu') 
    alphas = tf.nn.softmax(vu, name='alphas')        

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    return output, alphas
        
#Creat Attention bi-RNN branch model
def ARNN(x,weights,biases):
    gru_fw_cell = tf.nn.rnn_cell.GRUCell(num_hidden)
    gru_bw_cell = tf.nn.rnn_cell.GRUCell(num_hidden)
    outputs, _  = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, x, dtype=tf.float32)
    a_output, alphas = SpectralAttention(outputs, ATTENTION_SIZE)
    outputs     = tf.nn.xw_plus_b(a_output, spe_weights, spe_biases)
    return outputs 

'''////////////////////////////
       Spatial branch
///////////////////////////'''

def SpatialAttention(feature_map, weight_decay=0.0004, scope="", reuse=None):
    with tf.variable_scope(scope, 'SpatialAttention', reuse=reuse):
        # Tensorflow's tensor is in BHWC format. H for row split while W for column split.
        _, H, W, C = tuple([x for x in feature_map.get_shape()])
        w_s = tf.get_variable("SpatialAttention_w_s_1",[C, 1],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.0001),
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        b_s = tf.get_variable("SpatialAttention_b_s", [1],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        spatial_attention_fm = tf.matmul(tf.reshape(feature_map, [-1, C]), w_s) 
        spatial_attention_fm = tf.tanh(spatial_attention_fm + b_s)
        spatial_attention_fm = tf.nn.sigmoid(tf.reshape(spatial_attention_fm, [-1, W * H]))
        attention   = tf.reshape(tf.concat([spatial_attention_fm] * C, axis=1), [-1, H, W, C])
        attended_fm = attention * feature_map
        return attended_fm
        
# Create some wrappers for simplicity
def Conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def Maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
                          
#Creat Attention CNN branch model
def ACNN(x, spa_weights, spa_biases, dropout):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x  = tf.reshape(x, shape=[-1, 27, 27, 4])
    
    ax = SpatialAttention(x)

    conv1 = Conv2d(ax, spa_weights['wc1'], spa_biases['bc1'])
    conv1 = Maxpool2d(conv1, k=2)

    conv2 = Conv2d(conv1, spa_weights['wc2'], spa_biases['bc2'])
    conv2 = Maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, spa_weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, spa_weights['wd1']), spa_biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, spa_weights['out']), spa_biases['out'])
    return out

'''////////////////////////////
        Merge model
///////////////////////////'''

def SSAN(spe_x, spe_weights, spe_biases, spa_x, spa_weights, spa_biases, keep_prob):
    spe_logits = ARNN(spe_x, spe_weights, spe_biases)    
    spa_logits = ACNN(spa_x, spa_weights, spa_biases, keep_prob)

    spe_fc1 = tf.reshape(spe_logits, [-1, m_weights['wf1'].get_shape().as_list()[0]])
    spe_fc1 = tf.add(tf.matmul(spe_fc1, m_weights['wf1']), m_biases['bf1'])
    spe_fc1 = tf.nn.relu(spe_fc1)
    
    spa_fc1 = tf.reshape(spa_logits, [-1, m_weights['wf1'].get_shape().as_list()[0]])
    spa_fc1 = tf.add(tf.matmul(spa_fc1, m_weights['wf1']), m_biases['bf1'])
    spa_fc1 = tf.nn.relu(spa_fc1)

    merge   = tf.keras.layers.concatenate([spe_fc1, spa_fc1])

    m_fc2   = tf.reshape(merge, [-1, m_weights['wf2'].get_shape().as_list()[0]])
    m_fc2   = tf.add(tf.matmul(m_fc2, m_weights['wf2']), m_biases['bf2'])
    m_fc2   = tf.nn.relu(m_fc2)
    
    logits = tf.add(tf.matmul(m_fc2, m_weights['mout']), m_biases['mout'])

    return logits

logits = SSAN(spe_X, spe_weights, spe_biases, spa_X, spa_weights, spa_biases, keep_prob)
tf.add_to_collection('pre_prob', logits)
prediction = tf.nn.softmax(logits)
tf.add_to_collection('pred_label', prediction)

# Define loss and optimizer
loss_op  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
tf.summary.scalar('loss_op', loss_op)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

# Evaluate model (with test logits, for dropout to be dis
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('batch_accuracy', accuracy)

'''////////////////////////////
           Training
///////////////////////////'''

init  = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=50)

with tf.Session() as sess:
    best = 0.8
    sess.run(init)
    merged = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(
        '/data/pan/data/paviac/merge/model/loss_record', sess.graph)
    for step in range(1, training_steps+1):
        batch_spe_x, batch_spa_x, batch_y = data.train.next_batch(batch_size)
        batch_spe_x = batch_spe_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={spe_X: batch_spe_x, spa_X: batch_spa_x, 
                                      Y: batch_y, keep_prob: 1.0})
        if step % display_step == 0 or step == 1:
            summary, loss, acc = sess.run([merged, loss_op, accuracy], feed_dict={spe_X: batch_spe_x,spa_X: batch_spa_x,
                                                                 Y: batch_y, keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            train_summary_writer.add_summary(summary, step)
        #start validation
        if step % 1000 == 0:
            batch_sizeall = data.valid.num_examples
            val_batch_spe_x, val_batch_spa_x, val_batch_y = data.valid.next_batch(batch_sizeall)
            val_batch_spe_x = val_batch_spe_x.reshape((val_batch_spe_x.shape[0], timesteps, num_input))
            val_acc = sess.run(accuracy, feed_dict={spe_X: val_batch_spe_x, spa_X: val_batch_spa_x, 
                                                    Y: val_batch_y, keep_prob: 1.0})
            # valid_summary_writer.add_summary(summary, step)
            print("valid accuracy = " + "{:.3f}".format(val_acc))
            if val_acc > best:           
                best = val_acc
                print("Step " + str(step))
                filename = ('ssan030.ckpt')
                filename = os.path.join('/data/pan/data/paviac/merge/model/',filename)
                saver.save(sess, filename)
            print("best valid accuracy = " + "{:.3f}".format(best))
    print("Optimization Finished!")





