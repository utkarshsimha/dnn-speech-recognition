import tensorflow as tf
from sklearn.externals import joblib
import numpy as np
from keras.preprocessing import sequence
import sklearn.preprocessing
import pdb

NUM_FEATS = 39
NUM_CLASSES = 27 
DATA_DIR = "../data"
def loadData( data_dir ):
    train = ( joblib.load( data_dir+"/"+"train_data.npy" ), joblib.load( data_dir+"/"+"train_target.npy" ) )
    test = ( joblib.load( data_dir+"/"+"test_data.npy" ), joblib.load( data_dir+"/"+"test_target.npy" ) )
    return train, test

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

class Model:
    def __init__( self, n_feats, hidden_layers, n_classes ):
        self.n_feats = n_feats
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder( tf.float32, shape=[ None, None, self.n_feats ], name="X" )
            self.labels = tf.sparse_placeholder( tf.int32, name="labels" ) #CTC requires it to be a SparseTensor O.o
            self.seq_len_vector = tf.placeholder( tf.int32, name="seq_len_vectorj" ) #CTC requires it to be a single element list
            self.seq_len = tf.placeholder( tf.int32, name="seq_len" )

            ''' Long short term memory Recurrent Neural Network '''
            #One of the INTERSPEECH papers says to use peepholes for speech recognition
            #They basically allow cell outputs to go to the input and output gates
            #self.lstm_cell = lambda n_hid: tf.contrib.rnn.LSTMCell( n_hid, use_peepholes=True )
            self.rnn_cell = lambda n_hid: tf.contrib.rnn.BasicRNNCell( n_hid )
            #TODO Add dropout
            self.stack_lstm = tf.contrib.rnn.MultiRNNCell( list( map( self.rnn_cell, self.hidden_layers ) ) )
            self.rnn_outs, state = tf.nn.dynamic_rnn( self.stack_lstm, self.x, self.seq_len, dtype=tf.float32 )

            #Affine combination to get output
            n_hid = self.hidden_layers[-1] #Output of last hidden layer
            n_out = self.n_classes + 1
            batch_size = tf.shape( self.x )[0] 
            seq_len = tf.shape( self.x )[1] #Why not use self.seq_len? 
            #Merge batch_size and seq_len of RNN output -- ( batch_size * seq_len, n_hid )
            self.hid = tf.reshape( self.rnn_outs, [ -1, n_hid ] )
            self.W = tf.Variable( tf.truncated_normal( [ n_hid, n_out ] ), name="W" )
            self.B = tf.Variable( tf.zeros( [ n_out ] ), name="B" )
            self.logits = tf.matmul( self.hid, self.W ) + self.B 
            #CTC requires the following shape: [max_time, batch_size, num_classes]
            self.logits = tf.reshape( self.logits, [ -1, batch_size, n_out ] )

            ''' Connectionist Temporal Classification '''
            self.cost = tf.reduce_mean( tf.nn.ctc_loss( inputs=self.logits, labels=self.labels, sequence_length=self.seq_len_vector ) )
            
            ''' Adam optimizer '''
            self.optimizer = tf.train.AdamOptimizer().minimize( self.cost )

            ''' Decoder predictions '''
            ###Greedy decoder (special case of beam search)
            decoded_out, log_probs = tf.nn.ctc_greedy_decoder( self.logits, self.seq_len_vector )

            ###Beam search decoder
            #decoded_out, log_probs = tf.nn.ctc_beam_search_decoder( self.logits, self.seq_len_vector )

            self.pred = decoded_out[0] #Single element list
            self.dense_pred = tf.sparse_tensor_to_dense( self.pred )

            ''' Character error rate '''
            self.cer = tf.reduce_mean( tf.edit_distance( tf.cast( self.pred, tf.int32 ), self.labels ) )
    def fit( self, X, y ):
        batch_size = 32 
        n_epochs = 1000
        with tf.Session( graph = self.graph ) as sess:
            sess.run( tf.global_variables_initializer() )
            for ep in range( n_epochs ):
                ''' Mini-batching '''
                offset = (ep * batch_size) % (y.shape[0] - batch_size)
                batch_X = sequence.pad_sequences( X[ offset:(offset + batch_size) ] )
                batch_seq_len = batch_X.shape[1]
                batch_y = y[ offset:(offset + batch_size) ]
                sparse_batch_y = sparse_tuple_from( batch_y )
                feed_dict = { self.x: batch_X, self.labels: sparse_batch_y,
                        self.seq_len: batch_seq_len, self.seq_len_vector: [ batch_seq_len ] * batch_size }
                _, cost, pred, cer = sess.run( [ self.optimizer, self.cost, self.pred, self.cer ], feed_dict=feed_dict )
                print "Cost: {}, CER:{}".format( cost, cer )
                if ep % 10 == 0:
                    idx = np.random.randint( 0, test[0].shape[0] )
                    test_inp = test[0][idx]
                    test_feed = { self.x: test_inp.reshape( 1, -1, self.n_feats ),\
                            self.seq_len: test_inp.shape[1], self.seq_len_vector: [ test_inp.shape[1] ] }
                    pred = sess.run( [ self.dense_pred ], feed_dict=test_feed )
                    idxToChar = lambda seq: "".join( [ " " if idx == 0 else chr(idx + ( ord('a') - 1 )) for idx in seq] )
                    print "Predictionn: {}, index:{} ; \n Actual: {}".format( \
                            idxToChar(pred[0].tolist()[0]), pred[0].tolist()[0], idxToChar(test[1][idx]) )


if __name__ == '__main__':
	train, test = loadData( DATA_DIR )
        #pdb.set_trace()
	HIDDEN_LAYERS = [ 256, 256 ]
	model = Model( NUM_FEATS, HIDDEN_LAYERS,  NUM_CLASSES )
	model.fit( train[0], train[1] )

