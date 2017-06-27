import tensorflow as tf
import sys
sys.path.append('/Users/DanielLongo/Desktop/Rnn_Model/MCreates')
from Me_create import create_XY #(stocks,start,end,length)

def make_list(file): # takes Nasdaq or NYSE file and creates a list of tickers
    tickers = []
    counter = 0
    for line in open(file,"r"):
        counter += 1
        y = line.split(',')
        tickers.append((y[0].strip('""')))
        if counter > 2000: #change ten to change the number of columns
            break
    tickers.remove('Symbol')
    return tickers

class Model(object):
    def __init__(self):
        self.exPerBatch= 5
        self.start_date = '2017-01-01'
        self.end_date = '2017-02-01'
 #       self.tickers = make_list('Nasdaq.csv')
        self.state_size = 100 # depth of rnn number of hidden layers 
        self.sequence_length = 20 # len(time)
        self.epoch = 1000 #iterations of model
        self.lr = .1 #learning rate
        self.stocks = ['ZNGA']
        self.tickers= self.stocks
        self.features = ['Open','CLose','AdjClose','Volume','High','Low']
        self.Spred = None
        
    def add_placeholders(self):
        self.input_mask_placeholder = tf.placeholder(tf.bool,(None,self.sequence_length,(len(self.features)*len(self.stocks))))
        self.labels_mask_placeholder = tf.placeholder(tf.bool, (None,(len(self.features)*len(self.stocks))))
        self.input_placeholder = tf.placeholder(tf.float32, (None,self.sequence_length,(len(self.features)*len(self.stocks))))
        self.labels_placeholder = tf.placeholder(tf.float32, (None,(len(self.features)*len(self.stocks))))

    def create_feed_dict(self,inputs_batch,labels_batch = None,inputs_mask=None,labels_mask=None):
        feed_dict = {
            self.input_mask_placeholder: inputs_mask,
            self.labels_mask_placeholder: labels_mask,
            self.input_placeholder: inputs_batch,
            self.labels_placeholder: labels_batch}
        return feed_dict

    def add_prediction_op(self):
        lstm_cell = tf.contrib.rnn.LSTMCell(self.state_size)
        xavier = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable('Weights',(self.state_size,(len(self.stocks)*len(self.features))),initializer = xavier)
        B = tf.get_variable('Biasis',(1,len(self.stocks)*len(self.features)))
        Output,State = tf.nn.dynamic_rnn(lstm_cell,self.input_placeholder,dtype= tf.float32)
        State=State[1]
        #print('Output',Output)
        #print('state',State)
        #print('w',W)
        self.Spred = tf.matmul(State,W)+B
        #print('Spred',Spred)
        return self.Spred

    def add_loss_op(self,preds):
        Diff = (tf.subtract(self.labels_placeholder,preds))
        batch_loss = tf.sqrt(tf.reduce_sum(tf.square(Diff),axis=1))
        mean_loss= tf.reduce_mean(batch_loss)
        return mean_loss

    def add_training_op(self,loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return train_op

    def train_on_batch(self,sess,inputs_batch,labels_batch,inputs_mask=None,labels_mask=None):
        feed = self.create_feed_dict(inputs_batch,labels_batch=labels_batch,inputs_mask=inputs_mask,labels_mask=labels_mask)
#        print(self.Spred.eval(session=sess,feed_dict=feed))
#        print(self.labels_placeholder.eval(session=sess,feed_dict=feed))
        _, loss = sess.run([self.train_op,self.loss],feed_dict=feed)
        return loss
    
    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        
    def rnn(self):
        init = tf.global_variables_initializer()
        sess= tf.Session()
        saver = tf.train.import_meta_graph("TtestT.meta")
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        sess.run(init)
        batches = list(create_XY(self.tickers,self.start_date,self.end_date,self.sequence_length,self.exPerBatch,self.features)) #  return batch_X,batch_Y,masksX,masksY
        print(batches)
        for i in range(self.epoch):
            for batchX,batchY,maskX,maskY in batches:
 #               print(len(batchX[0]))
                print(self.train_on_batch(sess,batchX,batchY,inputs_mask=maskX,labels_mask=maskY))
        print('H')
        saver.save(sess,"TtestT")

def main():
    model = Model()
    model.build()
    return model.rnn()

if __name__ == "__main__":
    main()
    
    
