import tensorflow as tf
import sys
sys.path.append('/Users/DanielLongo/Desktop/Rnn_Model/MCreates')
from MCFin import create_XY #(stocks,start,end,length)
from time import strftime, gmtime
import numpy as np
#tensorboard --logdir=path/to/directory

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
        self.exPerBatch = 16
        self.start_date = '2007-01-01'
        self.end_date = '2017-02-01'
 #       self.tickers = make_list('Nasdaq.csv')
        self.state_size = 100 # depth of rnn number of hidden layers 
        self.sequence_length = 15 # len(time)
        self.epoch = 5000000 #iterations of model
        self.lr = .0001 #learning rate
        self.stocks = ['AAPL','MSFT']
        self.tickers= self.stocks
        self.features = [0,1,1,3,1,1] #shows if data is to be normalized and to what number feature #['Open','CLose','AdjClose','Volume','High','Low']
        self.Spred = None # -1 is for no normalization
        self.logs_path = 'tensorboard/'+strftime("%Y_%m_%d_%H_%M_%S",gmtime())
        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, (None,self.sequence_length,(len(self.features)*len(self.stocks))))
        self.labels_placeholder = tf.placeholder(tf.float32, (None,(len(self.features)*len(self.stocks))))

    def create_feed_dict(self,inputs_batch,labels_batch = None):
        feed_dict = {}
        feed_dict[self.input_placeholder] = inputs_batch
        if labels_batch != None:
                sel
                self.labels_placeholder: labels_batch}
                return feed_dict
        feed_dict = 
     #   print("Here",np.shape(self.input_placeholder),np.shape(inputs_batch))
#        print(feed_dict[self.input_placeholder])
        return feed_dict

    def add_prediction_op(self):
 #       masked_input = tf.boolean_mask(self.input_placeholder,self.input_mask_placeholder)
        lstm_cell = tf.contrib.rnn.LSTMCell(self.state_size)
#        print(masked_input,self.input__placeholder)
        xavier = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable('Weights',(self.state_size,(len(self.stocks)*len(self.features))),initializer = xavier)
        B = tf.get_variable('Biasis',(1,len(self.stocks)*len(self.features)))
#        Output,State = tf.nn.dynamic_rnn(lstm_cell,masked_input,dtype= tf.float32)
        Output,State = tf.nn.dynamic_rnn(lstm_cell,self.input_placeholder,dtype= tf.float32)
        State = State[1] # 0th is the initial state
        self.Spred = tf.matmul(State,W)+B
        #print('Spred',Spred)
        return self.Spred
    def add_loss_op(self,preds):
 #       masked_loss = tf.boolean_mask(preds,labels_masks_placeholder)
        Diff = (tf.subtract(self.labels_placeholder,preds)) #############################
        batch_loss = tf.sqrt(tf.reduce_sum(tf.square(Diff),axis=1))
        mean_loss = tf.reduce_mean(batch_loss)
        return mean_loss

    def add_training_op(self,loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return train_op

    def train_on_batch(self,sess,inputs_batch,labels_batch):
        feed = self.create_feed_dict(inputs_batch,labels_batch=labels_batch)
       # print(self.Spred.eval(session=sess,feed_dict=feed))
        #print(self.labels_placeholder.eval(session=sess,feed_dict=feed))
        _, loss,summary = sess.run([self.train_op,self.loss_op,self.merged_summary_op],feed_dict=feed)       
        self.train_writer.add_summary(summary,self.counter)
        self.train_writer.flush()
        return loss
    
    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss_op = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss_op)
        tf.summary.scalar('Loss', self.loss_op)
        self.merged_summary_op = tf.summary.merge_all()

      
    def rnn(self):
        init = tf.global_variables_initializer()
        sess= tf.Session()
 #       saver = tf.train.import_meta_graph("TtestT.meta")
#        saver.restore(sess,tf.train.latest_checkpoint('./'))
        sess.run(init) #initializes all global variables
        self.train_writer = tf.summary.FileWriter(self.logs_path+'/train',sess.graph) #creates a summary path for files !!!!!!
        batches = list(create_XY(self.tickers, # tickers
        self.start_date,  # start (list style)
        self.end_date, # end (list style)
        self.sequence_length, # len(time)
        self.exPerBatch,
        self.features)) #  return batch_X,batch_Y,masksX,masksY
 #       print(batches[0])
        if batches == []:
            print("ERROR")
            return None
        self.counter = 0 # counts for tensorboard summary
        #print(np.shape(batches),np.shape(batches[0]),np.shape(batches[1]))
        #print(len(batches[0]),len(batches[1]))
        #print("batches",batches[0])
        #print("batchesII",batches[1])
        for i in range(self.epoch):
            for batchX,batchY in batches:
                print(self.train_on_batch(sess,batchX,batchY))
                self.counter += 1
#        saver.save(sess,"TtestT")

def main():
    model = Model()
    model.build()
    return model.rnn()

if __name__ == "__main__":
    main()

print("All Done! :)")
    
