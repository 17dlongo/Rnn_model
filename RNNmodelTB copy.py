import tensorflow as tf
import sys
sys.path.append('/Users/DanielLongo/Desktop/Rnn_Model/MCreates')
from Mh_create import create_XY #(stocks,start,end,length)
from time import strftime, gmtime
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
        self.exPerBatch= 5
        self.start_date = '2017-01-01'
        self.end_date = '2017-02-01'
 #       self.tickers = make_list('Nasdaq.csv')
        self.state_size = 100 # depth of rnn number of hidden layers 
        self.sequence_length = 20 # len(time)
        self.epoch = 100000 #iterations of model
        self.lr = .1 #learning rate
        self.stocks = ['AAPL']
        self.tickers= self.stocks
        self.features = [0,1,1,-1,0,0] #shows if data is to be normalized and to what number feature #['Open','CLose','AdjClose','Volume','High','Low']
        self.Spred = None # -1 is for no normalization
        self.logs_path = 'tensorboard/'+strftime("%Y_%m_%d_%H_%M_%S",gmtime())
        
    def add_placeholders(self):
        self.input_mask_placeholder = tf.placeholder(tf.bool,(None,self.sequence_length))
        self.labels_mask_placeholder = tf.placeholder(tf.bool, (None,(len(self.features)*len(self.stocks))))
        self.input_placeholder = tf.placeholder(tf.float32, (None,self.sequence_length,(len(self.features)*len(self.stocks))))
        self.labels_placeholder = tf.placeholder(tf.float32, (None,(len(self.features)*len(self.stocks))))

    def create_feed_dict(self,inputs_batch,labels_batch = None,inputs_mask=None,labels_mask=None):
        feed_dict = {
            self.input_mask_placeholder: inputs_mask,
            self.labels_mask_placeholder: labels_mask,
            self.input_placeholder: inputs_batch,
            self.labels_placeholder: labels_batch}
#        print(feed_dict[self.input_placeholder])
        return feed_dict

    def add_prediction_op(self):
 #       masked_input = tf.boolean_mask(self.input_placeholder,self.input_mask_placeholder)
        lstm_cell = tf.contrib.rnn.LSTMCell(self.state_size)
#        print(masked_input,self.input_mask_placeholder)
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
        Diff = (tf.subtract(self.labels_placeholder,preds))
        batch_loss = tf.sqrt(tf.reduce_sum(tf.square(Diff),axis=1))
        mean_loss = tf.reduce_mean(batch_loss)
        return mean_loss

    def add_training_op(self,loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return train_op

    def train_on_batch(self,sess,inputs_batch,labels_batch,inputs_mask=None,labels_mask=None):
        feed = self.create_feed_dict(inputs_batch,labels_batch=labels_batch,inputs_mask=inputs_mask,labels_mask=labels_mask)
        print(self.Spred.eval(session=sess,feed_dict=feed))
        print(self.labels_placeholder.eval(session=sess,feed_dict=feed))
        _, loss,summary = sess.run([self.train_op,self.loss_op,self.merged_summary_op],feed_dict=feed)       
        self.train_writer.add_summary(summary,self.counter)
        self.train_writer.flush()
        return loss
    
    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        print("pred")
        self.loss_op = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss_op)
        print("loss")
        tf.summary.scalar('Loss', self.loss_op)
        self.merged_summary_op = tf.summary.merge_all()
        print("Finsihed Building")

      
    def rnn(self):
        init = tf.global_variables_initializer()
        sess= tf.Session()
 #       saver = tf.train.import_meta_graph("TtestT.meta")
#        saver.restore(sess,tf.train.latest_checkpoint('./'))
        sess.run(init)
        self.train_writer = tf.summary.FileWriter(self.logs_path+'/train',sess.graph)
        batches = list(create_XY(self.tickers,self.start_date,self.end_date,self.sequence_length,self.exPerBatch,self.features)) #  return batch_X,batch_Y,masksX,masksY
 #       print(batches[0])
        self.counter = 0
        print("train")
        for i in range(self.epoch):
            for batchX,batchY,maskX,maskY in batches:
                #print('batchX',batchX)
                print(self.train_on_batch(sess,batchX,batchY,inputs_mask=maskX,labels_mask=maskY))
                self.counter += 1
#        saver.save(sess,"TtestT")

def main():
    model = Model()
    model.build()
    return model.rnn()

if __name__ == "__main__":
    main()

print("All Done! :)")
    
