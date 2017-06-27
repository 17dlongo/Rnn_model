import tensorflow as tf
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
        self.start_date = '2016-01-01'
        self.end_date = '2017-03-03'
 #       self.tickers = make_list('Nasdaq.csv')
        self.state_size = 100 # depth of rnn number of hidden layers 
        self.sequence_length = 20 # len(time)
        self.epoch = 3 #iterations of model
        self.lr = .1 #learning rate
        self.stocks = make_list('Nasdaq.csv')
        self.tickers= self.stocks
        self.features = ['Open','CLose','AdjClose','Volume','High','Low']
        
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
        W = tf.get_variable('W',(self.state_size,(len(self.stocks)*len(self.features))),initializer = xavier)
        B = tf.get_variable('B',(1,len(self.stocks)*len(self.features)))
        Output,State = tf.nn.dynamic_rnn(lstm_cell,self.input_placeholder,dtype= tf.float32)
        State=State[1]
        #print('Output',Output)
        #print('state',State)
        #print('w',W)
        Spred = tf.matmul(State,W)+B 
        #print('Spred',Spred)
        return Spred

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
 #       saver = tf.train.import_meta_graph("TtestT.meta")
   #     saver.restore(sess,tf.train.latest_checkpoint('./'))
        sess.run(init)
        batches = list(create_XY(self.tickers,self.start_date,self.end_date,self.sequence_length,self.exPerBatch,self.features)) #  return batch_X,batch_Y,masksX,masksY
        for i in range(self.epoch):
            for batchX,batchY,maskX,maskY in batches:
                print(self.train_on_batch(sess,batchX,batchY,inputs_mask=maskX,labels_mask=maskY))
        saver = tf.train.Saver()
        saver.save(sess,"TtestT")

def main():
    model = Model()
    model.build()
    return model.rnn()

if __name__ == "__main__":
    main()
    
    






#########################################
import datetime
from datetime import timedelta
import os
import tensorflow as tf
import numpy as np
#404 = Stock not found Found
#state_size = 100 # depth of rnn
#sequence_length = 10 #Numbers of days per matrix
features = []
def create_vector(stocks,time): # (stock ticker, dates) date = [year-0month-0day]
    data = []
    datab = [] #datab is a 0 and 1 matrix if the data for the specified point is avaliable 
    for i in range(len(stocks)):
        add = None
        try:
            with open("/Users/DanielLongo/Desktop/StockModule/StockFiles/"+stocks[i]+"","r") as file:
                for line in file:
                    line = line.split(',')
                    Date = line[0] # The Date
                    if Date != time:
                        continue
                    PO = float(line[1]) # Price Open
                    PC = float(line[4]) # Price Close
                    Vol = float(line[5]) # Volume
                    AdjClose = float(line[6]) # Adjusted Close
                    High = float(line[2]) # Day's High
                    Low = float(line[3]) # Day's Lowif data == []:
                    add = [PO,PC,AdjClose,Vol,High,Low] #price open,price close
            if add == None:
                add = [float(-444),float(-444),float(-444),float(-444),float(-444),float(-444)]
        except FileNotFoundError:
            add = [float(-444),float(-444),float(-444),float(-444),float(-444),float(-444)]
             #add= 'FILENOTFOUND','FILENOTFOUND','FILENOTFOUND','FILENOTFOUND','FILENOTFOUND','FILENOTFOUND'
        data += add
        i = -1
        for item in add:#if there is a value the next matrix marks that as 1 else is 0
            i += 1
            if item == -444:
                #print(item,'0')
                add[i] = float(0)
            else:
                #print(item,'1')
                add[i] = float(1)
        datab += add
    return [data,datab]


def get_dates(start,end): #"Year-Month-Day" 
    d1 = str__date(start) #start date
    d2 = str__date(end) # end date
    dates = [d1 + timedelta(days=x) for x in range((d2-d1).days+1)] # the mystery line
    ndates = []
    for item in dates: # converts dateime to a string
        ndates  += [date__str(item)]
    return ndates # returns a list of dates between start and end

def str__date(day):
    day = day.split("-")
    return datetime.date(int(day[0]),int(day[1]),int(day[2]))

def date__str(date):
    return date.strftime('%Y-%m-%d')

def create_X(stocks,start,end):
    dates = get_dates(start,end)
    global features
    for i in range(len(dates)):
        vector = (create_vector(stocks,dates[i]))
        Batch = np.array(vector[0])
        Mask = np.array(vector[1])
        if i == 0:
            V1 = Batch
            V2 = Mask # array be is the map of missing values for stock data
            continue
        V1 = np.vstack((V1,np.array(Batch)))
        V2 = np.vstack((V2,np.array(Mask)))
    V1 = np.reshape(V1,(len(dates),len(stocks)*len(features)))
    V2 = np.reshape(V2,(len(dates),len(stocks)*len(features)))
    return V1,V2

def create_Y(stocks,end):
    end = str__date(end)
    predictdate = end + timedelta(days=1)
    while True:
        vector = create_vector(stocks,date__str(predictdate))
        if vector[0][1] == -444 and vector[0][3] == -444:
            predictdate += timedelta(days=1)
            continue 
            
        Y = np.array(vector[0])
        Masks = np.array(vector[1])
        return Y,Masks

def create_XY(stocks,start,end,length,exPerBatch,Features):
    length -= 1 #makes indexing exlusive
    global features
    features= Features
    days = 0
    start = str__date(start)
    end = str__date(end)
    endb = start + timedelta(days=length)
    startb = start
    lengthb = (end-start).days
    batches_X =[]
    batches_Y = []
    batchesM_Y = []
    batchesM_X =[]
    batch_X =[]
    batch_Y = []
    batchM_Y = []
    batchM_X =[]
    while True:
        if days > lengthb:
            break
 #       print((stocks,date__str(startb),date__str(endb)))
        x,maskX = create_X(stocks,date__str(startb),date__str(endb))
        y,maskY = create_Y(stocks,date__str(endb))
        days += length
        startb += timedelta(length)
        endb += timedelta(length)
        batch_X.append(x)
        batch_Y.append(y)
        batchM_X.append(maskX)
        batchM_Y.append(maskY)
    '''
    print('batch_X',batches_X)
    print('batch_Y',batches_Y)
    print('batchM_X',batchesM_X)
    print('batchM_Y',batchesM_Y)
    '''
    for i in range(0,len(batch_X),exPerBatch):
        start = i
        end = i + exPerBatch
        batches_X.append(batch_X[start:end])
        batches_Y.append(batch_Y[start:end])
        batchesM_X.append(batchM_X[start:end])
        batchesM_Y.append(batchM_Y[start:end])
    return zip(batches_X,batches_Y,batchesM_X,batchesM_Y)


def make_list(file): # takes Nasdaq or NYSE file and creates a list of tickers
    tickers = []
    counter = 0
    for line in open(file,"r"):
        counter += 1
        y = line.split(',')
        tickers.append((y[0].strip('""')))
        if counter > 5: #change ten to change the number of columns
            break
    tickers.remove('Symbol')
    return tickers
                      
tickers = make_list('Nasdaq.csv')
#print(len(tickers))
#x = (create_XY(['AAPL','SQ'],'2016-12-25','2017-02-02',10,5,['Open','CLose','AdjClose','Volume','High','Low'])) #stocks,start,end,number of days per matrixh
# list(tickers),str(start),str(end),int(len(time)),int(examples per batch),list(features)
#(batches_X,batches_Y,batchesM_X,batchesM_Y
'''
print('###################batches_X',x[0])
print('###################batches_Y',x[1])
print('###################batchesM_X',x[2])
print('###################batchesM_Y',x[3])
'''
