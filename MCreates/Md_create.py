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
    global features
    features= Features
    days = 0
    start = str__date(start)
    end = str__date(end)
    endb = start + timedelta(days=length)
    startb = start
    lengthb = (end-start).days
    while True:
        if days > lengthb:
            break
        x,maskX = create_X(stocks,date__str(startb),date__str(endb))
        y,maskY = create_Y(stocks,date__str(endb))
        days += length
        startb += timedelta(length)
        endb += timedelta(length)
        if days == length:
            batch_X = np.array(x)
            batch_Y = np.array(y)
            masksX = np.array(maskX)
            masksY = np.array(maskY)
            continue
        batch_X = np.vstack((batch_X,x))
        batch_Y = np.vstack((batch_Y,y))
        masksX = np.vstack((masksX,maskX))
        masksY = np.vstack((masksY,maskY))
    print(batch_X)
    print(batch_Y)
    print(masksX)
    print(masksY)
    for i in range(0,batch_X.shape[0],exPerBatch):
        start = i
        end = i + exPerBatch
        if start == 0:
            batches_X = np.array(batch_X[start:end])
            batches_Y = np.array(batch_Y[start:end])       
            batchesM_X = np.array(masksX[start:end])
            batchesM_Y = np.array(masksY[start:end])
            continue
            
        batches_X = np.vstack((batches_X,np.array(batch_X[start:end])))
        batches_Y = np.vstack((batches_Y,np.array(batch_Y[start:end])))       
        batchesM_X = np.vstack((batchesM_X,np.array(masksX[start:end])))
        batchesM_Y = np.vstack((batchesM_Y,np.array(masksY[start:end])))
    return zip(batches_X,batches_Y,batchesM_X,batchesM_Y)
'''
def create_XY(stocks,start,end,length,exPerBatch):
    session = tf.InteractiveSession()
    days = 0
    start = str__date(start)
    end = str__date(end)
    endb = start + timedelta(days=length)
    startb = start
    lengthb = (end-start).days
    batch_X = []
    batch_Y = []
    masksX = []
    masksY = []
    batches_X = []
    batches_Y = []
    batchesM_X = []
    batchesM_Y= []
    while True:
        if days > lengthb:
            break
        x,maskX = create_X(stocks,date__str(startb),date__str(endb))
        y,maskY = create_Y(stocks,date__str(endb))
        days += length
        startb += timedelta(length)
        endb += timedelta(length)
        batch_X.append(x)
        batch_Y.append(y)
        masksX.append(maskX)
        masksY.append(maskY)
    batch_X = np.array
    batch_Y = tf.stack(batch_Y,axis=0)
    masksX = tf.stack(masksX,axis=0)
    masksY = tf.stack(masksY,axis=0)
    for i in range(0,batch_X.shape[0],exPerBatch):
        start = i
        batches_X = np.concatenate((batches_X,batch_X[start:end]))
        batches_Y = np.concatenate((batches_Y,batch_Y[start:end]))       
        batchesM_X = np.concatenate((batchesM_X,masksX[start:end]))
        batchesM_Y = np.concatenate((batchesM_Y,masksY[start:end]))
    return zip(batches_X,batches_Y,batchesM_X,batchesM_Y)
'''
'''
def create_XY(stocks,start,end,length,exPerBatch):
    days = 0
    start = str__date(start)
    end = str__date(end)
    endb = start + timedelta(days=length)
    startb = start
    lengthb = (end-start).days
    batch_X = []
    batch_Y = []
    masksX = []
    masksY = []
    batches_X = []
    batches_Y = []
    batchesM_X = []
    batchesM_Y= []
    while True:
        if days > lengthb:
            break
        x,maskX = create_X(stocks,date__str(startb),date__str(endb))
        y,maskY = create_Y(stocks,date__str(endb))
        days += length
        startb += timedelta(length)
        endb += timedelta(length)
        batch_X.append(x)
        batch_Y.append(y)
        masksX.append(maskX)
        masksY.append(maskY)
    batch_X += [batch_X]
    batch_Y += [batch_Y]
    masksX += [masksX]
    masksY += [masksY]
    print(batch_X)
    batch_X = np.array(batch_X,dtype=float)
    batch_Y = np.array(batch_Y,dtype=float)
    masksX = np.array(masksX,dtype=float)
    masksY = np.array(masksY,dtype=float)
    print('shape of x',batch_X.shape)
    for i in range(0,batch_X.shape[0],exPerBatch):
        start = i
        end = i +exPerBatch
        batches_X.append(batch_X[start:end])
        batches_Y.append(batch_Y[start:end])       
        batchesM_X.append(masksX[start:end])
        batchesM_Y.append(masksY[start:end])
    return zip(batches_X,batches_Y,batchesM_X,batchesM_Y)
    '''

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
x = (create_XY(['AAPL'],'2017-01-25','2017-02-02',3,5,['Open','CLose','AdjClose','Volume','High','Low'])) #stocks,start,end,number of days per matrixh
# list(tickers),str(start),str(end),int(len(time)),int(examples per batch),list(features)
#(batches_X,batches_Y,batchesM_X,batchesM_Y
'''
print('###################batches_X',x[0])
print('###################batches_Y',x[1])
print('###################batchesM_X',x[2])
print('###################batchesM_Y',x[3])
'''

