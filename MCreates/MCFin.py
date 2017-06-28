import datetime
from datetime import timedelta
import os
import tensorflow as tf
import numpy as np
#404 = Stock not found Found
#state_size = 100 # depth of rnn
#sequence_length = 10 #Numbers of days per matrix
features = []

def get_data(stock,time):
    add = []
    if type(time) == datetime.date:
        time = date__str(time)
    try:
        with open("/Users/DanielLongo/Desktop/Stocks/"+stock+"","r") as file:
            for line in file:
                line = line.split(',')
                Date = line[0] # The Date
                if Date != time:
                    continue
                PO = float(line[1]) # Price Open
                PC = float(line[4]) # Price Close
                Vol = float(line[6]) # Volume
                AdjClose = float(line[5]) # Adjusted Close
                High = float(line[2]) # Day's High
                Low = float(line[3]) # Day's Low data == []:
                add = [PO,PC,AdjClose,Vol,High,Low] #price open,price close
    except FileNotFoundError:
        print("FileNotFoundError")
        add = [float(-444),float(-444),float(-444),float(-444),float(-444),float(-444)]
    if add == []:
        add = [float(-444),float(-444),float(-444),float(-444),float(-444),float(-444)]
    return add

def create_vector(stocks,date):
    data = []
    masks = []
    date = str__date(date)
    for i in range(len(stocks)):
        stock = stocks[i]
        today = get_data(stock,date)
        if today[0] == -444:
  #          yesterday = [float(-444),float(-444),float(-444),float(-444),float(-444),float(-444)]
            data += today
            masks += getMasks(today)
            continue
        yesterday = getYesterday(date,stock)
        add = normalize(today,yesterday)
        data += add
        masks += getMasks(add)
    return [data,masks]

def normalize(today,yesterday):
    global featues
    new = today
    if today[0] == -444:
        return [float(-444),float(-444),float(-444),float(-444),float(-444),float(-444)]
    for i in range(len(today)):
        compareNumber = features[i] #shows which value today normalizes with
        todays = today[i]
        yesterdays = yesterday[i]
        if compareNumber == -1:
            new[i] = today[i]
            continue
        new[i] = (today[i]-yesterday[compareNumber])/yesterday[compareNumber]
    return new

def getYesterday(date,stock):
    counter = 0
    if type(date) == list:
        date = str__date(date)
    while True:
        counter += 1
        newDate = date - timedelta(days=counter)
        yesterday = get_data(stock,newDate)
        if yesterday[0] != -444:
            return yesterday
        
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

def masked_arrayX(array,mask,shift = None):
    final = []
    if shift != None:
        array = array[shift:]
    for i in range(len(array)):
        if mask[i][0] == 0:
            continue
        final +=  [array[i]]
    return final

def masked_arrayY(array,mask,shift = None): #mask is the mask X so is a different dimension
    final = []
    print(array)
    if shift != None: # shifts the begining of array and end of mask
        array = array[shift:]
        mask = mask[:-shift]
    for i in range(len(mask)):
        if mask[i][0] == 0:
            continue
        final += [array[i]]
    return final

def stack(a,b):
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return np.vstack((a,b))

def getMasks(data):
    add = data
    for i in range(len(data)):#if there is a value the next matrix marks that as 1 else is 0
        if data[i] == -444:
            add[i] = 0
            continue
        add[i] = 1
    return add

def create_XY(stocks,start,end,length,exPerBatch,Features):
    global features
    features= Features
    days = 0
    start = str__date(start)
    end = str__date(end) # actual end date
    #endb = start + timedelta(days=length) 
    #startb = start
    #lengthb = (end-start).days # length of time from start to finish 
    batches_X = []
    batches_Y = []
    batch_X = []
    batch_Y = []
    X,maskX = create_X(stocks,date__str(start),date__str(end))
    X = masked_arrayX(X,maskX)
    ending = 0
    starting = 0
    while True:
        ending += length
        if ending + 1 > len(X):
            break
        x = X[starting:ending]
        starting += length
        y = X[ending]
        batch_X.append(x)
        batch_Y.append(y)
    batch_X = np.array(batch_X)
    batch_Y = np.array(batch_Y)
    for i in range(0,len(batch_X),exPerBatch):
        start = i
        end = i + exPerBatch
        batches_X.append(batch_X[start:end])
        batches_Y.append(batch_Y[start:end])
   # print(len(batches_X),len(batches_Y))
  #  print(np.shape(batches_X[0]),np.shape(batches_Y[0]))
    return zip(batches_X,batches_Y)

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
                      

#print(len(tickers))
#x = (create_XY(['AAPL'],'2017-02-02','2017-02-25',10,5,['Open','CLose','AdjClose','Volume','High','Low'])) #stocks,start,end,number of days per matrixh
# list(tickers),str(start),str(end),int(len(time)),int(examples per batch),list(features)
#(batches_X,batches_Y,batchesM_X,batchesM_Y
'''
print('###################batches_X',x[0])
print('###################batches_Y',x[1])
print('###################batchesM_X',x[2])
print('###################batchesM_Y',x[3])
'''
#print(x[0])
'''
x = create_XY(
    stocks = ['ZNGA'],
    start = '2017-02-01',
    end = '2017-02-28',
    length = 4,
    exPerBatch = 2,
    Features = [0,1,1,3,0,0])

print("All Done! :)")
'''
