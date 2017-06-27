import datetime
from datetime import timedelta
import os
import tensorflow as tf
import numpy as np
import numpy.ma as ma
#404 = Stock not found Found
#state_size = 100 # depth of rnn
#sequence_length = 10 #Numbers of days per matrix
features = []
def get_data(stock,time):
    add = []
    if type(time) == datetime.date:
        time = date__str(time)
    try:
        with open("/Users/DanielLongo/Desktop/StockModule/StockFiles/"+stock+"","r") as file:
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
                Low = float(line[3]) # Day's Low data == []:
                add = [PO,PC,AdjClose,Vol,High,Low] #price open,price close
    except FileNotFoundError:
        print("FileNotFoundError")
        add = [float(-444),float(-444),float(-444),float(-444),float(-444),float(-444)]
    if add == []:
        add = [float(-444),float(-444),float(-444),float(-444),float(-444),float(-444)]
    return add

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
        
def getMasks(data):
    add = data
    for i in range(len(data)):#if there is a value the next matrix marks that as 1 else is 0
        if data[i] == -444:
            add[i] = 0
            continue
        add[i] = 1
    return add

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


def get_dates(start,end):
    d1 = start
    d2 = end
    if type(start) == list: #"Year-Month-Day" 
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

def stack(a,b):
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return np.vstack((a,b))

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
    V1 = np.reshape(V1,(len(dates),len(stocks)*len(features))) #v1 is BAtch
    V2 = np.reshape(V2,(len(dates),len(stocks)*len(features))) #v2 is Masks
    return V1,V2

def create_Y(stocks,end):
    if type(end) == list:
        end = str__date(end)
    predictdate = end + timedelta(days=1)
    while True:
        vector = create_vector(stocks,date__str(predictdate))
        if vector[0][1] == -444:
            predictdate = predictdate + timedelta(1)
            continue 
        
        Y = np.array(vector[0])
        return Y

def masked_array(array,mask):
    final = []
    for i in range(len(array)):
        if mask[i][0] == 0:
            continue
        final +=  [array[i]]
    return final

def getY(examples_X):
    return examples_X.pop(0)

def create_XY(stocks,start,end,sequence_length,exPerBatch,Features):
    global features
    features = Features
    start = str__date(start)
    end = str__date(end)
    starting = start
    ending = starting + timedelta(sequence_length)
    max_length = (end-start).days
    days = 0
    examples_X = []
    batches_X = []
    examples_Y = []
    batches_Y = []
    while True:
        days += sequence_length
        if days > max_length:
            break 

        X,maskX = create_X(stocks,starting,ending)
        X = masked_array(X,maskX)
        Y = create_Y(stocks,ending+timedelta(1))
        examples_X.append(X)
        examples_Y.append(Y)
        starting += timedelta(sequence_length)
        ending = starting + timedelta(sequence_length)
    final = None #so i can be referenced outside the for loop
    return examples_X,examples_Y

    for i in range(0,len(examples_X),exPerBatch):
        start = i
        end = start + exPerBatch
        batches_X.append(examples_X[start:end])
        batches_Y.append(examples_Y[start:end])
        final = i
    
    addX = examples_X[final:]
    addY = examples_Y[final:]
    if addX != []:
        batches_X.append(addX)
        batches_Y.append(addY)
    print('Y',batches_Y)
    print('X',batches_X)
    return zip(batches_X,batches_Y)


x = create_XY(
    stocks = ['ZNGA'],
    start = '2017-02-01',
    end = '2017-02-28',
    sequence_length = 2,
    exPerBatch = 5,
    Features = [0,1,1,3,0,0])

print(x)
print(np.shape(np.array(x[0][2])))

print("All Done! :)")
