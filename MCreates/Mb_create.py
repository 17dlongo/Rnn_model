import datetime
from datetime import timedelta
import os
import tensorflow as tf
#404 = Stock not found Found
#state_size = 100 # depth of rnn
#sequence_length = 10 #Numbers of days per matrix
def create_vector(stocks,time): # (stock ticker, dates) date = [year-0month-0day]
    data = []
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
                    add = PO,PC,AdjClose,Vol,High,Low #price open,price close
            if add == None:
                add = float(-403),float(-403),float(-403),float(-403),float(-403),float(-403)
        except FileNotFoundError:
            add = float(-404),float(-404),float(-404),float(-404),float(-404),float(-404)
             #add= 'FILENOTFOUND','FILENOTFOUND','FILENOTFOUND','FILENOTFOUND','FILENOTFOUND','FILENOTFOUND'
        data += add
    return data


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
    session = tf.InteractiveSession()
    dates = get_dates(start,end)
    for i in range(len(dates)):
        vector = (create_vector(stocks,dates[i]))
        if i == 0:
            tensor = vector
            continue
        tensor = tf.concat([tensor,vector],0)
   # print(tensor.eval())
    tensor = tf.reshape(tensor,(len(dates),len(stocks)*6)) #IF FEATURES CHANGE CHANGE 6 TO NUMBER OF FEATURE
#    print(type(tensor))
    session.close()
    return tensor

def create_Y(stocks,end):
    end = str__date(end)
    predictdate = end - timedelta(days=1)
    while True:
        vector = create_vector(stocks,date__str(predictdate))
        if vector[1] == -403 and vector[8] == -403:
            predictdate -= timedelta(days=1)
            continue
        return tf.stack([vector],0)
        
def create_XY(stocks,start,end,length):
    length -= 1
    session = tf.InteractiveSession()
    days = 0
    start = str__date(start)
    end = str__date(end)
    endb = start + timedelta(days=length)
    startb = start
    lengthb = (end-start).days
    X = []
    Y = []
    while True:
        if days > lengthb:
            break
        x = create_X(stocks,date__str(startb),date__str(endb))
        y = create_Y(stocks,date__str(endb))
        days += length
        startb += timedelta(length)
        endb += timedelta(length)
        X += [x]
        Y += [y]
    return [X,Y]

#print(create_XY(['AAPL','ABEO'],'2017-01-02','2017-02-02',10)) #stocks,start,end,number of days per matrix
