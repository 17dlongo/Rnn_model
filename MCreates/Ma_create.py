import datetime
from datetime import timedelta
import os
#404 = Stock not found Found
def open_file(stock,start,end): # (stock ticker, dates) date = [year-0month-0day]
    data = []
    dates = get_dates(start, end)
    with open("/Users/DanielLongo/Desktop/StockModule/StockFiles/"+stock+"","r") as file:
        for line in file:
            line = line.split(',')
            Date = line[0] # The Date
            if Date not in dates:
                continue
            PO = float(line[1]) # Price Open
            PC = float(line[4]) # Price Close
            Vol = float(line[5]) # Volume
            AdjClose = float(line[5]) # Adjusted Close
            High = float(line[2]) # Day's High
            Low = float(line[3]) # Day's Low
            data += [[Date,PO,PC,AdjClose,Vol,High,Low]]
    if data == []:
        return 404
    return data

def sort(data): #input is [Date,PO,PC,AdjClose,Vol,High,Low]
    date = data[0]
    date = date.split('-')
    return (int(date[0]) + (int(date[1])*32) + int(date[2]))

def get_dates(start,end): #"Year-Month-Day" 
    d1 = str__date(start) #start date
    d2 = str__date(end) # end date
    dates = [d1 + timedelta(days=x) for x in range((d2-d1).days+1)] # the mystery line
    ndates = []
    for item in dates: # converts dateime to a string
        ndates  += [date__str(item)]
    return ndates # returns a list of dates between start and end

def date__str(date):
    return date.strftime('%Y-%m-%d')

def str__date(day):
    day = day.split("-")
    return datetime.date(int(day[0]),int(day[1]),int(day[2]))

def create_y(stock,end):
    end = date__str(end)
    end += datetime.timedelta(days=1)
    end = str__date(end)
    for l in open(("Users/DanielLongo/Desktop/StockModule/StockFiles/"+stock), 'r'):
        if nextt == True:
            Date = l[0]
            PO = int(l[1]) # Price Open
            PC = int(l[4]) # Price Close
            Vol = int(l[5]) # Volume
            AdjClose = int(l[5]) # Adjusted Close
            High = int(l[2]) # Day's High
            Low = int(l[3]) # Day's Low
            data += [[Date,PO,PC,AdjClose,Vol,High,Low]]
            return data
        if l == end:
            nextt = True
    
    
def create_X(stock,start,end):
    data = open_file(stock,start,end)
    data = sorted(data,key=sort,reverse=True)
    print(data)

def create_array(stock,start,time):
    start = str__date(start)
    end -= date__str(start - timedelta(days=time))
    start = date__str(start)
    create_X(stock,start,end)
    create_y(stock,start,end)

def create_full_array(stocks,start,time):
    start = str__date(start)
    end -= date__str(start - timedelta(days=time))
    start = date__str(start)
    X = np.zeros(0,len(stocks))
    for i in range(len(stocks)):
        X += [create_X(stocks[i],start,end)]
        Y += [create_y(stocks[i],start,end)]

    
    
    
create_X('AAPL','2017-02-28','2017-03-02')
