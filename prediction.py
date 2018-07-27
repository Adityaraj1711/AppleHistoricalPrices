import csv
import numpy as p
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def get_data(filename):
  global dates
  global price
  with open(filename , 'r') as csvfile:
    csvFileReader = csv.reader(csvfile)
    next(csvFileReader)
    for row in csvFileReader:
      dates.append(int(row[0].split('_')[0]))
      prices.append(float(row[1]))
  return
  
def predict_prices(data,price,x):
  global dates
  global price
  dates = np.reshape(dates,(len(dates),1))
  svr_lin = SVR(kernel = 'linear',C = 1e3)
  svr_poly = SVR(kernel = 'poly',C = 1e3,degree = 2)
  svr_rbf = SVR(kernel='rbf',C=1e3,gamma = 0.1)
  svr_lin.fit(dates,prices)
  svr_poly.fit(dates,prices)
  svr_rbf.fit(dates,prices)
  plt.scatter(dates,price,color='black',label = 'dates')
  plt.plot(dates,svr_rbf.predict(dates),color='red',label='rbf model')
  plt.plot(dates,svr_lin.predict(dates),color='green',label='linear model')
  plt.plot(dates,svr_poly.predict(dates),color='red',label='polynomial model')
  plt.xlabel('dates')
  plt.ylabel('prices')
  plt.title('support vector regression')
  plt.legend(loc='best')
  plt.show()
  return svr_rbf.predict(x)[0],svr_lin.predict(x)[0],svr_poly.predict(x)[0]
  
if __name__ =='__main__':
  dates = []
  prices  = []
  get_data('aapl.csv')
  predicted_price = predict_price(dates,price,29)
  print(predicted_price)
  
