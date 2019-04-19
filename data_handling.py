import numpy as np
import pandas as pd

def get_points_and_labels(datapath):
    '''
    This function will read the csv and return the points for furthur
    outlier detection.
    '''
    df=pd.read_csv(datapath)
    print(df.head())
    print("Size of dataframe:",df.shape)

    #Extracting out the labels and data
    labels=df.loc[:,"Class":]
    points=df.loc[:,"V1":"Amount"]
    print("\nPrinting points sample")
    print(points.head())
    print("Printing labels sample")
    print(labels.head())
    #Getting the dataframe as numpy array
    labels=labels.values[0:20000]
    points=points.values[0:20000]
    print("shape of points: ",points.shape)
    print("shape of labels: ",labels.shape)

    #Now mean normalizing the numpy arrays
    points=(points-np.mean(points,axis=0))/np.std(points,axis=0)

    return points,labels

if __name__=="__main__":
    datapath="dataset/creditcard.csv"
    get_points_and_labels(datapath)