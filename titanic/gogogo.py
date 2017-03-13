import numpy as np
import pandas as pd
from collections import Counter  
from math import isnan
#Print you can execute arbitrary python code
train = pd.read_csv("./train.csv", dtype={"Age": np.float64}, )
# test = pd.read_csv("./test.csv", dtype={"Age": np.float64}, )
TRAIN_HEADER = list(train)
# head of train: PassengerId    Survived    Pclass  Name    Sex Age SibSp   Parch   Ticket  Fare    Cabin   Embarked

#Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(train.head())

# print("\n\nSummary statistics of training data")
# print(train.describe())

#Any files you save will be available in the output tab below
# train.to_csv('copy_of_the_training_data.csv', index=False)

survived_data = train.iloc[:,1:2].values

pclass = train.iloc[:,2:3].values

data = train.iloc[:,:].values
def statistic(data, survived_col=1, to_judge_col=2):  # nice function to learn about the data!
    counter = Counter()
    total = Counter()
    for i in data:
        if isinstance(i[to_judge_col], float) and isnan(i[to_judge_col]):
            continue
        total[i[to_judge_col]] += 1
        if i[survived_col]==1:
            counter[i[to_judge_col]] += 1
    print 'Survived distribution of column {}:'.format(TRAIN_HEADER[to_judge_col]) 
    for i in counter:
        print '{}: {}----{} of {}'.format(i, 1.0*counter[i]/total[i], counter[i], total[i])

statistic(data, to_judge_col=2)

# data = np.genfromtxt('train.csv')
