
## Importing necessary libraries


import numpy as np
import pandas as pd
import os
from numpy import linalg as np_la
import warnings
warnings.filterwarnings("ignore")


data=pd.read_csv('../data/AllBooks_baseline_DTM_Labelled.csv')            #Import the "Religious text" dataset
data=data.drop([13], axis=0)                                      # Removing the 13th row
data_labels=np.array(data['Unnamed: 0'])   

#loop to remove characters after '_' in the class label

i=0;
for item in data['Unnamed: 0']:
    x=item.find("_")
    data_labels[i]=item[0:x]
    i=i+1
    
#replace the class labels with the new one (data1)
data=data.drop(['Unnamed: 0'], axis=1)
data.insert(0, 'Unnamed: 0', data_labels)

# Part-A(2)

data_labels=np.array(data['Unnamed: 0'])
df=data.drop(['Unnamed: 0'], axis=1)     #drop the class labels (not required for tf-idf values calculation)


## TF-IDF using scikit learn

from sklearn.feature_extraction.text import TfidfTransformer          #import necessary scikit lern libraries

tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(df)

tf_idf_vector = tfidf_transformer.transform(df)

feature_names = np.array(df.columns)
dataframes = []

for i in range(len(df)):                                       #loop to find tf-idf of each data point in the dataset
    document_vector=tf_idf_vector[i]
    dataframe = pd.DataFrame(document_vector.T.todense(), index=feature_names, columns=[data_labels[i]])
    dataframe = dataframe/np_la.norm(document_vector.T.todense())
    dataframes.append(dataframe.T)
    
df_tfidf=pd.concat(dataframes)
df_tfidf.to_csv("../data/TF_IDF.csv")                                  #export the tf-idf csv file to desired location

