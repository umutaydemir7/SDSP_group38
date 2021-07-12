import streamlit as st
import pandas as pd
import numpy as np 
import scipy
import sklearn
import matplotlib
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from collections import Counter
from sklearn.datasets import make_classification
from pandas import DataFrame
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


dataFile = "sdsp_patients.xlsx"

df = pd.read_excel(dataFile)

total = df.isnull().sum().sort_values(ascending=False)

null_count = df.isnull().count()
null= df.isnull().sum()

percentage_missing = (null/null_count).sort_values(ascending=False)
missing_data = pd.concat([total, percentage_missing*100], axis=1, keys=['Total missing', 'Percentage'])

dropped= missing_data[missing_data['Percentage']>1]
x = dropped.index

for i in x:   
    df.drop([i],axis=1, inplace=True)

columns = df.columns
for i in columns:
    if(len( df[i].unique()) < 2):
        df.drop([i],axis=1, inplace=True)

matplotlib_axes_logger.setLevel('ERROR')
corralation = df.corr().round(2)

most_common_value= df['Feature_47'].value_counts().idxmax()
df.loc[df['Feature_47'].str.contains(' ' , regex=True, na=False), 'Feature_47'] = most_common_value

most_common_value= df['Feature_48'].value_counts().idxmax()
df.loc[df['Feature_48'].str.contains(' ' , regex=True, na=False), 'Feature_48'] = most_common_value

most_common_value= df['Feature_49'].value_counts().idxmax()
df.loc[df['Feature_49'].str.contains(' ' , regex=True, na=False), 'Feature_49'] = most_common_value

most_common_value= df['Feature_50'].value_counts().idxmax()
df.loc[df['Feature_50'].str.contains(' ' , regex=True, na=False), 'Feature_50'] = most_common_value


columns = df.columns
for i in columns:
    if(len( df[i].unique()) == 2):
        df.loc[df[i].str.contains('No' , regex=True, na=False), i] = 0
        df.loc[df[i].str.contains('Yes' , regex=True, na=False), i] = 1


df.loc[df['Feature_1'].str.contains('Male' , regex=True, na=False), 'Feature_1'] = 0
df.loc[df['Feature_1'].str.contains('Female' , regex=True, na=False), 'Feature_1'] = 1

df.loc[df['Feature_29'].str.contains('No Difference' , regex=True, na=False), 'Feature_29'] = 0
df.loc[df['Feature_29'].str.contains('Evenings' , regex=True, na=False), 'Feature_29'] = 1
df.loc[df['Feature_29'].str.contains('Mornings' , regex=True, na=False), 'Feature_29'] = 2

df.loc[df['Feature_28'].str.contains('Every Day' , regex=True, na=False), 'Feature_28'] = 0
df.loc[df['Feature_28'].str.contains('1-2 Days a Week' , regex=True, na=False), 'Feature_28'] = 1
df.loc[df['Feature_28'].str.contains('3-4 Days a Week' , regex=True, na=False), 'Feature_28'] = 2
df.loc[df['Feature_28'].str.contains('1-2 Days a Month' , regex=True, na=False), 'Feature_28'] = 3

most_common_value= df['Feature_28'].value_counts().idxmax()
df['Feature_28'].fillna(most_common_value, inplace=True)

most_common_value= df['Feature_3'].value_counts().idxmax()
df.loc[df['Feature_3'].str.contains(' ' , regex=True, na=False), 'Feature_3'] = most_common_value
df['Feature_3'] = pd.to_numeric(df['Feature_3'])

df.loc[df['Disease'].str.contains('Disease_1' , regex=True, na=False), 'Disease'] = 0
df.loc[df['Disease'].str.contains('Disease_2' , regex=True, na=False), 'Disease'] = 1
df.loc[df['Disease'].str.contains('Disease_3' , regex=True, na=False), 'Disease'] = 2
df.loc[df['Disease'].str.contains('Disease_4' , regex=True, na=False), 'Disease'] = 3


X = df.iloc[:,1:]  
y = df.iloc[:,0]   

columns = X.columns
for i in columns:
    if(X[i].dtype == object):       
        X[i] = pd.to_numeric(X[i])

y=y.astype(int)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=43)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores=featureScores.sort_values(by='Score', ascending=False)

total= featureScores['Score'].sum()
featureScores['Score']= (100 / total) * featureScores['Score']
count = 0
score=0
for i in featureScores['Score']:
    if(score < 95 ):
        score +=i
        count += 1
    else:
        break

df2 =featureScores.iloc[count:,0]

for feature in list(df2):
    X.drop([feature],axis=1, inplace=True)

columns = X.columns
for i in columns:
    if(len( X[i].unique()) > 4):       
        X[i]=(X[i]-X[i].mean()) / X[i].std()  


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(19,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Transform into a categorical variable
df.Disease = pd.Categorical(df.Disease)

# Assign a number to each category (label encoding)
df.Disease = df.Disease.cat.codes 

# Import to_categorical from keras utils module
from keras.utils.np_utils import to_categorical

y = to_categorical(df.Disease)

st.title("We are helping physicians with their diagnosis by using machine learning")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("\n\n\n\nPlease provide the features below, and click the predict button")
df_new = dict()

for x in X.keys():
    
    if x == "Feature_1":      
        df_new[x] = st.selectbox("Select " + x, ("Male", "Female"))

        if df_new[x] == "Male":
            df_new[x] = 0
        elif df_new[x] == "Female":
            df_new[x] = 1        
        
    elif x == "Feature_2" or x =="Feature_3" or x =="Feature_4" or x =="Feature_5":
        df_new[x] = st.slider("Select " + x, 0, 999) 
        
    elif x == "Feature_28":
        df_new[x] = st.selectbox("Select " + x, ("Every Day", "1-2 Days A Week", "3-4 Days A Week","1-2 Days a Month"))
        if df_new[x] == "Every Day":
            df_new[x] = 0
        elif df_new[x] == "1-2 Days A Week":
            df_new[x] = 1
        elif df_new[x] == "3-4 Days A Week":
            df_new[x] = 2 
        elif df_new[x] == "1-2 Days a Month":
            df_new[x] = 3

    elif x == "Feature_29":
        df_new[x] = st.selectbox("Select " + x, ("No Difference", "Evenings", "Mornings"))
        if df_new[x] == "No Difference":
            df_new[x] = 0
        elif df_new[x] == "Evenings":
            df_new[x] = 1
        elif df_new[x] == "Mornings":
            df_new[x] = 2 

    else:
        df_new[x] = st.selectbox("Select " + x, ("Yes", "No"))
        
        if df_new[x] == "No":
            df_new[x] = 0
        elif df_new[x] == "Yes":
            df_new[x] = 1        
 
if st.button("Predict"):
    for key in X.keys():
        if key == "Feature_2" or key =="Feature_3" or key =="Feature_4" or key =="Feature_5":
            if(df_new[key] > 4):                          
                df_new[key]=(df_new[x]-df[x].mean()) / df[x].std()                   
    test_data = pd.DataFrame(df_new, index=[0])       
    preds = model.predict(test_data)   
    disease_numbers = [i+1 for i in range(preds.size)]
    disease_dict = dict(zip(preds[0], disease_numbers))
    sorted = sorted(disease_dict.keys())
    keys_list = list(sorted)
    st.write("Possibilities of Diseases")
    for i in range(3) :  
        st.write("Disease ", str(disease_dict[sorted[3-i]]), " : ", str(sorted[3-i]))
