# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:38:41 2018

@author: AkashSrivastava
"""

import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from bokeh.io import show
from bokeh.plotting import figure
from sklearn.model_selection import train_test_split
import os



#################################################     Functions to be used in the program     ####################
# function to plot attributes with target attribute
def plot_function(feature,data,title,title1):
    a=data[feature].unique()
    a=pd.DataFrame(a)
    a.columns=['categories']
    a[feature]=a.index
    sum_list=[]
    for ab in a['categories']:
        ck=0
        nck=0
        for x in range(len(ploting_df)):
            if ab==data[feature][x] and ploting_df['Target class'][x]=='ckd':
                ck=ck+1
            elif ab==data[feature][x] and ploting_df['Target class'][x]=='notckd':
                nck=nck+1
        sum_list.append([ab,ck,nck])   
    sum_list=pd.DataFrame(sum_list)
    sum_list.columns=['categories','CK','NCK'] 
    
    p = figure(x_range=list(data[feature].unique()), plot_height=250, plot_width=500, title=title)
    p.vbar(x=list(data[feature].unique()), top=sum_list['CK'], width=.5, color='blue')
    p.xaxis.axis_label = feature
    p.yaxis.axis_label = 'count'
    show(p)


    p = figure(x_range=list(data[feature].unique()), plot_height=250, plot_width=500, title=title1)
    p.vbar(x=list(data[feature].unique()), top=sum_list['NCK'], width=.5, color='red')
    p.xaxis.axis_label = feature
    p.yaxis.axis_label = 'count'

    show(p)


###function to create model
def model_creation():
    #defining model	
    model = keras.Sequential()
    model.add(keras.layers.Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compiling model
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.01), metrics=['accuracy'])
    return model





########################################################      Importing Data        ##################################################
# loading the dataset
dirname = os.getcwd() 
filename = os.path.join(dirname, 'chronic_kidney_disease.csv')

df=pd.read_csv(filename)
df.head()
df.describe()



########################################################     cleaning dataset      #########################################

##handling the null values

df.isnull().sum()


#since red blood cell, red blood cell count, white blood cell count attribute has lot of null values therefore dropiing this attribute
df=df.drop("red blood cell", axis=1)
df=df.drop("red blood cell count", axis=1)
df=df.drop("white blood cell count", axis=1)

#replacing null values in numeric attributes with their median
df['age']=df['age'].fillna(df['age'].median())
df['blood pressure']=df['blood pressure'].fillna(df['blood pressure'].median())
df['Specific gravity']=df['Specific gravity'].fillna(df['Specific gravity'].median())
df['albumin']=df['albumin'].fillna(df['albumin'].median())
df['sugar']=df['sugar'].fillna(df['sugar'].median())
df['blood glucose random']=df['blood glucose random'].fillna(df['blood glucose random'].median())
df['blood urea']=df['blood urea'].fillna(df['blood urea'].median())
df['serum creatinine']=df['serum creatinine'].fillna(df['serum creatinine'].median())
df['sodium']=df['sodium'].fillna(df['sodium'].median())
df['potassium']=df['potassium'].fillna(df['potassium'].median())
df['hemoglobin']=df['hemoglobin'].fillna(df['hemoglobin'].median())
df['packed cell volume']=df['packed cell volume'].fillna(df['packed cell volume'].median())



# replacing null values in non numeric attributes to unknown

df['pus cell']=df['pus cell'].fillna("unknown")
df['pus cell clumps']=df['pus cell clumps'].fillna("unknown")
df['bacteria']=df['bacteria'].fillna("unknown")
df['hypertension']=df['hypertension'].fillna("unknown")
df['diabetes mellitus']=df['diabetes mellitus'].fillna("unknown")
df['coronary artery disease']=df['coronary artery disease'].fillna("unknown")
df['appetite']=df['appetite'].fillna("unknown")
df['pedal edema']=df['pedal edema'].fillna("unknown")
df['anemia']=df['anemia'].fillna("unknown")


df.isnull().sum()

#taking copy of the dataframe for plotting purpose
ploting_df=df.copy()


#converting non numeric attributes to numeric

lb=LabelEncoder()
df['pus cell']=lb.fit_transform(df['pus cell'])
df['pus cell clumps']=lb.fit_transform(df['pus cell clumps'])
df['bacteria']=lb.fit_transform(df['bacteria'])
df['hypertension']=lb.fit_transform(df['hypertension'])
df['diabetes mellitus']=lb.fit_transform(df['diabetes mellitus'])
df['coronary artery disease']=lb.fit_transform(df['coronary artery disease'])
df['appetite']=lb.fit_transform(df['appetite'])
df['pedal edema']=lb.fit_transform(df['pedal edema'])
df['anemia']=lb.fit_transform(df['anemia'])
df['Target class']=lb.fit_transform(df['Target class'])

#findning correlation between the attributes

correlation=df.corr()
print(correlation['Target class'])




# plotting correlation of Target class with other attrubutes
p = figure(x_range=list(df), plot_height=250, plot_width=1500, title="Correlation of target class with other attributes")
p.vbar(x=list(df), top=correlation['Target class'], width=.5)
show(p)



# removing the attributes that have very low correlation(<|0.3|) with the target class
df=df.drop('age',axis=1)
df=df.drop('blood pressure',axis=1)
df=df.drop('pus cell',axis=1)
df=df.drop('pus cell clumps',axis=1)
df=df.drop('bacteria',axis=1)
df=df.drop('serum creatinine',axis=1)
df=df.drop('potassium',axis=1)
df=df.drop('coronary artery disease',axis=1)


########################################################       Data Visualization      #############################################
#plotting the graph of hypertension with target attribute
plot_function('hypertension',ploting_df,"People with Kidney Disease and hypertension","People with hypertension and without Kidney Disease ")


#plotting the graph of diabetes with target attribute
plot_function('diabetes mellitus',ploting_df,"People with Kidney Disease and diabetes mellitus","People with diabetes mellitus and without Kidney Disease")


##plotting the graph of appetite with target attribute
plot_function('appetite',ploting_df,"People with Kidney Disease","People without Kidney Disease")


#plotting the graph of pedal edema with target attribute
plot_function('pedal edema',ploting_df,"People with Kidney Disease and pedal edema","People with pedal edema and without Kidney Disease")


#plotting the graph of anemia with target attribute
plot_function('anemia',ploting_df,"People with Kidney Disease and anemia","People with anemia and without Kidney Disease")

#######################################################    Model Building   ########################################## 

#dividing training data and testing data 
y=df['Target class']
df=df.drop('Target class', axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(df,y, test_size=0.25)

# evaluate model with standardized dataset

mod=model_creation()
mod_data=mod.fit(X_train.values,Y_train.values, epochs=10)



########################################################    Model Testing   #######################################
loss,accuracy=mod.evaluate(X_test.values,Y_test.values)
print("loss for testing set : ",loss)
print("Accuracy for testing set : ",accuracy*100,'%')


#plotting accuracy vs epoch
mod_data.history['acc']
l=[]
for a in range(len(mod_data.history['acc'])):
    l.append(str(a+1))
p = figure(x_range=l, plot_height=250, plot_width=1000, title="Accuracy/Loss vs epoch")
p.line(l,mod_data.history['acc'],color='green', legend='Accuracy')

#plotting loss vs epoch
l=[]
for a in range(len(mod_data.history['loss'])):
    l.append(str(a+1))

p.line(l,mod_data.history['loss'],color='red', legend='Loss')
p.xaxis.axis_label = 'Epochs'
p.yaxis.axis_label = 'Range'
show(p)







