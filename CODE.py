#!/usr/bin/env python
# coding: utf-8

# # Predicting Hotel Booking Cancellation

# In[138]:


# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[139]:


# let try to load the data
df = pd.read_csv('/Users/waadalotaibi/Desktop/DS/HotelData11.csv')


# In[140]:


# lets try to look the data
df.head()


# In[141]:


# lets try to check the general information of all columns in dataset
df.info()


# In[142]:


# lets try to check the shape of data
df.shape


# In[143]:


type(df)


# In[144]:


# check the column names of data
print(df.columns.tolist())


# In[145]:


'''
The describe method shows basic statistical characteristics of each numerical feature (int64 and float64 types):
number of non-missing values, mean, standard deviation,range, median, 0.25 and 0.75 quartiles. 

'''
df.describe().T


# In[146]:


#non-numeric
df.describe(include=['bool','object']).T


# In[147]:


#check the missin value
df.isna().head()


# In[148]:


# check the distribution of target column booking_status 
target_var = pd.DataFrame(df['booking_status'].value_counts()).reset_index()
target_var_percenatage = pd.DataFrame(df['booking_status'].value_counts(normalize=True)).reset_index()

target_var.columns = ['booking_status','count']
target_var_percenatage.columns = ['booking_status','percentage']
target_var['percentage'] = target_var_percenatage['percentage']
descending_order = df['booking_status'].value_counts().sort_values(ascending=False).index
print(target_var)


# In[149]:


#visulaing the target columns distribution 
pd.DataFrame(df['booking_status'].value_counts())
plt.figure(figsize=(10,5))
sns.countplot(x = 'booking_status', data=df, order = descending_order)
plt.xticks()
plt.show()


# In[150]:


# checking the target columns distribution through Pie chart 
explode = (0.1, 0.0)
colors = ( "orange", "cyan")
wp = { 'linewidth' : 1, 'edgecolor' : "green" } # Wedge properties
df['booking_status'].value_counts().plot(kind="pie", figsize=(5,5),startangle=90, shadow=True,autopct="%1.1f%%",explode = explode,
                                   colors=colors,wedgeprops = wp)
plt.show()


# In[151]:


# Overall idea about distribution of data histogram numeric columns
df.hist(bins=40, figsize=(20,15))
plt.show()


# In[152]:


# Distribution of categorical variables
import matplotlib.pyplot as plt

categorical_features = df.select_dtypes(include=['object']).columns

fig, ax = plt.subplots(1, len(categorical_features), figsize=(25, 4))
for i, categorical_feature in enumerate(df[categorical_features]):
    df[categorical_feature].value_counts().plot(kind="bar", ax=ax[i], color='purple').set_title(categorical_feature, weight='bold')


# # Exploratory Data Analysis (EDA)
# 1.What are the busiest months in the hotel? ...
# 

# In[153]:


# lets try to get the dataframe where booking status is not canceled
dff = df[df['booking_status']=='Not_Canceled']

busiest_month_df = dff.groupby('arrival_month')['booking_status'].size().reset_index()
busiest_month_df.set_index('arrival_month', inplace=True)
busiest_month_df = busiest_month_df['booking_status'].sort_values(ascending=False)
busiest_month_df.reset_index()


# In[154]:


# visualize the booking status count on each month
busiest_month_df.plot(kind='bar', figsize=(10,5), color='y')
plt.show()


# # 2. Which market segment do most of the guests come from?

# In[83]:


#get the dataframe where repeated_guest is Yes
dfff = df[df['repeated_guest']==1]
market_segment_type = dfff.groupby('market_segment_type')['repeated_guest'].size().reset_index()
market_segment_type.set_index('market_segment_type', inplace=True)
market_segment_type = market_segment_type['repeated_guest'].sort_values(ascending=False)
market_segment_type.reset_index()


# In[84]:


# visualize the repeted gusts in every market segment type
market_segment_type.plot(kind='bar', figsize=(15,5))
plt.show()


# # 3.What are the differences in room prices in different market segments?

# Hotel rates are dynamic and change according to demand and customer demographics.

# In[161]:


# lets try to get the dataframe 
market_segment_type_room = df.groupby('market_segment_type')['avg_price_per_room'].size().reset_index()
market_segment_type_room.set_index('market_segment_type', inplace=True)
market_segment_type_room = market_segment_type_room['avg_price_per_room'].sort_values(ascending=False)
market_segment_type_room.reset_index()


# In[166]:


market_segment_type_room.plot(kind='bar', figsize=(15,5))
plt.show()


# # 4. What percentage of bookings are canceled?

# In[88]:


cancelled_booking_perc = round(df['booking_status'].value_counts(normalize=True)[1],4)
print(f"{cancelled_booking_perc*100}% percentage of bookings are canceled.")


# # 5.What percentage of repeating guests cancel?

# Repeating guests are the guests who stay in the hotel often and are important to brand equity.
# 
# 

# In[89]:


# lets try to get the dataframe where booking status is canceled
dff_canc = df[(df['booking_status']=='Canceled')]

cancelled_booking_perc = round(dff_canc['repeated_guest'].value_counts(normalize=True)[1],4)
print(f"{cancelled_booking_perc*100}% percentage of repeated_guest are canceled.")


# # Data Preprocessing

# In[90]:


# checking the missing values in dataset
df.isnull().sum()


# In[91]:


# remove the irrelevent columns that are not useful for the model prediction
df.drop(['arrival_year','arrival_month','arrival_date'], axis = 1, inplace=True)


# In[92]:


#Transform non-numerical labels to numerical labels
from sklearn.preprocessing import StandardScaler, LabelEncoder
labelEnc = LabelEncoder()
for i in list(df.columns):
    if df[i].dtype == 'object':
        df[i] = labelEnc.fit_transform(df[i])


# In[93]:


## Extract Dependent and Independent Variables
X = df[['no_of_adults', 'no_of_weekend_nights', 'no_of_week_nights',
       'type_of_meal_plan', 'room_type_reserved', 'lead_time',
       'market_segment_type', 'avg_price_per_room', 'no_of_special_requests']]
Y = df[['booking_status']]


# In[94]:


# independent variables
X.head()


# In[95]:


# dependent variable
Y.head()


# # Split Train Test

# In[96]:


#splitting the dataset 80% for training and 20% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)


# In[97]:


#to check the shape of training and testing
print("Training Shape :",X_train.shape)
print("Testing Shape :",X_test.shape)


# In[98]:


# standardization our data that better for model prediction because values lies in a specific range (0,1)
st_x= StandardScaler()  
X_train= st_x.fit_transform(X_train) 
X_test= st_x.transform(X_test) 


# ## Importing Libraries

# In[167]:


# Import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
from tabulate import tabulate


# In[168]:


# import methods for measuring accuracy, precision, recall etc
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,  
)


# In[169]:


# function for evaluation metrics precision, recall, f1 etc
def modelEvaluation(predictions, y_test_set, model_name):
    # Print model evaluation to predicted result    
    print("==========",model_name,"==========")
    print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test_set, predictions)))
    print ("Precision on validation set: {:.4f}".format(precision_score(y_test_set, predictions)))    
    print ("Recall on validation set: {:.4f}".format(recall_score(y_test_set, predictions)))
    print ("F1_Score on validation set: {:.4f}".format(f1_score(y_test_set, predictions)))
    print ("\nClassification report : \n", classification_report(y_test_set, predictions))
    print ("\nConfusion Matrix : \n", confusion_matrix(y_test_set, predictions))
    results = [accuracy_score(y_test_set, predictions),precision_score(y_test_set, predictions),
              recall_score(y_test_set, predictions),f1_score(y_test_set, predictions)]
    return results


# # Model building - KNN

# In[170]:


#Create KNN Classifier
knn_model = KNeighborsClassifier()
#Train the model using the training sets
knn_model.fit(X_train, Y_train) 
#Predict the response for test dataset
pred_knn = knn_model.predict(X_test) 

# Print model evaluation to predicted result    
results_knn = modelEvaluation(pred_knn, Y_test, 'k-nearest neighbours') 


# In[171]:


#Convert the target variable into numeric format 0 or 1
classes = ['1','0']
    
#plot a confusion matrix to visualize the classifer's performance
sns.heatmap(confusion_matrix(Y_test, pred_knn),annot=True,yticklabels=classes
               ,xticklabels=classes,cmap='Blues', fmt='g')
plt.tight_layout()
plt.show()


# # Model building - SVM

# In[172]:


#Create a svm Classifier
svc_model = SVC()
#Train the model using the training sets
svc_model.fit(X_train, Y_train)
#Predict the response for test dataset
pred_svc = svc_model.predict(X_test)

# Print model evaluation to predicted result  
results_svc = modelEvaluation(pred_svc, Y_test, 'Support Vector Machine')


# In[173]:


#Convert the target variable into numeric format 0 or 1
classes = ['1','0']

#plot a confusion matrix to visualize the classifer's performance
sns.heatmap(confusion_matrix(Y_test, pred_svc),annot=True,yticklabels=classes
           ,xticklabels=classes,cmap='Greens', fmt='g')
plt.tight_layout()
plt.show()


# # Model building -  Random Forest

# In[174]:


#Create a RandomForest Classifier
rf_model = RandomForestClassifier()
#Train the model using the training sets
rf_model.fit(X_train, Y_train)
#Predict the response for test dataset
pred_rf = rf_model.predict(X_test)

# Print model evaluation to predicted result  
results_rf = modelEvaluation(pred_rf, Y_test, 'Random Forest')


# In[175]:


#Convert the target variable into numeric format 0 or 1
classes = ['1','0']

#plot a confusion matrix to visualize the classifer's performance
sns.heatmap(confusion_matrix(Y_test, pred_rf),annot=True,yticklabels=classes
               ,xticklabels=classes,cmap='Reds', fmt='g')
plt.tight_layout()
plt.show()


# # Model building - Decision Tree

# In[176]:


#Create a DecisionTree Classifier
dt_model = DecisionTreeClassifier()
#Train the model using the training sets
dt_model.fit(X_train, Y_train)
#Predict the response for test dataset
pred_dt = dt_model.predict(X_test)

# Print model evaluation to predicted result  
results_dt = modelEvaluation(pred_dt, Y_test, 'Decision tree')


# In[177]:


#Convert the target variable into numeric format 0 or 1
classes = ['1','0']

#plot a confusion matrix to visualize the classifer's performance
sns.heatmap(confusion_matrix(Y_test, pred_dt),annot=True,yticklabels=classes
               ,xticklabels=classes,cmap='BuPu', fmt='g')
plt.tight_layout()
plt.show()


# # Model Performance Evaluation

# In[178]:


# showing all models result
dic = {
    'Metrics':['accuracy','precision','recall','f1-score'],
    'decision tree' : results_dt,
    'k-nearest neighbours' : results_knn,
    'support vector machine' : results_svc,
    'random forest' : results_rf,
}
metrics_df = pd.DataFrame(dic)
metrics_df = metrics_df.set_index('Metrics')
print(tabulate(metrics_df, headers = 'keys', tablefmt = 'psql'))


# In[179]:


#visualise the results from all models 
metrics_df.plot(kind='bar', figsize=(22,8))
plt.show()


# **We can see in above table Random forest giving a good score as compare to others.**
