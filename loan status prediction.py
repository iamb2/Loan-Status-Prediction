import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
%matplotlib inline 
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")
#reading the data
train=pd.read_csv("train_u6lujuX_CVtuZ9i.csv") 
test=pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")
#making a copy of original data
train_original=train.copy() 
test_original=test.copy()
#understading data to see what is target variable-loan status is the target variable
train.columns
test.columns
#dimensions of data sets
train.shape
test.shape
#univariate analysis
train['Loan_Status'].value_counts()
# Normalize can be set to True to print proportions instead of number 
train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar()
#visualizing categorical,ordinal and numerical data variables
#1. categoricalvalues (values that can categorize )
plt.figure(1)
 plt.subplot(221) 
 train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
plt.subplot(222)
 train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223)
 train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224)
 train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()
#2.ordinal(values that have an order)
plt.figure(1)
 plt.subplot(131)
  train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') 
plt.subplot(132)
 train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 
plt.subplot(133)
 train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 
plt.show()
#3.numerical variables
plt.figure(1) 
plt.subplot(121)
 sns.distplot(train['ApplicantIncome']); 
plt.subplot(122)
 train['ApplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()
#doing this to see outliers
train.boxplot(column='ApplicantIncome', by = 'Education') plt.suptitle("")
#Let’s look at the Coapplicant income distribution.
plt.figure(1) plt.subplot(121) sns.distplot(train['CoapplicantIncome']); 
plt.subplot(122) train['CoapplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()
#Let’s look at the distribution of LoanAmount variable.

plt.figure(1) plt.subplot(121) df=train.dropna() sns.distplot(train['LoanAmount']); 
plt.subplot(122) train['LoanAmount'].plot.box(figsize=(16,5)) 
plt.show()
#bivariate analysis
#Categorical Independent Variable vs Target Variable
Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
#Now let us visualize the remaining categorical variables vs target variable.

Married=pd.crosstab(train['Married'],train['Loan_Status'])
 Dependents=pd.crosstab(train['Dependents'],train['Loan_Status']) 
 Education=pd.crosstab(train['Education'],train['Loan_Status']) 
 Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show() 
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
 plt.show() 
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
 plt.show()
 #Now we will look at the relationship between remaining categorical independent variables and Loan_Status.

Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
 Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status']) 
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
 plt.show()
 #numericalvariable vs targetvariabe
 train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
 #make salary bins for above line for more detail
 bins=[0,2500,4000,6000,81000]
 group=['Low','Average','High', 'Very high']
 train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
plt.ylabel('Percentage')
#We will analyze the coapplicant income and loan amount variable in similar manner.

bins=[0,1000,3000,42000]
 group=['Low','Average','High'] t
 rain['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
 Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
 plt.ylabel('Percentage')
 #Let us combine the Applicant Income and Coapplicant Income and see the combined effect of Total Income on the Loan_Status.

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
 group=['Low','Average','High', 'Very high'] train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
 plt.xlabel('Total_Income') 
 plt.ylabel('Percentage')
 #Let’s visualize the Loan amount variable.

bins=[0,100,200,700] 
group=['Low','Average','High'] train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
 LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
  plt.xlabel('LoanAmount') 
   plt.ylabel('Percentage')
   #We will replace N with 0 and Y with 1.Let’s drop the bins which we created for the exploration part.
   # We will change the 3+ in dependents variable to 3 to make it a numerical variable
   #We will also convert the target variable’s categories into 0 and 1 so that we can find its correlation with numerical variables. 
   #One more reason to do so is few models like logistic regression takes only numeric values as input. 
   train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
train['Dependents'].replace('3+', 3,inplace=True) test['Dependents'].replace('3+', 3,inplace=True) 
train['Loan_Status'].replace('N', 0,inplace=True) train['Loan_Status'].replace('Y', 1,inplace=True)
# use the heat map to visualize the correlation.
# Heatmaps visualize data through variations in coloring. 
#The variables with darker color means their correlation is more.
matrix = train.corr() f, ax = plt.subplots(figsize=(9, 6)) sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
#We see that the most correlated variables are (ApplicantIncome - LoanAmount) and (Credit_History - Loan_Status).
# LoanAmount is also correlated with CoapplicantIncome.
#missing value and outlier treatment
#find missing values
train.isnull().sum()
#Missing value imputation
#For numerical variables: imputation using mean or median
#For categorical variables: imputation using mode
#There are very less missing values in Gender, Married, Dependents, Credit_History and Self_Employed features so we can fill them using the mode of the features.

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
 train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
 train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
 train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
 #Now let’s try to find a way to fill the missing values in Loan_Amount_Term. We will look at the value count of the Loan amount term variable.

train['Loan_Amount_Term'].value_counts()
#360.0    512 180.0     44 480.0     15 300.0     13 84.0       4 240.0      4 120.0      3 36.0       2 60.0       2 12.0       1 Name: Loan_Amount_Term, dtype: int64
#It can be seen that in loan amount term variable, the value of 360 is repeating the most. So we will replace the missing values in this variable using the mode of this variable.

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
# the LoanAmount variable We will use median to fill the null values as earlier,
# we saw that loan amount have outliers so the mean will not be the proper approach 
#as it is highly affected by the presence of outliers.
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
#Now lets check whether all the missing values are filled in the dataset.

train.isnull().sum()
#as all missing values are filled lets do the same with test data
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
 test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
 #there are outliers in loan amout as seen in univariate analysis so we apply log transformation 
 # to achieve something similar to a normal distribution to both train and test
 train['LoanAmount_log'] = np.log(train['LoanAmount'])
  train['LoanAmount_log'].hist(bins=20)
   test['LoanAmount_log'] = np.log(test['LoanAmount'])
#next  Let’s build a logistic regression model and make predictions for the test dataset.
#Logistic Regression is a classification algorithm. It is used to predict a binary outcome
# (1 / 0, Yes / No, True / False) given a set of independent variables.
#drop the Loan_ID variable as it do not have any effect on the loan status
train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)
#Sklearn requires the target variable in a separate dataset.
# So, we will drop our target variable from the train dataset and save it in another dataset.
X = train.drop('Loan_Status',1) 
y = train.Loan_Status
#Dummy variable turns categorical variables into a series of 0 and 1, making them lot easier to quantify and compare
X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)
#We will use the train_test_split function from sklearn to divide our train dataset. So, first let us import train_test_split.

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
#import LogisticRegression and accuracy_score from sklearn and fit the logistic regression model.

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
	intercept_scaling=1, max_iter=100, multi_class='ovr',
 n_jobs=1, penalty='l2',random_state=1, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
#Let’s predict the Loan_Status for validation set and calculate its accuracy.

pred_cv = model.predict(x_cv)
#Let us calculate how accurate our predictions are by calculating the accuracy.

accuracy_score(y_cv,pred_cv)
#0.7945945945945946
#So our predictions are almost 80% accurate, i.e. we have identified 80% of the loan status correctly.

#Let’s make predictions for the test dataset.

pred_test = model.predict(test)
#Lets import the submission file which we have to submit on the solution checker.

submission=pd.read_csv("Sample_Submission_ZAuTl8O_FK3zQHh.csv")
#We only need the Loan_ID and the corresponding Loan_Status for the final submission. we will fill these columns with the Loan_ID of test dataset and the predictions that we made, i.e., pred_test respectively.

submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']
#Remember we need predictions in Y and N. So let’s convert 1 and 0 to Y and N.

submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)
#Finally we will convert the submission to .csv format and make submission to check the accuracy on the leaderboard.

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')
#From this submission we got an accuracy of 0.7847 on the leaderboard.
#use validaTION METHODS TO improve accuracy of the model
#Let’s import StratifiedKFold from sklearn and fit the model.

from sklearn.model_selection import StratifiedKFold
#Now let’s make a cross validation logistic model with stratified 5 folds and make predictions for test dataset.

i=1 kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
 for train_index,test_index in kf.split(X,y):     print('\n{} of kfold {}'.format(i,kf.n_splits))   
xtr,xvl = X.loc[train_index],X.loc[test_index] 
ytr,yvl = y[train_index],y[test_index]      
model = LogisticRegression(random_state=1) 
model.fit(xtr, ytr)    
pred_test = model.predict(xvl)   
score = accuracy_score(yvl,pred_test)  
print('accuracy_score',score)     i+=1 pred_test = model.predict(test) pred=model.predict_proba(xvl)[:,1]
#The mean validation accuracy for this model turns out to be 0.81. Let us visualize the roc curve.

from sklearn import metrics fpr, tpr, _ = metrics.roc_curve(yvl,  pred)
auc = metrics.roc_auc_score(yvl, pred) 
plt.figure(figsize=(12,8)) plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 
plt.xlabel('False Positive Rate') plt.ylabel('True Positive Rate') plt.legend(loc=4) plt.show()

#We got an auc value of 0.77.

submission['Loan_Status']=pred_test submission['Loan_ID']=test_original['Loan_ID']
#Remember we need predictions in Y and N. So let’s convert 1 and 0 to Y and N.

submission['Loan_Status'].replace(0, 'N',inplace=True) submission['Loan_Status'].replace(1, 'Y',inplace=True)
#Lets convert the submission to .csv format and make submission to check the accuracy on the leaderboard.

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Logistic.csv')
#From this submission we got an accuracy of 0.78472 on the leaderboard. Now we will try to improve this accuracy using different approaches.