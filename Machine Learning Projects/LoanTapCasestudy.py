#!/usr/bin/env python
# coding: utf-8

# # Business Case: LoanTap Logistic Regression

# 
# ## Define Problem Statement and perform Exploratory Data Analysis
Context:

LoanTap is an online platform committed to delivering customized loan products to millennials. They innovate in an otherwise dull loan segment, to deliver instant, flexible loans on consumer friendly terms to salaried professionals and businessmen.
The data science team at LoanTap is building an underwriting layer to determine the creditworthiness of MSMEs as well as individuals.
LoanTap deploys formal credit to salaried individuals and businesses 4 main financial instruments:
    1. Personal Loan
    2. EMI Free Loan
    3. Personal Overdraft
    4. Advance Salary Loan
This case study will focus on the underwriting process behind Personal Loan only
# #### Problem Statement:

# Given a set of attributes for an Individual, determine if a credit line should be extended to them. 
# If so, what should the repayment terms be in business recommendations?
Data dictionary:

1. loan_amnt : The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
2. term : The number of payments on the loan. Values are in months and can be either 36 or 60.
3. int_rate : Interest Rate on the loan
4. installment : The monthly payment owed by the borrower if the loan originates.
5. grade : LoanTap assigned loan grade
6. sub_grade : LoanTap assigned loan subgrade
7. emp_title :The job title supplied by the Borrower when applying for the loan.*
8. emp_length : Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means      ten or more years.
9. home_ownership : The home ownership status provided by the borrower during registration or obtained from the credit report.
10. annual_inc : The self-reported annual income provided by the borrower during registration.
11. verification_status : Indicates if income was verified by LoanTap, not verified, or if the income source was verified
12. issue_d : The month which the loan was funded
13. loan_status : Current status of the loan - Target Variable
14. purpose : A category provided by the borrower for the loan request.
15. title : The loan title provided by the borrower
16. dti : A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage     and the requested LoanTap loan, divided by the borrower’s self-reported monthly income.
17. earliest_cr_line :The month the borrower's earliest reported credit line was opened
18. open_acc : The number of open credit lines in the borrower's credit file.
19. pub_rec : Number of derogatory public records
20. revol_bal : Total credit revolving balance
21. revol_util : Revolving line utilization rate, or the amount of credit the borrower is using relative to all available           revolving credit.
22. total_acc : The total number of credit lines currently in the borrower's credit file
23. initial_list_status : The initial listing status of the loan. Possible values are – W, F
24. application_type : Indicates whether the loan is an individual application or a joint application with two co-borrowers
25. mort_acc : Number of mortgage accounts.
26. pub_rec_bankruptcies : Number of public record bankruptcies
27. Address: Address of the individual
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE
import scipy.stats as stats


# In[2]:


data = pd.read_csv('logistic_regression.csv')


# In[3]:


data.head()


# In[4]:


data.dtypes


# In[5]:


data.shape


# There are total 27 Features and 396030 records for a given data set

# In[6]:


data.info()


# From the above info we can see there are null values for emp_title, emp_length, title, revol_util, mort_acc and pub_rec_bankruptcies features

# In[7]:


data.nunique()


# In[8]:


data.describe().round(3).T


# In[9]:


data.describe(include = 'all').T


# In[10]:


data.describe(include=[np.object]).T


# In[11]:


data.duplicated().sum()


# In[12]:


data.isnull().sum()/len(data)*100


# In[13]:


data.pub_rec_bankruptcies.value_counts()


# In[14]:


toCategory = ['term', 'grade', 'sub_grade','emp_length', 'home_ownership', 'verification_status','loan_status','initial_list_status','application_type','pub_rec_bankruptcies']
for x in toCategory:
    data[x] = data[x].astype('category')


# In[15]:


duplicate = data[data.duplicated(keep = 'last')]
duplicate


# In[16]:


for x in toCategory:
    print('\033[1m' + "Unique values for Feature " + x  + '\033[0m')
    print("Number of Unique values: " + str(data[x].nunique()))
    print(data[x].unique())
    print()
    print('\033[1m' + "Value Counts for Feature " + x  + '\033[0m')
    print(data[x].value_counts())
    print()
    print('\033[1m' + "Normalized Value Counts for Feature " + x  + '\033[0m')
    print(data[x].value_counts(normalize=True).round(2))
    print()
    print('\033[1m' + "-------------------------------------------------------------------------------------------" + '\033[0m')
    print()


# In[17]:


loan_Status = pd.DataFrame(data['loan_status'].value_counts(normalize=True) * 100)
loan_Status.reset_index(inplace = True)
loan_Status.columns = ['loan_status', 'percentage']
loan_Status


# In[18]:


graph = sns.barplot(x = loan_Status['loan_status'], y = loan_Status['percentage'])
for i in graph.containers:
    graph.bar_label(i,)
graph.set_title("Loan Status Distribution")
plt.show()

1.	What percentage of customers have fully paid their Loan Amount?

Around 80.3871% Customers fully paid their loan amount and remaining are charged off.
# In[19]:


fig, axs = plt.subplots(2, 2, figsize=(30, 25), sharey=False)
fig.suptitle("Loan Status Distribution" , fontsize=25, fontweight='bold')
grade = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
graph = sns.countplot(x = 'grade', data= data, ax = axs[0][0],palette='bright', order = grade)
graph.set_title("Grade Distibution", fontsize = 25)
graph.set_xlabel('grade', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)


sub_grade = sorted(data.sub_grade.unique().tolist())
graph = sns.countplot(x = 'sub_grade', data = data, ax = axs[0][1], palette='bright', order = sub_grade)
graph.set_title("Sub Grade Distribution", fontsize = 25)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_xlabel('sub_grade', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)

graph = sns.countplot(x = 'term', data= data, ax = axs[1][0],palette='bright')
graph.set_title("Term Distibution", fontsize = 25)
graph.set_xlabel('term', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)

order = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years',]
graph = sns.countplot(x = 'emp_length', data = data, ax = axs[1][1], palette='bright', order = order)
graph.set_title("emp_length Distribution", fontsize = 25)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_xlabel('emp_length', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)

plt.tight_layout()
plt.show()


# In[20]:


fig, axs = plt.subplots(3, 2, figsize=(30, 30), sharey=False)

fig.suptitle("Loan Status Distribution" , fontsize=35, fontweight='bold')
graph = sns.countplot(x = 'home_ownership', data= data, ax = axs[0][0],palette='bright')
graph.set_title("home_ownership Distibution", fontsize = 25)
graph.set_xlabel('home_ownership', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)

graph = sns.countplot(x = 'verification_status', data = data, ax = axs[0][1], palette='bright')
graph.set_title("verification_status Distribution", fontsize = 25)
graph.set_xlabel('verification_status', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)


graph = sns.countplot(x = 'initial_list_status', data= data, ax = axs[1][0],palette='bright')
graph.set_title("initial_list_status Distibution", fontsize = 25)
graph.set_xlabel('initial_list_status', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)

graph = sns.countplot(x = 'application_type', data = data, ax = axs[1][1], palette='bright')
graph.set_title("application_type Distribution", fontsize = 25)
graph.set_xlabel('application_type', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)

graph = sns.countplot(x = 'pub_rec_bankruptcies', data = data, ax = axs[2][0], palette='bright')
graph.set_title("pub_rec_bankruptcies Distribution", fontsize = 25)
graph.set_xlabel('pub_rec_bankruptcies', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)

graph = sns.countplot(x = 'purpose', data = data, ax = axs[2][1], palette='bright')
graph.set_title("purpose Distribution", fontsize = 25)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_xlabel('purpose', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
plt.tight_layout()
plt.show()

The majority of people have home ownership as?

The majority people have home ownership as MORTGAGE. Around 50 Percent of People have MORTAGE type home ownership1. Out of total Customers 76% customers took loan for 36 months while rest of the customers(24%) took loan for 72 months.
2. 29 % of Customers belongs to B grade while 26 % belongs to C grade and remaining 45 % customers belongs to remaining 5 grades.
3. Customers who are employed for 10+ years are 33% followed 9% customers who are employed for 2  years
4. 50 % of total customers who took loan have home_ownership as Mortage, while 40% are in the rent.
5. There are almost equal number of verified, source verified and not verified customers as 35%, 33% and 32% respectively
6. Around 80 % of customers fully paid thier loan and 20 % charged off.
7. Initial listing status of loan is 60% for f and 40 % for w.
8. Almsot 99.82 % Customers have individual application type while remaining has Joint and direct pay application
9. There are 9 unique values of public record bankruptcies. Out of which 90% have no public record of bankrupcies and remaining have some record of bankrupcies.
# In[21]:


def outlierAnalysis(x):
    Q3, Q1 = np.percentile(data[x], [75 ,25])
    IQR = Q3 - Q1
    maxExcludingOutlier = Q3 + 1.5 * IQR
    minExcludingOutlier = Q1 - 1.5 * IQR
    mean = round(data[x].mean(),2)
    median = round(data[x].median(),2)
    mode = round(data[x].mode(), 2)
    print("For the given sample " +x +" Analysis is as Follows: ")
    print("Q1: ", Q1)
    print("Q3: ", Q3)
    print("Mean: ", round(data[x].mean(),2))
    print("Median: ", data[x].median())
    print("Mode: " , round(data[x].mode(), 2))
    print("IQR: " , IQR)
    print("Maximum " + x +" Excluding Outlier: " , maxExcludingOutlier)
    print("Minimum " + x + " Purchase Excluding Outlier: " , minExcludingOutlier)


# In[22]:


def plotfor_continuousdata(x):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
    plt.suptitle(x+" Analysis", fontsize = 15)
    g1 = sns.histplot(data[x], kde = True, edgecolor='green', ax = axs[0, 0], bins = 20)
    g1.axvline(data[x].mean(), ls = '--', color = "red", lw = 2.5, label = "mean")
    g1.axvline(data[x].median(), ls = '--' ,color = 'blue', lw = 2.5, label = 'Median')
    g1.axvline(data[x].mode()[0], ls = '--', color = 'green', lw = 2.5, label = 'Mode')
    g1.legend()
    stats.probplot(data[x], dist="norm", plot=axs[0, 1])
    plt.tight_layout()
    plt.show()


# In[23]:


num = data.select_dtypes(include='number')
for x in num:
    plotfor_continuousdata(x)
    print("-------------------------------------------------------------------------------------------------------------------")


# In[24]:


for x in num:
    print("Statistical Info of ", x)
    print(data[x].describe())
    print("**************************************")
    print("Outlier Analysis for ", x)
    print(outlierAnalysis(x))
    print("######################################################################################################")
    print()


# In[25]:


fig, axs = plt.subplots(4, 2, figsize=(30, 50), sharey=False)
graph = sns.boxplot(x = "loan_amnt", data= data,dodge=False, ax=axs[0][0])
title1 = "Boxplot Analysis for loan amount" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('loan_amnt', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "int_rate", data= data, dodge=False, ax=axs[0][1])
title1 = "Boxplot Analysis for int_rate" 
graph.set_xlabel('int_rate', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)
graph.set_title(title1, fontsize = 25)


graph = sns.boxplot(x = "annual_inc", data= data, dodge=False, ax=axs[1][0])
title1 = "Boxplot Analysis for annual_inc" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('annual_inc', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "dti", data= data, dodge=False, ax=axs[1][1])
title1 = "Boxplot Analysis for dti" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('dti', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "open_acc", data= data, dodge=False, ax=axs[2][0])
title1 = "Boxplot Analysis for open_acc" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('open_acc', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "revol_bal", data= data, dodge=False, ax=axs[2][1])
title1 = "Boxplot Analysis for revol_bal" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('revol_bal', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "revol_util", data= data, dodge=False, ax=axs[3][0])
title1 = "Boxplot Analysis for revol_utils" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('revol_util', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "total_acc", data= data, dodge=False, ax=axs[3][1])
title1 = "Boxplot Analysis for total_acc" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('total_acc', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)
plt.show()


# We can see most of the numerical data we have is right skewed so, outlier treatment will be must

# In[26]:


plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(method='spearman'), annot=True)
plt.show()

Comment about the correlation between Loan Amount and Installment features.

Here, we can see that intsallment and int_rate are highly correlated with each other.
This two features are dependent on each other. So we can remove any one feature from the data set
# In[27]:


data.drop(columns=['installment'], axis=1, inplace=True)


# In[28]:


plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(method='spearman'), annot=True)
plt.show()


# In[29]:


fig, axs = plt.subplots(2, 2, figsize=(30, 25), sharey=False)
fig.suptitle("Loan Status Distribution" , fontsize=35, fontweight='bold')
grade = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
graph = sns.countplot(x = 'grade', hue= 'loan_status', data= data, ax = axs[0][0],palette='bright', order = grade)
graph.set_title("Grade Distibution", fontsize = 25)
graph.set_xlabel('grade', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

sub_grade = sorted(data.sub_grade.unique().tolist())
graph = sns.countplot(x = 'sub_grade', hue = 'loan_status', data = data, ax = axs[0][1], palette='bright', order = sub_grade)
graph.set_title("Sub Grade Distribution", fontsize = 25)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_xlabel('sub_grade', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

graph = sns.countplot(x = 'term', hue= 'loan_status', data= data, ax = axs[1][0],palette='bright')
graph.set_title("Term Distibution", fontsize = 25)
graph.set_xlabel('term', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

order = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years',]
graph = sns.countplot(x = 'emp_length', hue = 'loan_status', data = data, ax = axs[1][1], palette='bright', order = order)
graph.set_title("emp_length Distribution", fontsize = 25)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_xlabel('emp_length', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)
plt.tight_layout()
plt.show()


# In[30]:


table = pd.crosstab(data["grade"],data["loan_status"],normalize='index')
print('contingency_table :\n',table)

People with grades ‘A’ are more likely to fully pay their loan. (T/F)

Out of total customers who belongs to A grade around 94 percent fully paid thier loans. 
So we can say that people who belongs to A grade have high chances of paying thier loans.
# In[31]:


table = pd.crosstab(data["sub_grade"],data["loan_status"],normalize='index')
print('contingency_table :\n',table)


# In[32]:


fig, axs = plt.subplots(3, 2, figsize=(30, 30), sharey=False)
sns.set()
fig.suptitle("Loan Status Distribution" , fontsize=35, fontweight='bold')
graph = sns.countplot(x = 'home_ownership', hue= 'loan_status', data= data, ax = axs[0][0],palette='bright')
graph.set_title("home_ownership Distibution", fontsize = 25)
graph.set_xlabel('home_ownership', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)


sub_grade = sorted(data.sub_grade.unique().tolist())
graph = sns.countplot(x = 'verification_status', hue = 'loan_status', data = data, ax = axs[0][1], palette='bright')
graph.set_title("verification_status Distribution", fontsize = 25)
graph.set_xlabel('verification_status', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

graph = sns.countplot(x = 'initial_list_status', hue= 'loan_status', data= data, ax = axs[1][0],palette='bright')
graph.set_title("initial_list_status Distibution", fontsize = 25)
graph.set_xlabel('initial_list_status', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

graph = sns.countplot(x = 'application_type', hue = 'loan_status', data = data, ax = axs[1][1], palette='bright')
graph.set_title("application_type Distribution", fontsize = 25)
graph.set_xlabel('application_type', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

graph = sns.countplot(x = 'pub_rec_bankruptcies', hue = 'loan_status', data = data, ax = axs[2][0], palette='bright')
graph.set_title("pub_rec_bankruptcies Distribution", fontsize = 25)
graph.set_xlabel('pub_rec_bankruptcies', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

graph = sns.countplot(x = 'purpose', hue = 'loan_status', data = data, ax = axs[2][1], palette='bright')
graph.set_title("purpose Distribution", fontsize = 25)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_xlabel('purpose', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)
plt.tight_layout()
plt.show()


# In[33]:


fig, axs = plt.subplots(4, 2, figsize=(30, 50), sharey=False)
graph = sns.boxplot(x = "loan_amnt", data= data, y = "loan_status",dodge=False, ax=axs[0][0])
title1 = "Boxplot Analysis for loan_amnt and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('loan_amnt', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "int_rate", data= data, y = "loan_status", dodge=False, ax=axs[0][1])
title1 = "Boxplot Analysis for int_rate and loan_status" 
graph.set_xlabel('int_rate', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)
graph.set_title(title1, fontsize = 25)


graph = sns.boxplot(x = "annual_inc", data= data, y = "loan_status", dodge=False, ax=axs[1][0])
title1 = "Boxplot Analysis annual_inc and loan status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('annual_inc', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "dti", data= data, y = "loan_status", dodge=False, ax=axs[1][1])
title1 = "Boxplot Analysis for dti and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('dti', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "open_acc", data= data, y = "loan_status", dodge=False, ax=axs[2][0])
title1 = "Boxplot Analysis for open_acc and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('open_acc', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "revol_bal", data= data, y = "loan_status", dodge=False, ax=axs[2][1])
title1 = "Boxplot Analysis for revol_bal and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('revol_bal', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "revol_util", data= data, y = "loan_status", dodge=False, ax=axs[3][0])
title1 = "Boxplot Analysis for revol_util and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('revol_util', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "total_acc", data= data, y = "loan_status", dodge=False, ax=axs[3][1])
title1 = "Boxplot Analysis for total_acc and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('total_acc', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)
plt.show()


# In[34]:


data.columns


# In[35]:


data.emp_title = data.emp_title.str.lower()


# In[36]:


data.emp_title.value_counts(normalize=True)[:10]


# In[37]:


fig, ax = plt.subplots(figsize=(15, 6))
graph = sns.barplot( x = data.emp_title.value_counts()[:10],y = data['emp_title'].value_counts()[:10].index, palette='bright')
graph.set_title("Top 15 afforded job titles", fontsize = 20)
plt.xlabel("Count")
plt.ylabel("emp_title")
plt.tight_layout()
plt.show()

Name the top 2 afforded job titles:
    1. Manager
    2. Teacher
# In[38]:


data['title'] = data.title.str.lower()


# In[39]:


data.title.value_counts(normalize=True)[:10]

Around 42 % peple put their title as debt consolidation, but thier is non uniformaty with these feature, so we will drop these column in further analysis
# In[40]:


data.mort_acc.value_counts()[:5]


# In[41]:


data.isnull().sum()/len(data)*100


# In[42]:


data['mort_acc'].fillna(np.round(data.mort_acc.mean()), inplace=True)


# In[43]:


# Feature Engineering
# Home Ownership type Other None and Any has very less number of data so convert it to Other
data.home_ownership.replace(['NONE', 'ANY'], 'OTHER', inplace=True)
data.purpose.replace(['moving', 'vacation','house','wedding','renewable_energy','educational'],'other', inplace = True)
data['mort_acc'] = np.where((data.mort_acc >= 1),1,data.mort_acc)
# data.replace({'mort_acc' : { 0 : 0, 1 : 1, 2:2, 3:3}})
data['pub_rec'] = np.where((data.pub_rec >= 1),1,data.pub_rec)
def pub_rec_bankruptcies(val):
    if val == 0.0:
        return 0
    elif val >= 1.0:
        return 1
    else:
        return val
data.pub_rec_bankruptcies = data.pub_rec_bankruptcies.apply(pub_rec_bankruptcies)


# In[44]:


data.isnull().sum()/len(data)*100


# In[45]:


data.emp_length.unique()


# In[46]:


data['emp_length'] = data['emp_length'].str.rstrip(" years")
def emp_length(val):
    if val == "10+":
        return 10
    elif val == "< 1":
        return 0
    elif val in ["1","2", "3", "4", "5", "6", "7", "8", "9"]:
        return int(val)
    else:
        return val


# In[47]:


data['emp_length'] = data.emp_length.apply(emp_length)


# In[48]:


data['emp_length'].fillna(6.0, inplace = True)
data['emp_length'].astype(int)
data.isnull().sum()/len(data)*100


# In[49]:


data.dropna(inplace=True)
data.isnull().sum()/len(data)*100


# In[50]:


data['zip_code'] = data.address.apply(lambda x: x[-5:])
data['zip_code'].value_counts(normalize=True)


# In[51]:


fig, ax = plt.subplots(figsize=(15, 6))
graph = sns.barplot( x = data.zip_code.value_counts(),y = data['zip_code'].value_counts().index, palette='bright')
graph.set_title("Customer Geographical Distribution", fontsize = 20)
plt.xlabel("Count")
plt.ylabel("zip_code")
plt.tight_layout()
plt.show()


# In[52]:


fig, ax = plt.subplots(figsize=(15, 6))
graph = sns.countplot(x = 'zip_code', hue = 'loan_status', data = data, palette='bright')
graph.set_title("zip_code Distribution wrt loan status", fontsize = 25)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_xlabel('zip_code', fontsize=20);
graph.set_ylabel('Count', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.show()


# We Could see location with zip code 11650, 86630, 93700  are charged off, Nopeople from these areas fully paid thier loans. while people wit 05113 00813 and 29597 fully paid thier loans

# In[53]:


fig, ax = plt.subplots(figsize=(15, 6))
emp_length = sorted(data.emp_length.unique().tolist())
graph = sns.countplot(x = 'emp_length', hue = 'loan_status', data = data, palette='bright')
graph.set_title("emp_length Distribution", fontsize = 20)
plt.show()


# In[54]:


data.describe()


# In[55]:


fig, axs = plt.subplots(3, 2, figsize=(30, 30), sharey=False)
sns.set()
fig.suptitle("Loan Status Distribution" , fontsize=35, fontweight='bold')
graph = sns.countplot(x = 'home_ownership', hue= 'loan_status', data= data, ax = axs[0][0],palette='bright')
graph.set_title("home_ownership Distibution", fontsize = 25)
graph.set_xlabel('home_ownership', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)


sub_grade = sorted(data.sub_grade.unique().tolist())
graph = sns.countplot(x = 'verification_status', hue = 'loan_status', data = data, ax = axs[0][1], palette='bright')
graph.set_title("verification_status Distribution", fontsize = 25)
graph.set_xlabel('verification_status', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

graph = sns.countplot(x = 'initial_list_status', hue= 'loan_status', data= data, ax = axs[1][0],palette='bright')
graph.set_title("initial_list_status Distibution", fontsize = 25)
graph.set_xlabel('initial_list_status', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

graph = sns.countplot(x = 'application_type', hue = 'loan_status', data = data, ax = axs[1][1], palette='bright')
graph.set_title("application_type Distribution", fontsize = 25)
graph.set_xlabel('application_type', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

graph = sns.countplot(x = 'pub_rec_bankruptcies', hue = 'loan_status', data = data, ax = axs[2][0], palette='bright')
graph.set_title("pub_rec_bankruptcies Distribution", fontsize = 25)
graph.set_xlabel('pub_rec_bankruptcies', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)

graph = sns.countplot(x = 'purpose', hue = 'loan_status', data = data, ax = axs[2][1], palette='bright')
graph.set_title("purpose Distribution", fontsize = 25)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_xlabel('purpose', fontsize=25);
graph.set_ylabel('Count', fontsize=25);
graph.tick_params(axis='both', which='major', labelsize=25)
graph.legend(fontsize = 25)
plt.tight_layout()
plt.show()


# In[56]:


toCategory.extend(["pub_rec","mort_acc", "pub_rec_bankruptcies"])


# In[57]:


for x in toCategory:
    data[x] = data[x].astype('category')


# In[58]:


numerical_data = data.select_dtypes(include='number')
num_cols = numerical_data.columns
len(num_cols)


# ### Outlier Treatment with respect to Loan Status

# In[59]:


data.groupby(by='loan_status').mean()


# In[60]:


data.groupby(by='loan_status').std()


# In[61]:


upperLimit = data.groupby(by='loan_status').mean() + 3 * data.groupby(by='loan_status').std()
upperLimit


# In[62]:


lowerLimit = data.groupby(by='loan_status').mean() - 3 * data.groupby(by='loan_status').std()
lowerLimit


# In[63]:


data[((data.loan_status == 'Charged Off') & (data['loan_amnt'] > lowerLimit['loan_amnt']["Charged Off"]) & (data['loan_amnt'] < upperLimit['loan_amnt']["Charged Off"])) | ((data.loan_status == 'Fully Paid') & (data['loan_amnt'] > lowerLimit['loan_amnt']["Fully Paid"]) & (data['loan_amnt'] < upperLimit['loan_amnt']["Fully Paid"]))]


# In[64]:


# Outlier Treatment
for x in num_cols:
    data = data[((data.loan_status == 'Charged Off') & (data[x] > lowerLimit[x]["Charged Off"]) & (data[x] < upperLimit[x]["Charged Off"])) | ((data.loan_status == 'Fully Paid') & (data[x] > lowerLimit[x]["Fully Paid"]) & (data[x] < upperLimit[x]["Fully Paid"]))]


# In[65]:


data.shape


# In[66]:


fig, axs = plt.subplots(4, 2, figsize=(30, 50), sharey=False)
graph = sns.boxplot(x = "loan_amnt", data= data,dodge=False, ax=axs[0][0])
title1 = "Boxplot Analysis for loan amount" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('loan_amnt', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "int_rate", data= data, dodge=False, ax=axs[0][1])
title1 = "Boxplot Analysis for int_rate" 
graph.set_xlabel('int_rate', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)
graph.set_title(title1, fontsize = 25)


graph = sns.boxplot(x = "annual_inc", data= data, dodge=False, ax=axs[1][0])
title1 = "Boxplot Analysis for annual_inc" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('annual_inc', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "dti", data= data, dodge=False, ax=axs[1][1])
title1 = "Boxplot Analysis for dti" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('dti', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "open_acc", data= data, dodge=False, ax=axs[2][0])
title1 = "Boxplot Analysis for open_acc" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('open_acc', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "revol_bal", data= data, dodge=False, ax=axs[2][1])
title1 = "Boxplot Analysis for revol_bal" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('revol_bal', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "revol_util", data= data, dodge=False, ax=axs[3][0])
title1 = "Boxplot Analysis for revol_utils" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('revol_util', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "total_acc", data= data, dodge=False, ax=axs[3][1])
title1 = "Boxplot Analysis for total_acc" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('total_acc', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)
plt.show()

After outlier treatment we could see drastic change in the boxplots
# In[67]:


fig, axs = plt.subplots(4, 2, figsize=(30, 50), sharey=False)
graph = sns.boxplot(x = "loan_amnt", data= data, y = "loan_status",dodge=False, ax=axs[0][0])
title1 = "Boxplot Analysis for loan_amnt and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('loan_amnt', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "int_rate", data= data, y = "loan_status", dodge=False, ax=axs[0][1])
title1 = "Boxplot Analysis for int_rate and loan_status" 
graph.set_xlabel('int_rate', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)
graph.set_title(title1, fontsize = 25)


graph = sns.boxplot(x = "annual_inc", data= data, y = "loan_status", dodge=False, ax=axs[1][0])
title1 = "Boxplot Analysis annual_inc and loan status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('annual_inc', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "dti", data= data, y = "loan_status", dodge=False, ax=axs[1][1])
title1 = "Boxplot Analysis for dti and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('dti', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "open_acc", data= data, y = "loan_status", dodge=False, ax=axs[2][0])
title1 = "Boxplot Analysis for open_acc and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('open_acc', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "revol_bal", data= data, y = "loan_status", dodge=False, ax=axs[2][1])
title1 = "Boxplot Analysis for revol_bal and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('revol_bal', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "revol_util", data= data, y = "loan_status", dodge=False, ax=axs[3][0])
title1 = "Boxplot Analysis for revol_util and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('revol_util', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)

graph = sns.boxplot(x = "total_acc", data= data, y = "loan_status", dodge=False, ax=axs[3][1])
title1 = "Boxplot Analysis for total_acc and loan_status" 
graph.set_title(title1, fontsize = 25)
graph.set_xlabel('total_acc', fontsize=20);
graph.set_ylabel('loan_status', fontsize=20);
graph.tick_params(axis='both', which='major', labelsize=20)
plt.show()


# In[68]:


data.earliest_cr_line.value_counts()


# In[69]:


data.drop(columns=['sub_grade','emp_title', 'issue_d','title','address', 'earliest_cr_line'], axis=1, inplace=True)


# ### Feature Encoding

# In[70]:


term_values = {' 36 months': 0, ' 60 months': 1}
data['term'] = data.term.map(term_values)
data['loan_status'] = data.loan_status.map({'Fully Paid':0, 'Charged Off':1})
data['pub_rec'] = data.pub_rec.map({0.0 : 0 , 1.0 : 1})
data['initial_list_status'] = data.initial_list_status.map({"f" : 0 , "w" : 1})
# data['mort_acc'] = data.mort_acc.map({0.0 : 0 , 1.0 : 1})
data['pub_rec_bankruptcies'] = data.pub_rec_bankruptcies.map({0.0 : 0 , 1.0 : 1})


# In[71]:


data.describe(include='all')


# In[72]:


dummies = ['grade','emp_length','home_ownership','verification_status','purpose','application_type','zip_code','mort_acc']
data = pd.get_dummies(data, columns=dummies, drop_first=True)


# In[73]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data.head()


# In[74]:


data.shape


# In[75]:


X = data.drop('loan_status', axis=1)
y = data['loan_status']


# In[76]:


X.columns


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)


# In[78]:


print(X_train.shape)
print(X_test.shape)


# In[79]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[80]:


logisticRegr = LogisticRegression(max_iter= 500)
logisticRegr.fit(X_train, y_train)


# In[81]:


y_pred = logisticRegr.predict(X_test)
accuracy = (logisticRegr.score(X_test, y_test))
print('Accuracy of Logistic Regression model on test data with Imbalanced Data: {:.4f}'.format(accuracy))


# In[82]:


print("Train Accurancy of Model ", (logisticRegr.score(X_train, y_train)))


# In[83]:


importance = logisticRegr.coef_[0]
# summarize feature importance
importanceDf = pd.DataFrame(list(zip(list(X.columns), importance)),columns =['Feature', 'Imporatnce'])
importanceDf


# In[84]:


importanceDf.sort_values(by=['Imporatnce'], ascending=False)


# In[85]:


conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix).plot()

Positive class = Charged Off
Negative Class = Fully Paid
# In[86]:


tn, fp, fn, tp = conf_matrix.ravel()
print("True Negative: ", tn)
print("False Positive: ", fp)
print("False Negative: ", fn)
print("True Positive: ", tp)

Here False negative is higher than number of false positive
Model to be more accurate false positive and false negative values must be as minimum as posiible.
# In[87]:


accuracy = np.round(np.diag(conf_matrix).sum() / conf_matrix.sum(),3)
accuracy

As data is imbalanced here so accurancy is not a good parameter to check the performace of our logistic regression model.


Case 1 : Classify a charged off loan as fully paid loan (False Negative) --> Recall
Case 2 : Misclassifying a customer who fully paid loan as charged off (False Positive) --> Precision

Both the cases will leads in following consequences:
case 1 : Considering charged off loan as fully paid which will result in financial losses for firm.
case 2 : Considering customer with fully paid loan as charged off loan which will eventually leads in loosing customers.

Here both the cases are important so reducing FN and FP both are critical here

# In[88]:


precision = precision_score(y_test, y_pred)
np.round(precision, 3)


# In[89]:


recall = recall_score(y_test, y_pred)
np.round(recall, 3)


# In[90]:


print(classification_report(y_test, y_pred))

For our model precision is 0.89 and recall is 0.467 which means A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels. 
# In[91]:


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Precision:', precision)
print('Recall:', recall)

f1_score = 2*precision*recall/(precision + recall)
print('F1 score:',f1_score)


# In[92]:


roc_auc = roc_auc_score(y_test, y_pred)
roc_auc


# In[93]:


dumbModelProb = [0 for _ in range(len(y_test))]
roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, logisticRegr.predict_proba(X_test)[:,1])
ns_fpr, ns_tpr,thresholds = roc_curve(y_test, dumbModelProb)
plt.plot(ns_fpr, ns_tpr, linestyle='--')
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

Here we got AUC score as 0.73 which poor in our case
# In[94]:


precisions, recalls, thresholds = precision_recall_curve(y_test, logisticRegr.predict_proba(X_test)[:,1]) 
from sklearn.metrics import auc
auc = auc(recalls, precisions)
auc


# In[95]:


threshold_boundary = thresholds.shape[0]
plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
plt.plot(thresholds, recalls[0:threshold_boundary], linestyle = '--', label='recalls')
start, end = plt.xlim()
plt.xticks(np.round(np.arange(start, end, 0.1), 2))  
plt.xlabel('Threshold Value'); plt.ylabel('Precision and Recall Value')
plt.legend()
plt.show()

The area under curve for Precision recall curve is 0.77 which is poor for threshold 0.5
# #### Multicollinearity check using Variance Inflation Factor

# In[96]:


def vif(X):    
    vif = pd.DataFrame()
    vif['Feature'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by='VIF', ascending = False)
    return vif


# In[97]:


vif(X)[:5]


# In[98]:


X.drop('application_type_INDIVIDUAL', axis=1, inplace=True)
vif(X)[:5]


# In[99]:


X.drop('int_rate', axis=1, inplace=True)
vif(X)[:5]


# In[100]:


X.drop('purpose_debt_consolidation', axis=1, inplace=True)
vif(X)[:5]


# In[101]:


X.drop('open_acc', axis=1, inplace=True)
vif(X)[:5]


# In[102]:


X.drop('revol_util', axis=1, inplace=True)
vif(X)[:5]


# In[103]:


X.drop('annual_inc', axis=1, inplace=True)
vif(X)[:5]


# In[104]:


X.drop('total_acc', axis=1, inplace=True)
vif(X)[:5]


# In[105]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logisticRegr = LogisticRegression(max_iter= 500)
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)
accuracy = (logisticRegr.score(X_test, y_test))


# In[106]:


print('Accuracy of Logistic Regression model on test data Afer Removing features with high multicollinearity: {:.4f}'.format(accuracy))


# In[107]:


print(classification_report(y_test, y_pred))

Even after Removing features which were responsible for multicollinearity we are getting accuracy 0.8911
# ### Oversampling using SMOTE

# In[108]:


sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train, y_train)


# In[109]:


X_sm.shape


# In[110]:


y_sm.shape


# In[111]:


y_sm.value_counts()


# In[112]:


logisticRegr = LogisticRegression(max_iter= 500)
logisticRegr.fit(X_sm, y_sm)


# In[113]:


y_pred = logisticRegr.predict(X_test)
accuracy = (logisticRegr.score(X_test, y_test))
print('Accuracy of Logistic Regression model on test data after resampling using SMOTE is: {:.4f}'.format(accuracy))

After applying SMOTE accuracy of model reduced to 0.798.
# In[114]:


print(classification_report(y_test, y_pred))

But after applying smote our precesion and recall scores significantly increased.
# In[115]:


conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix).plot()


# In[116]:


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Precision:', precision)
print('Recall:', recall)

f1_score = 2*precision*recall/(precision + recall)
print('F1 score:',f1_score)

After application of smote Precision has been significantly decreased and Recall value has been increased
# In[117]:


dumbModelProb = [0 for _ in range(len(y_test))]
roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, logisticRegr.predict_proba(X_test)[:,1])
ns_fpr, ns_tpr,thresholds = roc_curve(y_test, dumbModelProb)
plt.plot(ns_fpr, ns_tpr, linestyle='--')
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# The higher the area under the ROC curve better the classifier AUC score increased from 0.72 to 0.80 which can be we now consider as fair.

# In[118]:


precisions, recalls, thresholds = precision_recall_curve(y_test, logisticRegr.predict_proba(X_test)[:,1])
threshold_boundary = thresholds.shape[0]
plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
plt.plot(thresholds, recalls[0:threshold_boundary], linestyle = '--', label='recalls')
start, end = plt.xlim()
plt.xticks(np.round(np.arange(start, end, 0.1), 2))
plt.xlabel('Threshold Value'); plt.ylabel('Precision and Recall Value')
plt.legend()
plt.show()

threshold increased to 0.63 which is responsible for increasing model's f1 score.

After oversampling our recall value increased but precision value significantly decreased, So we net more false positive and less false negatives.
Observations

• F1-score: Because of the high class imbalance and it might make more sense to look at the weighted (by support) F1 score. F1 score does not take into account the relative ratio of true negatives and false negatives and might be misleading to some extent, since in our highly imbalanced data, the number of false negatives is becoming comparable to the number of true positives, thereby reducing precision greatly (for the 1 class or the minority class). In fact any single metric is not sufficient to analyse on the basis of our needs.
• ROC-AUC: In simple terms. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes. But since we have a data imbalance between positive and negative samples, we should use F1-score because ROC averages over all possible thresholds.Even though AUC progressively increases after SMOTE oversampling, it is not best indicator to pick the model, as it affected the precision wrsely
# # Tradeoff Questions
# 
# ### 1. How can we make sure that our model can detect real defaulters and there are less false positives? This is important as we can lose out on an opportunity to finance more individuals and earn interest on it.
# 
# In order to get less False postives and detect real deafaulters, In case of low recall and high precision, consumer finance company might lose money as well as the customers due to strict rule and conservative approach. It would be better to have a low precision and high recall in our case with little risk as we don't want to lose customers and would be a good idea to alarm the company and provide loan at high interest rates even if there is a slight doubt about defaulter.It is important to have a balance between recall and precision, so a good F1-score will make sure that
# balance is maintained.
# As we have seen in our first model, by keeping precion score higher and then assigning priority to f1 score and then recall, we can make sure that real dafualters (TP) and low False positives (FP) in demominator of Precision. This is important as we can lose out on an opportunity to finance more
# supply chains and earn interest in it.
# 

# ### 2. Since NPA (non-performing asset) is a real problem in this industry, it’s important we play safe and shouldn’t disburse loans to anyone.
# Yes. LoanTap should not disburse loans to everyone. Company’s internal policy and analysis should be in place to identify the correct persons. Any asset or Customer who took loan which stops giving returns to its organization for a specified period of time is known as Non-Performing Asset (NPA).
# In this case, it can be okay to have slightly low precision i.e., Misclassifying a customer who fully paid loan as charged off, but high recall will play key role here, We should decrease False Negative values in order to get high recall. Disbursing loans to anyone can result into fiannacial losses to the organization and which will eventually worsen the market situation for the organization.
# In this case priority would be minimizing gap between precision and recall with maximizinng AUC. Because having less precision would also result into Customer attrition who can fully pay loans if we consider them as NPA, which will eventually lead into Financial losses.
# Company should improvise Technology and data analytics to identify the early warning signals that person would be deafualter.
# 
# The higher the precision the less likely it is to recruit defaulters, but the potential client pool becomes smaller. The higher the recall, the larger the potential pool of clients but the higher the risk recruiting defaulters. The balance of our recall and precision levels is a matter of risk appetite

# # Questionnaire 

# ### 1.	What percentage of customers have fully paid their Loan Amount?
# Around 80.3871% Customers fully paid their loan amount and remaining are charged off.
# 

# ### 2.	Comment about the correlation between Loan Amount and Instalment features.
# Correlation coefficient between two features Loan Amount and Instalment is very high. Correlation coefficient between these two features is 0.97 which indicates these to features are dependent on each other which is indicative of multicollinearity. We can drop any one feature between these two. So dropped Instalments feature
# 

# ### 3.	The majority of people have home ownership as _______.
# The majority people have home ownership as MORTGAGE. Around 50 Percent of customers have MORTAGE type home ownership. And around 40.35% customers have home ownership as rent
# 

# ### 4.	People with grades ‘A’ are more likely to fully pay their loan. (T/F).
# True People with grades ‘A’ are more likely to fully pay their loan.
# Even though Only 16% Customers belongs to A grade took loan but 94% Customers among them fully paid their loan which is highest as compared to other grades. 29% and 27% Customers belongs to grade B and Grade C respectively But only 87% and 78% fully paid their loans for B and C grade respectively.
# 
# 

# ### 5.	Name the top 2 afforded job titles.
# 1. Manager - Around 1.5 % Customers are managers
# 2. Teacher - Around 1.4 % Customers are Teachers
# 

# ### 6.	Thinking from a bank's perspective, which metric should our primary focus be on. 1. ROC AUC 2. Precision 3. Recall 4. F1 Score
# 1. In our case as we bank would not like to lose money as well as     customers, Precision and Recall both plays an important role. 
# 2. Precision is more important Missing out to identify/classify a good customer eligible for the loan is okay (low recall), but approving a loan to a bad customer (false positive) who may never repay it is undesirable.
# 3. A low recall or precision (one or both inputs) makes the F1-score more sensitive, which is great if you want to balance the two. The higher the F1- score the better the model
# 4. If we think in terms of not losing money in bad loans, Recall is the parameter which should be more focused. f we think in terms of not losing customers at the cost of losing some money in bad loans, Precision should be the main parameter that we should focus as in model 1.
# 5. In order to conclude, since NPA (non-performing asset) is a real problem in this industry, it's important we play safe and shouldn't disburse loans to anyone. And hence model 2 is the best model which is having highest Recall.
# 6. But we have to provide a balanced stance, then F1 score is the metric as it considers both Precision and Recall (harmonically).
# 

# ### 7.	How does the gap in precision and recall affect the bank?
# 1. When predicting whether or not a loan will default - it would be better to have a high recall because the banks don't want to lose money, so it would be a good idea to alert the bank even if there is a slight doubt about the borrower. Low precision in this case,] might be okay.
# 2. When a bank wants to grow faster and get more customers at the expense of losing some money in some cases. In this case, it would be ok to have a slightly higher precision compare the recall.
# 3. If it is important that the defaulters be predicted with better accuracy we should concentrate on recall. If customer base is important, we will disburse loan more liberally increase precision.
# 
# The gap in precision and recall affects the banks when if more difference in precision and recall, more are the chances of banks losing money in form of bad loans. In Model 2 we are getting less precision score than the recall score whereas in other model the difference between Precision and Recall is around 50 % (Precision being the highest and recall being in range of 40-50 %. This clearly shows that there's a risk of banks giving bad loans to defaulters and thus the NPA of the banks are currently rising.
# 

# ### 8.	Which were the features that heavily affected the outcome?
# Zip Code, Grade, Debt to Income ratio and Term are heavily affected the outcome
# 

# ### 9.	Will the results be affected by geographical location? (Yes/No)
# Yes, results affected by geographical location. We Could see location with zip code 11650, 86630, 93700 are charged off. No people from these areas fully paid their loans. while people wit 05113, 00813 and 29597 fully paid their loans. Without considering geographical locations we were getting around 80% accuracy and after considering zip code accuracy increased to 90%
# 

# # Actionable Insights and Recommendations

# 1. Non paying asset is the rising and main concern for the loantap company, so it is important to think in perspective of not lossing money in loans, False Negative value plays an key role here we should not misclassiffy charged off customers as fully paid, which eventually result in loss of money so We shoul focuas on Recall Parameter. If there are derogatory public records in form of earlier banksruptcies, then also, the company should focus on not giving any loan to such customers/businesses.
# 2. Moreover, the banks assigning grades and subgrades are also equally important as if the grades are of low gradings such as E,F and G which have around 40-50 % probability of the customers defaulting on the loan taken. The banks should be careful in this perspective as well. LoanTap can increase their market presence in Zipcodes with high full paying ratio and minimize their marketing expenditure in Zipcodes where charging off ratio is high.
# 3. As we could see location with zip code 11650, 86630, 93700 are charged off, No people from these areas fully paid their loans so organization should reverify customers from these areas before considering them for loan approval. Loantap should reverify thier assests collaterals before giving them loans. 
# 4. Also we could see that loans took for deb't consolidation have large cahnces of charged off, so loantap should reverify the previous loans history of customer before considering them for loans.
# 5. Customers with Home ownership as Rent have high posibility of charging off, so banks should verify thier income sources, other assests, collaterals.
# 6. Individuals with higher income are more likely to pay off their loans. Income range more than 80000 has less chances of charging off.With increase in annual income charged off proportion got decreased. So Loantap should consider Work experience, experience in your industry and personal credit history for customers with income range less than 80000 before giving them loans and verify thier income sources.
# 7. Loantap can implement Technology and data analytics to identify the early warning signals regarding NPA's. They can develop thier internal skills for credit assessment. and can perform Forensic audits to understand the intent of the borrower.
