
# Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import pandas as pd
import seaborn as sns
import warnings
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve, \
    RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder,  RobustScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from datetime import date,timedelta
import datetime
import missingno as msno
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, \
    recall_score,precision_score,f1_score, roc_curve

#Adjustments
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

# Helper functions
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
#Unique counts for application dataset
def unique_counts(df):
    unique_counts = pd.DataFrame.from_records([(col, df[col].nunique()) for col in df.columns],
                                              columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
    return  unique_counts
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

#Outlier functions
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def outlier_graph(df, num_cols):
    for column in df[num_cols]:
        sns.boxplot(x=df[column])
        plt.show(block=True)
    return
#missing values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

#Color settings for the notebook
sns.set_context("notebook", font_scale=.7, rc={"grid.linewidth": 0.1, 'patch.linewidth': 0.0,
                                               "axes.grid": True,
                                               "grid.linestyle": "-",
                                               "axes.titlesize": 10,
                                               "figure.autolayout": True})
palette_1 = ['#FF5E5B', '#EC9B9A', '#00CECB', '#80DE99', '#C0E680', '#FFED66']
sns.set_palette(sns.color_palette(sns.color_palette(palette_1)))
plt.figure(figsize=(10,10))

# plot missing data
def plot_missing_data(df):
    msno.bar(df)
    plt.show(block=True)

    msno.matrix(df)
    plt.show(block=True)

# Encoding
# Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#Checking unique IDs and length of datasets
def lenofnunique(df, col_name):
    print(f'Length of dataframe is: {len(df)}')
    print(f'Length of ID column is: {df[col_name].nunique()}')
    return
#####################################################
# Data
#####################################################

#load data sets

df_data = pd.read_csv("Proje/application_record.csv")
df_record = pd.read_csv("Proje/credit_record.csv")

#First let's drop "FLAG_MOBIL" whic has 1 value only
df_data.drop("FLAG_MOBIL", axis=1, inplace=True)

#Now, drop "OCCUPATION_TYPE which has no correlation btw other and not affected target
df_data.drop("OCCUPATION_TYPE", axis=1, inplace=True)


#Now remove duplicate values and keep the last entry of the ID if its repeated
df_data = df_data.drop_duplicates("ID", keep="last")


#renaming columns
df_data.rename(columns={"CODE_GENDER":"Gender","FLAG_OWN_CAR":"Own_Car","FLAG_OWN_REALTY":"Own_Realty",
                        "CNT_CHILDREN":"Children_Count","AMT_INCOME_TOTAL":"Income","NAME_EDUCATION_TYPE":"Education",
                        "NAME_FAMILY_STATUS":"Family_Status","NAME_HOUSING_TYPE":"Housing_Type","DAYS_BIRTH":"Birthday",
                        "DAYS_EMPLOYED":"Employment_Date","FLAG_WORK_PHONE":"Own_Work_Phone",
                        "FLAG_PHONE":"Own_Phone","FLAG_EMAIL":"Own_Email","CNT_FAM_MEMBERS":"Family_Member_Count",
                        "NAME_INCOME_TYPE":"Income_Type"},inplace=True)

#Applying encodding process

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
columns = ["Gender",  "Own_Car", "Own_Realty", "Own_Work_Phone", "Own_Phone", "Own_Email"]

for col in columns:
    df_data[col] = le.fit_transform(df_data[col])
    print(le.classes_)


#Adding new features

df_data['Experience'] = df_data['Employment_Date'].apply(lambda x: (abs(x / 30)/12) if x <0 else 0)
df_data["Level_of_Experience"] = pd.cut(df_data.Experience, bins=(-1,5,20,30,80), labels=("junior","experienced",
                                                                                      "senior", "expert"))

#Seperate Income type according to is working or not
df_data["Is_Working"] = df_data["Income_Type"].replace(["Working","Commercial associate",
                                                        "State servant","Pensioner","Student"],
                                                       [1,1,1,0,0])

#Seperate Family_Status based on values in it
df_data.Family_Status.unique()
df_data["In_Relationship"] = df_data["Family_Status"].replace(["Married","Civil marriage",
                                                               "Single / not married",
                                                               "Separated","Widow"],
                                                              [1,1,0,0,0])

#Seperate Education based on values in it

education_type = {'Secondary / secondary special':'secondary',
                  'Lower secondary':'secondary',
                  'Higher education':'higher education',
                  'Incomplete higher':'higher education',
                  'Academic degree':'higher education'}
df_data["Education"] = df_data["Education"].map(education_type)


#Calculation of ages via Birthday feature
df_data['Age'] = df_data['Birthday'].apply(lambda x: (abs(x / 30)/12))
df_data['Age'].max() #70.00277777777778
df_data['Age'].min() #20.802777777777777
df_data.loc[(df_data["Age"]>= 20) & (df_data["Age"]<=30), 'Age_Cat'] = 'very young'
df_data.loc[(df_data["Age"]> 30) & (df_data["Age"]<= 45), 'Age_Cat'] = 'young'
df_data.loc[(df_data["Age"]> 45) & (df_data["Age"] <= 60), 'Age_Cat'] = 'middle aged'
df_data.loc[(df_data["Age"] > 60), 'Age_Cat'] = 'old'
df_data.head()

#Let's check Housing_Type in the same way
df_data["Own_House"] = df_data["Housing_Type"].replace(['Rented apartment', 'House / apartment', 'Municipal apartment',
                                                        'With parents', 'Co-op apartment', 'Office apartment'],
                                                       [0,1,0,0,0,0]) #Yeniden gruplandırılacak

#Let's check now "Family_Member_Count and Children_Count
df_data["Household_Size"] = df_data["Children_Count"]\
                            + df_data["In_Relationship"].apply(lambda x: 2 if x==1 else 1)

#Time to drop feature which are useless
df_data.drop("Income_Type", axis=1, inplace=True)
df_data.drop("Experience", axis=1, inplace=True)
df_data.drop("Employment_Date", axis=1, inplace=True)
df_data.drop("Family_Status", axis=1, inplace=True)
df_data.drop("Age", axis=1, inplace=True)
df_data.drop("Birthday", axis=1, inplace=True)
df_data.drop("Housing_Type", axis=1, inplace=True)
df_data.drop("Family_Member_Count", axis=1, inplace=True)
df_data.drop("Children_Count", axis=1, inplace=True)
df_data.drop("Own_Email", axis=1, inplace=True)

#Let's replace with thresholds for outliers
check_outlier(df_data,"Household_Size")
check_outlier(df_data,"Income")

outlier_thresholds(df_data,"Household_Size")
outlier_thresholds(df_data,"Income")

replace_with_thresholds(df_data,"Household_Size")
replace_with_thresholds(df_data,"Income")

#Label encoder for Education
le = preprocessing.LabelEncoder()
columns = ["Education"]

for col in columns:
    df_data[col] = le.fit_transform(df_data[col])
    print(le.classes_)

# One-Hot Encoding for Level_of_Experince, Age_Cat, Household_Size
ohe_cols = [col for col in df_data.columns if 20 >= df_data[col].nunique() >= 2]
df_ohe = one_hot_encoder(df_data, ohe_cols)
df_data = df_ohe

#Target value analysis
convert_to = {'C' : 'Good_Debt', 'X' : 'Good_Debt', '0' : 'Good_Debt',
              '1' : 'Neutral_Debt', '2' : 'Neutral_Debt',
              '3' : 'Bad_Debt', '4' : 'Bad_Debt', '5' : 'Bad_Debt'}
df_record.replace({'STATUS' : convert_to}, inplace=True)

credit = df_record.value_counts(subset=['ID', 'STATUS']).unstack(fill_value=0)
credit["Total"]= credit["Bad_Debt"]+credit["Good_Debt"]+credit["Neutral_Debt"]
credit["Bad_Debt_Perc"]= credit["Bad_Debt"]/credit["Total"]
credit["Good_Debt_Perc"]= credit["Good_Debt"]/credit["Total"]
credit["Neutral_Debt_Perc"]= credit["Neutral_Debt"]/credit["Total"]
credit["Result"]= credit[["Bad_Debt","Good_Debt","Neutral_Debt"]].idxmax(axis=1)
credit[credit["Result"]=="Bad_Debt"]

credit[credit['Result']=="Good_Debt"].head(25)
credit.loc[(credit["Result"] !="Bad_Debt"), 'CREDIT_APPROVAL_STATUS'] = 1
credit.loc[(credit["Result"] =="Bad_Debt"), 'CREDIT_APPROVAL_STATUS'] = 0
credit['CREDIT_APPROVAL_STATUS'] = credit['CREDIT_APPROVAL_STATUS'].astype('int')

credit_final = pd.DataFrame(credit["CREDIT_APPROVAL_STATUS"])
credit_final.reset_index()

#Now, let's create our train dataset
df_train = df_data.merge(credit_final, on="ID", how="inner")


#Now, let's create our test data
df_test = df_data[~df_data["ID"].isin(df_record["ID"])]
df_test.reset_index()

#Scaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
scaler = StandardScaler().fit(df_test[['Income']])
df_test[['Income']] = scaler.transform(df_test[['Income']])

scaler = StandardScaler().fit(df_train[['Income']])
df_train[['Income']] = scaler.transform(df_train[['Income']])

df_test.set_index("ID", inplace=True)
########################################################################################################################

##################################
# Modeling
##################################
#Model
y = df_train["CREDIT_APPROVAL_STATUS"]
X = df_train.drop(["ID", "CREDIT_APPROVAL_STATUS"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE (Synthetic Minority Oversampling Technique) to Balance Dataset
X_balance, y_balance = SMOTE().fit_resample(X_train, y_train)
X_balance = pd.DataFrame(X_balance, columns = X_train.columns)
y_balance = pd.DataFrame(y_balance, columns=["CREDIT_APPROVAL_STATUS"])

### BASE MODELS ###
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,roc_auc_score,f1_score,plot_confusion_matrix,plot_roc_curve,roc_curve
from sklearn.neighbors import KNeighborsClassifier

################################################
# Random Forest
################################################

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results = cross_validate(rf_model, X_balance, y_balance, cv=10, scoring=["accuracy", "f1", "roc_auc"])
{'test_accuracy' : cv_results['test_accuracy'].mean(),
'test_f1': cv_results['test_f1'].mean(),
'test_roc_auc': cv_results['test_roc_auc'].mean()}


rf_model.fit(X_balance, y_balance)
y_predict = rf_model.predict(X_test)
#################################################################################################

# hiperparametre optimizasyonu için rf_model.get_params() ile gördüğümüz hp lerin base değerlerinin etrafında bazı değer
# aralıkları belirleyelim sonrasında bunları gridsearch ile denettiricez ve en iyi değerlerin neler olabileceğini göreceğiz

rf_params = {"max_depth": [2, 5, None],
             "max_features": [3, 5],
             "max_leaf_nodes": [2,5,10, None],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100]}


rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_balance, y_balance.values.ravel())

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_balance, y_balance.values.ravel())

cv_results = cross_validate(rf_final, X_balance, y_balance.values.ravel(), cv=10, scoring=["accuracy", "f1", "roc_auc"])
{'test_accuracy' : cv_results['test_accuracy'].mean(),
'test_f1': cv_results['test_f1'].mean(),
'test_roc_auc': cv_results['test_roc_auc'].mean()}


import joblib
joblib.dump(rf_final, 'DecisionTreeModel.pkl')