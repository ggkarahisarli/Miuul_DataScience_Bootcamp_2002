import streamlit as st
import pandas as pd
import numpy as np





Income  = st.slider("Your Income",0,1000000000000000)
Gender = st.selectbox('Your Gender', ['M' 'F'])
Own_Car  = st.selectbox('Are you own a car?', ['Y', 'N'])
Own_Realty  = st.selectbox('Are you own a reality?', ['Y' ,'N'])
Children_Count = st.slider("Number of your children",0,25)
Education  = st.selectbox('Education type', ['Higher education' ,'Secondary / secondary special' ,
                                     'Incomplete higher' ,'Lower secondary',
                                     'Academic degree'])
Family_Status = st.selectbox('Family status',  ['Civil marriage', 'Married' 'Single / not married' ,'Separated', 'Widow'])
Housing_Type = st.selectbox('Housing type', ['Rented apartment' ,'House / apartment',  'Municipal apartment', 'With parents', 'Co-op apartment' ,'Office apartment'])

Age = st.slider("Age",18,90)
Employment_Date = st.slider("Years of working",0,80)
Own_Work_Phone  = st.selectbox('Are you own a work phone?', ['Y', 'N'])
Own_Phone  = st.selectbox('Are you own a phone?', ['Y', 'N'])
Own_Email = st.selectbox('Are you own an email?', ['Y', 'N'])
Family_Member_Count = st.slider("Number of your children",0,35)
Income_Type = st.selectbox('Income Type', ['Working', 'Commercial associate', 'Pensioner' ,
                                      'State servant', 'Student'])

res = pd.DataFrame(data =
        {"Income":[Income],"Gender":[Gender], "Own_Car":[Own_Car],"Own_Realty"  :[Own_Realty],
"Children_Count" :[Children_Count],"Education"  :[Education],"Family_Status" :[Family_Status],
"Housing_Type":[Housing_Type],"Age" :[Age],"Employment_Date" :[Employment_Date],"Own_Work_Phone"  :[Own_Work_Phone],
"Own_Phone"  :[Own_Phone],"Own_Email" :[Own_Email],"Family_Member_Count" :[Family_Member_Count],"Income_Type":[Income_Type]})



res["Level_of_Experience"] = pd.cut(df_data.Employment_Date, bins=(-1,5,20,30,80), labels=("junior","experienced",
                                                                                      "senior", "expert"))

res["Is_Working"] = res["Income_Type"].replace(["Working","Commercial associate",
                                                        "State servant","Pensioner","Student"],
                                                       [1,1,1,0,0])

res["In_Relationship"] = res["Family_Status"].replace(["Married","Civil marriage",
                                                               "Single / not married",
                                                               "Separated","Widow"],
                                                              [1,1,0,0,0])

#Seperate Education based on values in it

education_type = {'Secondary / secondary special':'secondary',
                  'Lower secondary':'secondary',
                  'Higher education':'higher education',
                  'Incomplete higher':'higher education',
                  'Academic degree':'higher education'}
res["Education"] =res["Education"].map(education_type)


#Calculation of ages via Birthday feature
res.loc[(res["Age"]>= 20) & (res["Age"]<=30), 'Age_Cat'] = 'very young'
res.loc[(res["Age"]> 30) & (res["Age"]<= 45), 'Age_Cat'] = 'young'
res.loc[(res["Age"]> 45) & (res["Age"] <= 60), 'Age_Cat'] = 'middle aged'
res.loc[(res["Age"] > 60), 'Age_Cat'] = 'old'




#Let's check Housing_Type in the same way
res["Own_House"] = res["Housing_Type"].replace(['Rented apartment', 'House / apartment', 'Municipal apartment',
                                                        'With parents', 'Co-op apartment', 'Office apartment'],
                                                       [0,1,0,0,0,0]) #Yeniden gruplandırılacak

#Let's check now "Family_Member_Count and Children_Count
res["Household_Size"] = res["Children_Count"]\
                            + res["In_Relationship"].apply(lambda x: 2 if x==1 else 1)

#Time to drop feature which are useless
res.drop("Income_Type", axis=1, inplace=True)
res.drop("Experience", axis=1, inplace=True)
res.drop("Employment_Date", axis=1, inplace=True)
res.drop("Family_Status", axis=1, inplace=True)
res.drop("Age", axis=1, inplace=True)
res.drop("Birthday", axis=1, inplace=True)
res.drop("Housing_Type", axis=1, inplace=True)
res.drop("Family_Member_Count", axis=1, inplace=True)
res.drop("Children_Count", axis=1, inplace=True)
res.drop("Own_Email", axis=1, inplace=True)



#Label encoder for Education
le = preprocessing.LabelEncoder()
columns = ["Education"]

for col in columns:
    res[col] = le.fit_transform(res[col])
    print(le.classes_)

# One-Hot Encoding for Level_of_Experince, Age_Cat, Household_Size
ohe_cols = [col for col in res.columns if 20 >= res[col].nunique() >= 2]
df_ohe = one_hot_encoder(res, ohe_cols)
res = df_ohe


#Scaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
scaler = StandardScaler().fit(res[['Income']])
res[['Income']] = scaler.transform(res[['Income']])



dt = joblib.load('DecisionTreeModel.pkl')
prediction = dt.predict(res)
prediction = str(prediction).strip('[]')

if prediction == '0':
   prediction = "Bad"
else:
    prediction="Good"
st.write("Decision Model Prediction: ",prediction)

if __name__ == '__main__':
    main()