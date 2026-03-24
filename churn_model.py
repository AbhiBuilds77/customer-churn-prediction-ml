import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

#Load dataset
df=pd.read_csv("Telco-Customer-Churn.csv")

#Display first 10 rows
print("First 10 rows: ")
print(df.head(10))

#Dataset shape
print("\nShape: ",df.shape)

#Identify features and target variables
Features=df.drop("Churn",axis=1)
Target=df["Churn"]
print("\nFeatures: ")
print(list(Features.columns))
print("\nTarget variable:")
print(Target.name)

#Data types
print("\nData types: ")
print(df.dtypes)

#Missing values
print("\nMissing values before: ")
print(df.isnull().sum())
df.fillna(df.mean(numeric_only=True),inplace=True)
print("\nMissing values after: ")
print(df.isnull().sum())

#Encode Categorical variables
df.drop("customerID",axis=1,inplace=True)

#Encode target
df["Churn"]=df["Churn"].map({"Yes":1,"No":0})
#Encode categorical features
df=pd.get_dummies(df,drop_first=True)

#Feature Scaling
X = df.drop("Churn", axis=1)
y = df["Churn"]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#Split data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

#Train Models

#Logistic regression
lr_model=LogisticRegression(max_iter=1000)
lr_model.fit(X_train,y_train)
y_pred_lr=lr_model.predict(X_test)

#Decision tree
dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
y_pred_dt=dt_model.predict(X_test)

#Ensemble method-Random forest
rf_model=RandomForestClassifier()
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)

#Model evaluation
def evaluate(y_test,y_pred,model_name):
    print("\n" + model_name +" performance: ")
    print("Accuracy: ",accuracy_score(y_test,y_pred))
    print("Precision: ",precision_score(y_test,y_pred))
    print("Recall: ",recall_score(y_test,y_pred))
    print("F1 Score: ",f1_score(y_test,y_pred))

evaluate(y_test,y_pred_lr,"Logistic Regression")  
evaluate(y_test,y_pred_dt,"Decision Tree")   
evaluate(y_test,y_pred_rf,"Random Forest")    

#Confusion matrix heatmap
def plot_cm(y_test,y_pred,model_name):
    cm=confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True,cmap="coolwarm",fmt='d')
    plt.title(model_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_cm(y_test,y_pred_lr,"Logistic Regression")
plot_cm(y_test,y_pred_dt,"Decision Tree")
plot_cm(y_test,y_pred_rf,"Random Forest")

#Feature importance

#Decision tree
feature_names=X.columns
dt_importances=dt_model.feature_importances_
f_df1=pd.DataFrame({
    "Feature":feature_names,
    "Importance":dt_importances
})

f_df1 = f_df1.sort_values(by="Importance", ascending=False)
f_df1.head(10).plot(x="Feature",y="Importance",kind="bar")
plt.title("Top 10 Feature Importance of Decision Tree")
plt.show()

#Random forest
rf_importances=rf_model.feature_importances_
f_df2=pd.DataFrame({
    "Feature":feature_names,
    "Importance":rf_importances
})

f_df2 = f_df2.sort_values(by="Importance", ascending=False)
f_df2.head(10).plot(x="Feature",y="Importance",kind="bar")
plt.title("Top 10 Feature Importance of random forest")
plt.show()

#Model comparision
result_df=pd.DataFrame({
    "Model":["Logistic Regression","Decision Tree","Random Forest"],
    "Accuracy":[
        accuracy_score(y_test,y_pred_lr),
        accuracy_score(y_test,y_pred_dt),
        accuracy_score(y_test,y_pred_rf)
    ]
})
result_df.plot(x="Model",y="Accuracy",kind="bar")
plt.title("Model Comparision")
plt.show()
