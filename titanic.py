
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
test_ids= test.PassengerId

def clean(data):
    data=data.drop(['Cabin','Ticket','Name','PassengerId'],axis=1)
    colmns=['Age','SibSp','Parch','Fare']
    for i in colmns:
        data[i].fillna(data[i].median(),inplace=True)
    data.Embarked.fillna('U',inplace=True)     
    return data

train=clean(train)
test=clean(test)

le=preprocessing.LabelEncoder()
colmns=['Sex','Embarked']
for i in colmns:
    train[i]=le.fit_transform(train[i])
    test[i]=le.transform(test[i])
    print(le.classes_)
print(train.head())


x=train.drop('Survived',axis=1)
y=train['Survived']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

model=LogisticRegression(max_iter=500).fit(x_train,y_train)
predictions=model.predict(x_test)
accuracy_score(y_test,predictions)

submission_predictions=model.predict(test)


df=pd.DataFrame({'PassengerId' :test_ids.values, 'Survived':submission_predictions  })



df.to_csv('submissions.csv',index=False)



