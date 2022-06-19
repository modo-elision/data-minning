import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#reading input csv files stored locally
vTargetMailCustomer=pd.read_csv('archive/vTargetMailCustomer.csv')
print(vTargetMailCustomer.shape)
# set of attributes that will affect to predict future bike buyers
df=vTargetMailCustomer[['CustomerKey','GeographyKey','Gender','MaritalStatus',
                                         'EnglishEducation','EnglishOccupation','HouseOwnerFlag',
                                         'Age','BikeBuyer',
                                         'NumberCarsOwned','NumberChildrenAtHome','TotalChildren','YearlyIncome']]

#print(df.head())'DateFirstPurchase','CommuteDistance','Region',

print(df.dtypes)


df['Gender']=df['Gender'].map({'M':1,'F':0})
df['MaritalStatus']=df['MaritalStatus'].map({'M':1,'S':0})

df =df.dropna()
#print(df.shape)
#df=df.drop(['EnglishEducation','EnglishOccupation'])
EnglishEducation_df=pd.get_dummies(df['EnglishEducation'],drop_first=True)
df.drop(columns=['EnglishEducation'],axis=1,inplace=True)
df=pd.concat([EnglishEducation_df,df],axis=1)


EnglishOccupation_df=pd.get_dummies(df['EnglishOccupation'],drop_first=True)
df.drop(columns=['EnglishOccupation'],axis=1,inplace=True)
df=pd.concat([EnglishOccupation_df,df],axis=1)
#df= df.join(EnglishEducation_df)
#df= df.join(EnglishOccupation_df)


from sklearn.preprocessing import Normalizer
norm=Normalizer().fit(df)
normalized_df=norm.transform(df)

#print(EnglishEducation_df)

from scipy.spatial import distance

print("Cosine Similarity btw Management and YearlyIncome: ",distance.cosine(df['Management'].values,df['YearlyIncome'].values))

print("Cosine Similarity btw Graduate Degree and YearlyIncome: ",distance.cosine(df['Graduate Degree'].values,df['YearlyIncome'].values))


print ("jaccard Similarity btw Management and YearlyIncome: ",distance.jaccard(df['Management'].values,df['YearlyIncome'].values))
print ("jaccard Similarity btw Graduate Degree and YearlyIncome: ",distance.jaccard(df['Graduate Degree'].values,df['YearlyIncome'].values))

from scipy.stats import pearsonr
print ("pearsonr stats btw Management and YearlyIncome: ",pearsonr(df['Management'].values,df['YearlyIncome'].values)[0])
print ("pearsonr stats btw Graduate Degree and YearlyIncome: ",pearsonr(df['Graduate Degree'].values,df['YearlyIncome'].values)[0])




print("Cosine Similarity btw 11000 and 11001: ",distance.cosine(normalized_df[1],normalized_df[2]))

print("Cosine Similarity btw 11000 and 11002: ",distance.cosine(normalized_df[1],normalized_df[3]))

#print(df)

#print('Shape: ',np.shape(df))
'''
def scaleDown(df):
    scaler=MinMaxScaler()
    scaled=scaler.fit_transform(df[['YearlyIncome','Age']])
    df['YearlyIncome_scaled']=scaled[:,0]
    df['Age_scaled']=scaled[:,1]
    df.drop(['YearlyIncome','Age'],axis=1,inplace=True)
    return df

df=scaleDown(df)
'''
#df['EnglishEducation']=df['EnglishEducation'].map({'Partial High School':1,'High School':2,'Partial College':3,'Bachelors':4,'Graduate Degree':5})

#df['EnglishOccupation']=df['EnglishOccupation'].map({'Manual':1,'Skilled Manual':2,'Clerical':3,'Management':4,'Professional':5})

