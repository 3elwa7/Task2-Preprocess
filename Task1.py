import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
from matplotlib import pyplot as plt

# read in all our data
data = pd.read_csv("companies.csv")

# drop nan value if nan value > 60%
data.dropna(thresh=data.shape[0]*0.6,how='any',axis=1,inplace = True )
nulls = data.isnull().sum()
nulls_prec = nulls[0:44]/data.shape[0]*100
print(data.describe())

# show unique come up per feature
uniques = data.select_dtypes(exclude='number').nunique()
print(uniques)

#Drop categorical features with too many categories
data.drop(columns=["id","entity_type","name","normalized_name","permalink","domain","homepage_url","overview","created_at","updated_at"],inplace = True)

# fill the null value
data = data.fillna(data.mode().iloc[0])
print(data)

# apply ordinal encoding
s = (data.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)

n = (data.dtypes != 'object')
number_cols = list(n[n].index)
print(number_cols)

label_X_train = data.copy()
ordinal_encoder = OrdinalEncoder()
data[object_cols] = ordinal_encoder.fit_transform(data[object_cols])
print(data)
#scale the data between 0 and 1
scaled_data = minmax_scaling(data, columns=["category_code","status","region","created_by","Unnamed: 0.1","entity_id","relationships"])

#Data visualization
sns.heatmap(scaled_data.corr(), cmap='RdBu',center=0)

sns.clustermap(scaled_data.corr(), cmap='RdBu',center=0)
plt.show()
