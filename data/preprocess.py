# https://competitions.codalab.org/competitions/11161#learn_the_details-data2
import numpy as np
import pandas as pd
# We are not using train-queries.csv and test-queries.csv. Instead we will use the other 4 datasets and merge it together.
# train-item-views.csv is the clicks. 
# train-purchases.csv is the buys and we merge both
# then we merge the result with product-categories.csv and products.csv to get category and price column.

dfp = pd.read_csv("data/products.csv", sep=';', usecols=[0,1])
dfpc = pd.read_csv("data/product-categories.csv", sep=';')
dftp = pd.read_csv("data/train-purchases.csv", sep=';')
dftv = pd.read_csv("data/train-item-views.csv", sep=';')

df = pd.concat([dftp.drop('ordernumber', axis=1), dftv], axis=0, sort=False)
df.sort_values(by = ['sessionId', 'timeframe'], inplace=True)
df = pd.merge(df, dfpc, how='left', on='itemId')
df = pd.merge(df, dfp, how='left', on='itemId')


# We are not using the items that have less than 5 views
df_filter = df['itemId'].value_counts()>5
df_filter = df_filter[df_filter].index
df = df[df['itemId'].isin(df_filter)]
grouped = df.groupby('sessionId')
df = grouped.filter(lambda x: x['itemId'].count() > 1)
df.drop('userId', axis=1, inplace=True)

df.columns = ['session_id', 'timestamp','date', 'item_id', 'category_id', 'price']
df = df[['session_id', 'item_id', 'category_id', 'timestamp', 'date' , 'price']]

df['date'] = pd.to_datetime(df['date'])
df.loc[:,'step'] = df.groupby('session_id').size().apply(range).explode().tolist()
df.loc[:,'duplicated'] = np.where(df[['session_id','item_id']].duplicated(), 1,0).tolist()

# derived features
df.loc[:,'day'] = df['date'].dt.day
df.loc[:,'month'] = df['date'].dt.month
df.loc[:,'hour'] = df['date'].dt.hour
df.loc[:,'day_of_week'] = df['date'].dt.dayofweek
df.loc[:,'week'] = df['date'].dt.week
df.loc[:,'dwell'] = df['timestamp'].diff()
df.loc[df['step']==0, 'dwell'] = 0
df.loc[:,'session_id'] = df['session_id'].astype('category').cat.codes
df.loc[:,'item_id'] = df['item_id'].astype('category').cat.codes
df.loc[:,'category_id'] = df['category_id'].astype('category').cat.codes
df.drop(['timestamp', 'date'], axis = 1, inplace=True)
df = df.drop_duplicates(subset=['session_id','item_id'])
df['item_id'] = df['item_id'].astype('category').cat.codes
df['category_id'] = df['category_id'].astype('category').cat.codes

p_max = df['price'].max()
p_min = df['price'].min()
def minmax_scale(x):
    return (x - p_min) / (p_max - p_min)
df['price'] = df['price'].apply(minmax_scale)

df.to_csv('data/diginetica_preprocessed.csv', index=False)