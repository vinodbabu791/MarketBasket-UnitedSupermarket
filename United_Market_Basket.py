# -*- coding: utf-8 -*-
"""
                Created on Sat Jul 14 23:59:12 2018

                @author: Vinod

                Title: Market Basket analysis of United Supermarket data
"""

'''
Step 1: import the necessary Libraries
'''

import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
from apyori import dump_as_json
import numpy as np


'''
Step 2:import item list file and transaction file;
    
item list consists of item description including its UPC, product name and department code;
transaction file consists of transactions; each transaction can involve multiple items;
item list and transaction file are linked using 'UPC' of product
'''
# import the item list 
itemList = pd.read_csv('C:/Users/Universe/Desktop/DataScience/Spring 2018/BI/Project/dataFiles/Item_List.txt',
            sep='|',
            usecols=[0,2,4,6,11],
            names=['UPC','ProductStatus','ProductName','ClassName','DepartmentCode'],
            encoding='latin1',
            dtype={'UPC':object,
                   'ProductStatus':'int32',
                   'ProductName':object,
                   'ClassName':object,
                   'DepartmentCode':object}
            )
# import the salesTransaction file
salesTrx = pd.read_csv('C:/Users/Universe/Desktop/DataScience/Spring 2018/BI/Project/dataFiles/SalesTrxCln.txt',
                       sep='|',
                       usecols=[2,6],
                       names=['Trx','UPC'],
                       dtype={'Trx':'int64','UPC':object})


'''
Step 3: Data Preprocessing
'''
# keep only active products: status == 1 implies active product
itemList.drop(index=itemList[itemList.ProductStatus!=1].index.values,inplace=True)
itemList.reset_index(inplace=True,drop=True)

itemList[['UPC','DepartmentCode']].groupby(['DepartmentCode']).count()

# drop the non-numeric department code
checkDigit = lambda x: x.isdigit()
itemList = itemList[[checkDigit(str(i)) for i in itemList.DepartmentCode]]

# change the department code type to int
itemList['DepartmentCode']=itemList['DepartmentCode'].astype('int64')

# summarise the number of products in each department
itemList[['DepartmentCode','UPC']].groupby(['DepartmentCode']).count()

# drop the product status as we have only one status i.e) Active
itemList.drop(axis=1,labels='ProductStatus',inplace=True)

# drop NA values in transaction file
salesTrx.isna().sum()
salesTrx.dropna(axis=0,how='any',inplace=True)

# merge sales transaction file with item list file based on 'UPC' of a product
salesTrx_items = salesTrx.merge(itemList,on='UPC',how='inner')

# consider a scenario where a person buys 2 cans of milk in same basket. we got to treat that as one product being bought 
# and not as 2 different product in same basket. So we consider one of the milk cans as duplicate entry and remove it. 
# We are interested in how many unique products being bought in the basket. so we drop duplicates
salesTrx_items.drop_duplicates(inplace=True)

# total number of transactions and total number of items
totTrx = len(salesTrx_items.Trx.unique())
totItems=len(salesTrx_items.UPC.unique())

# calculate frequency of each item
item_freq = salesTrx_items.groupby('ProductName')['Trx'].count().sort_values(ascending=False)


'''
Step 4: Let us visualize first 200 frequent items
'''
# set colormap
bar_cmap = plt.get_cmap('Blues')
# generate colors
colors = bar_cmap(np.linspace(0,1,200))
# flip the colors to have the darker shades of blue at first
colors = np.flip(colors,axis=0)
# plot subplots
for i in np.arange(1,5):
    plt.subplot(2,2,i)
    plt.bar(item_freq.index[(i*50-50):(i*50)],item_freq[(i*50-50):(i*50)],color=colors[(i*50-50):(i*50)])
    plt.xticks(rotation=90)
    plt.ylim(0,31000)
    plt.ylabel('Item Frequency')
plt.suptitle('Item Frequency Visualization')
plt.show()

'''
Step 5: calculate support for each product
'''
item_support = item_freq/totTrx
item_support.head(n=20)
# we can see the maximum support for a product is 0.05. 

'''
Step 6: create a basket with each row representing transaction and row elements representing items bought in that transaction
'''
basket=salesTrx_items.groupby('Trx')['ProductName'].apply(list).values.tolist()

'''
Step 7: implement apriori algorithm to find the association rules
'''
# we consider 1% of maximum support as min support and confidence of 60%. Also we will consider only
# those baskets with 2 or more items
market_basket = apriori(basket,min_support=0.0005,min_confidence=0.6,min_lift=1,min_length=2)
basket_rules = list(market_basket)

'''
Step 8: 
apriori algorithm results in RelationRecord format. 
we convert the RelationRecord format to json format using inbuilt function and then to dataframe to see the rules
'''

# writing the results to a file in json format
file = open('C:/Users/Universe/Desktop/DataScience/Project/MarketBasket/json_rules.txt','w')
for i in basket_rules:
    dump_as_json(i,file)
file.close()  

# reading the json file
records = pd.read_json('C:/Users/Universe/Desktop/DataScience/Project/MarketBasket/json_rules.txt',lines=True)

# processing the json output to proper dataframe structure
rules_df = pd.DataFrame()
for i in range(len(records.ordered_statistics)):
    for j in range(len(records.ordered_statistics[i])):
        rec = list(records.ordered_statistics[i][j].values())
        rec.append(records.support[i])
        rules_df=rules_df.append(pd.DataFrame([rec]))
rules_df.columns=['Antecedant','Consequent','Confidence','Lift','Support']
rules_df.sort_values(by='Lift',ascending=False,inplace=True)
rules_df.reset_index(drop=True,inplace=True)

