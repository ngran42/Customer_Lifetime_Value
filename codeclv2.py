# Databricks notebook source
from  pyspark.sql.types import IntegerType
from  pyspark.sql.types import FloatType
from  pyspark.sql.types import StringType
from  pyspark.sql.types import TimestampType

orders = sqlContext.table("customer_analytics_ng.orders").cache()
orders = orders.withColumn("InvoiceDate", orders["InvoiceDate"].cast(StringType()))
orders = orders.withColumn("InvoiceDate2", orders["InvoiceDate2"].cast(TimestampType()))
orders = orders.withColumn("UnitPrice", orders["UnitPrice"].cast(FloatType()))
orders = orders.withColumn("CustomerID", orders["CustomerID"].cast(IntegerType()))
orders = orders.withColumn("InvoiceNo", orders["InvoiceNo"].cast(StringType()))
orders = orders.withColumn("Quantity", orders["Quantity"].cast(IntegerType()))

# COMMAND ----------

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime as dt

# COMMAND ----------

data = orders.toPandas()

# COMMAND ----------

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# COMMAND ----------

data['InvoiceDate2'] = pd.to_datetime(data['InvoiceDate2'])

# COMMAND ----------

data.head()

# COMMAND ----------

data.tail()

# COMMAND ----------

data.info()

# COMMAND ----------

data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# COMMAND ----------

data['InvoiceDate2'].min(),data['InvoiceDate2'].max()

# COMMAND ----------

PRESENT = dt.datetime(2022,10,2)
data['InvoiceDate2'] = pd.to_datetime(data['InvoiceDate2'])

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md For Recency, Calculate the number of days between present date and date of last purchase each customer.
# MAGIC 
# MAGIC For Frequency, Calculate the number of orders for each customer.
# MAGIC 
# MAGIC For Monetary, Calculate sum of purchase price for each customer.

# COMMAND ----------

rfm= data.groupby('CustomerID').agg({'InvoiceDate2': lambda date: (PRESENT - date.max()).days,
                                        'InvoiceNo': lambda num: len(num),
                                        'TotalPrice': lambda price: price.sum()})

# COMMAND ----------

rfm.columns

# COMMAND ----------

# Change the name of columns
rfm.columns=['monetary','frequency','recency']

# COMMAND ----------

rfm['recency'] = rfm['recency'].astype(int)

# COMMAND ----------

rfm.head()

# COMMAND ----------

# MAGIC %md ###Ranking Customer’s based upon their recency, frequency, and monetary score

# COMMAND ----------

rfm['R_rank'] = rfm['recency'].rank(ascending=False)
rfm['F_rank'] = rfm['frequency'].rank(ascending=True)
rfm['M_rank'] = rfm['monetary'].rank(ascending=True)

# normalizing the rank of the customers
rfm['R_rank_norm'] = (rfm['R_rank']/rfm['R_rank'].max())*100
rfm['F_rank_norm'] = (rfm['F_rank']/rfm['F_rank'].max())*100
rfm['M_rank_norm'] = (rfm['F_rank']/rfm['M_rank'].max())*100

rfm.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

rfm.head()

# COMMAND ----------

'''
rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4'])
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1'])
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])
'''

# COMMAND ----------

rfm.head()

# COMMAND ----------

# MAGIC %md ###Calculating RFM score
# MAGIC RFM score is calculated based upon recency, frequency, monetary value normalize ranks. Based upon this score we divide our customers. Here we rate them on a scale of 5. Formula used for calculating rfm score is : 0.15*Recency score + 0.28*Frequency score + 0.57 *Monetary score

# COMMAND ----------

rfm['RFM_Score'] = 0.15*rfm['R_rank_norm']+0.28 * \
	rfm['F_rank_norm']+0.57*rfm['M_rank_norm']
rfm['RFM_Score'] *= 0.05
rfm = rfm.round(2)
rfm[['RFM_Score']].head(7)

# COMMAND ----------

import numpy as np

rfm["Customer_segment"] = np.where(rfm['RFM_Score'] >
									4.5, "Top Customers",
									(np.where(
										rfm['RFM_Score'] > 4,
										"High value Customer",
										(np.where(
	rfm['RFM_Score'] > 3,
							"Medium Value Customer",
							np.where(rfm['RFM_Score'] > 1.6,
							'Low Value Customers', 'Lost Customers'))))))
rfm[['RFM_Score', 'Customer_segment']].head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Visualizing the customer segments
# MAGIC Here we will use a pie plot to display all segments of customers.

# COMMAND ----------

plt.pie(rfm.Customer_segment.value_counts(),
		labels=rfm.Customer_segment.value_counts().index,
		autopct='%.0f%%')
plt.show()

# COMMAND ----------

# Filter out Top/Best cusotmers
rfm[rfm['RFM_Score']>4].sort_values('monetary', ascending=False).head()

# COMMAND ----------

rfm.head()

# COMMAND ----------

