# MAGIC %pip install lifetimes

# COMMAND ----------

import lifetimes

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# COMMAND ----------

from  pyspark.sql.types import IntegerType
from  pyspark.sql.types import FloatType
from  pyspark.sql.types import StringType
from  pyspark.sql.types import TimestampType

orders = sqlContext.table("customer_analytics_ng.orders").cache()
orders = orders.withColumn("InvoiceDate", orders["InvoiceDate"].cast(StringType()))
orders = orders.withColumn("InvoiceDate2", orders["InvoiceDate2"].cast(TimestampType()))
orders = orders.withColumn("UnitPrice", orders["UnitPrice"].cast(FloatType()))
orders = orders.withColumn("CustomerID", orders["CustomerID"].cast(IntegerType()))
orders = orders.withColumn("InvoiceNo", orders["InvoiceNo"].cast(IntegerType()))
orders = orders.withColumn("Quantity", orders["Quantity"].cast(IntegerType()))

# COMMAND ----------

data = orders.toPandas()

# COMMAND ----------

data.head()

# COMMAND ----------

data.info()

# COMMAND ----------

#data['InvoiceDate2'] = pd.to_datetime(data['InvoiceDate2'])

# COMMAND ----------

from lifetimes.utils import summary_data_from_transaction_data

summary = summary_data_from_transaction_data(data, 'CustomerID', 'InvoiceDate2', observation_period_end='2022-10-01')

print(summary.head())

# COMMAND ----------

from lifetimes import BetaGeoFitter

# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])
print(bgf)

# COMMAND ----------

bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# COMMAND ----------

from lifetimes.plotting import plot_frequency_recency_matrix
fig = plt.figure(figsize=(12,8))
plot_frequency_recency_matrix(bgf)

# COMMAND ----------

from lifetimes.plotting import plot_probability_alive_matrix
fig = plt.figure(figsize=(12,8))
plot_probability_alive_matrix(bgf)

# COMMAND ----------

t = 1
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], summary['T'])
data.sort_values(by='predicted_purchases').tail(5)

# COMMAND ----------

from lifetimes.plotting import plot_period_transactions
fig = plt.figure(figsize=(12,8))
plot_period_transactions(bgf)

# COMMAND ----------

# MAGIC %md
# MAGIC ##More model fitting
# MAGIC With transactional data, we can partition the dataset into a calibration period dataset and a holdout dataset. This is important as we want to test how our model performs on data not yet seen (think cross-validation in standard machine learning literature). Lifetimes has a function to partition our dataset like this:

# COMMAND ----------

from lifetimes.utils import calibration_and_holdout_data

summary_cal_holdout = calibration_and_holdout_data(data, 'CustomerID', 'InvoiceDate2',
                                        calibration_period_end='2022-06-02',
                                        observation_period_end='2022-10-01' )
print(summary_cal_holdout.head())

# COMMAND ----------

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])

# COMMAND ----------

#plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Customer Predictions
# MAGIC Based on customer history, we can predict what an individuals future purchases might look like:

# COMMAND ----------

t = 10 #predict purchases in 10 periods
individual = summary.iloc[20]
# The below function is an alias to `bfg.conditional_expected_number_of_purchases_up_to_time`
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])

# COMMAND ----------

from lifetimes.plotting import plot_history_alive

id = 111
days_since_birth = 148
sp_trans = data.loc[data['CustomerID'] == id]
plot_history_alive(bgf, days_since_birth, sp_trans, 'InvoiceDate2')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Estimating customer lifetime value using the Gamma-Gamma model
# MAGIC For this whole time we didn’t take into account the economic value of each transaction and we focused mainly on transactions’ occurrences. To estimate this we can use the Gamma-Gamma submodel. But first we need to create summary data from transactional data also containing economic values for each transaction (i.e. profits or revenues).

# COMMAND ----------

data['amt'] = data['Quantity'] * data['UnitPrice']

# COMMAND ----------

data.info()

# COMMAND ----------

# set the last transaction date as the end point for this historical dataset
current_date = data['InvoiceDate2'].max()

# calculate the required customer metrics
metrics_pd = (
  lifetimes.utils.summary_data_from_transaction_data(
    data,
    customer_id_col='CustomerID',
    datetime_col='InvoiceDate2',
    observation_period_end = current_date, 
    freq='D',
    monetary_value_col='amt'  # use sales amount to determine monetary value
    )
  )

# display first few rows
metrics_pd.head(10)

# COMMAND ----------

returning_customers_summary = metrics_pd[metrics_pd['frequency']>0]

print(returning_customers_summary.head())

# COMMAND ----------

returning_customers_summary[['monetary_value', 'frequency']].corr()

# COMMAND ----------

# MAGIC %md
# MAGIC At this point we can train our Gamma-Gamma submodel and predict the conditional, expected average lifetime value of our customers.

# COMMAND ----------

from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value'])
print(ggf)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now estimate the average transaction value:

# COMMAND ----------

print(ggf.conditional_expected_average_profit(
        metrics_pd['frequency'],
        metrics_pd['monetary_value']
    ).head(10))

# COMMAND ----------

print("Expected conditional average profit: %s, Average profit: %s" % (
    ggf.conditional_expected_average_profit(
        metrics_pd['frequency'],
        metrics_pd['monetary_value']
    ).mean(),
    metrics_pd[metrics_pd['frequency']>0]['monetary_value'].mean()
))

# COMMAND ----------

# MAGIC %md
# MAGIC While for computing the total CLV using the DCF method (https://en.wikipedia.org/wiki/Discounted_cash_flow) adjusting for cost of capital:

# COMMAND ----------

bgf.fit(metrics_pd['frequency'], metrics_pd['recency'], metrics_pd['T'])

print(ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    metrics_pd['frequency'],
    metrics_pd['recency'],
    metrics_pd['T'],
    metrics_pd['monetary_value'],
    time=365, # days
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
).head(10))

# COMMAND ----------



# COMMAND ----------

