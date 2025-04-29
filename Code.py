#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Group 41

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Loading data

Customers=pd.read_csv("olist_customers_dataset.csv")
Geolocation=pd.read_csv("olist_geolocation_dataset.csv")
Order_items=pd.read_csv("olist_order_items_dataset.csv")
Order_Payments=pd.read_csv("olist_order_payments_dataset.csv")
Order_Reviews=pd.read_csv("olist_order_reviews_dataset.csv")
Order_Status=pd.read_csv("olist_orders_dataset.csv")
Products=pd.read_csv("olist_products_dataset.csv")
Sellers=pd.read_csv("olist_sellers_dataset.csv")
Product_Translations=pd.read_csv("product_category_name_translation.csv")

# Defining a function to calculate the number of working days between purchase and delivery

def calculate_working_days(start, end):
    # Handle missing values
    if pd.isna(start) or pd.isna(end):
        return None
    # Calculate working days using numpy.busday_count
    return np.busday_count(start.date(), end.date())

def plot_total_missing_unique_values(db,db_name):
    
    # Visualise missing values and unique values

    # Calculate metrics
    total_rows = len(db)  # Total rows
    column_names = db.columns

    # Create DataFrame to hold the statistics
    stats = pd.DataFrame({
        'Column': column_names,
        'Total_Rows': total_rows,
        'Non_Missing_Values': db.notnull().sum().values,
        'Unique_Values': db.nunique().values
    })

    # Add Missing Rows
    stats['Missing_Values'] = stats['Total_Rows'] - stats['Non_Missing_Values']

    # Plot a stacked bar chart
    plt.figure(figsize=(12, 7))

    # Calculate bottom positions for stacking
    bottom_non_missing = stats['Non_Missing_Values']
    bottom_missing = stats['Non_Missing_Values'] + stats['Missing_Values']

    # Plot Non-Missing Rows (this will form the bottom part of the bar)
    bars_non_missing = plt.bar(stats['Column'], stats['Non_Missing_Values'], label='Duplicated Values', color='lightsteelblue', edgecolor='black')

    # Plot Missing Rows on top of Non-Missing Rows
    bars_missing = plt.bar(stats['Column'], stats['Missing_Values'], label='Missing Values', bottom=bottom_non_missing, color='gainsboro', edgecolor='black')

    # Plot Unique Values on top of Non-Missing Rows (this segment will show within the Non-Missing Rows)
    bars_unique = plt.bar(stats['Column'], stats['Unique_Values'], label='Unique Values', bottom=0, color='cornflowerblue', alpha=0.7, edgecolor='black')

    # Add labels and title
    plt.title(f'Total Rows, Duplicated, Missing, and Unique Values by Column in {db_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Column', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)

    # Place the legend to the right of the chart
    plt.legend(title='Metric', fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))

    # Add count and percentage labels to the bars
    # for bar, value, bottom in zip(bars_non_missing, stats['Non_Missing_Rows'], [0] * len(stats['Non_Missing_Rows'])):
    #     count = value
    #     percentage = (value / total_rows) * 100
    #     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
    #              f'{count}\n({percentage:.2f}%)', ha='center', va='center', color='black')

    for bar, value, bottom in zip(bars_unique, stats['Unique_Values'], [0] * len(stats['Unique_Values'])):
        count = value
        percentage = (value / total_rows) * 100
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f'{count}\n({percentage:.2f}%)', ha='center', va='center', color='black')

    # Adjust layout to avoid clipping
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Print Table for Reference
    print(stats)


# DATABASES:
#   Customers
#   Geolocation
#   Order_items
#   Order_Payments
#   Order_Reviews
#   Order_Status
#   Products
#   Sellers
#   Product_Translations


# I. DATA INTEGRITY VALIDATION

# i.CUSTOMERS

## Checking Customers database
Customers.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 99441 entries, 0 to 99440
# Data columns (total 5 columns):
#  #   Column                    Non-Null Count  Dtype 
# ---  ------                    --------------  ----- 
#  0   customer_id               99441 non-null  object  -- This is unique related to the order 
#  1   customer_unique_id        99441 non-null  object  -- Related to the customer (contains duplicates)
#  2   customer_zip_code_prefix  99441 non-null  int64   -- 
#  3   customer_city             99441 non-null  object  -- There are multiple cities for 1 zipcode
#  4   customer_state            99441 non-null  object
# dtypes: int64(1), object(4)
# memory usage: 3.8+ MB

## Checking Missing values
Customers.isnull().sum()

# Result in console:
# customer_id                 0
# customer_unique_id          0
# customer_zip_code_prefix    0
# customer_city               0
# customer_state              0
# dtype: int64

## There aren't missing values

## Checking %Missing values
round(Customers.isnull().sum()/len(Customers) * 100,3)

# Result in console:
# customer_id                 0.0
# customer_unique_id          0.0
# customer_zip_code_prefix    0.0
# customer_city               0.0
# customer_state              0.0
# dtype: float64

## There is 0% of missing values.

## Checking Duplicates in all rows
Customers.duplicated().value_counts()

# Result in console:
# False    99441
# Name: count, dtype: int64

## Checking Duplicates in each column
print(Customers.apply(lambda col: col.duplicated().sum()))

# Result in console:
# customer_id                     0
# customer_unique_id           3345
# customer_zip_code_prefix    84447
# customer_city               95322
# customer_state              99414
# dtype: int64

## Checking %Duplicates in each column
print(round(Customers.apply(lambda col: col.duplicated().sum()/len(col))*100,3))

# Result in console:
# customer_id                  0.000
# customer_unique_id           3.364
# customer_zip_code_prefix    84.922
# customer_city               95.858
# customer_state              99.973
# dtype: float64

## Checking number of Unique Values in each column
print(Customers.apply(lambda col: col.nunique()))

# Result in console:
# customer_id                 99441
# customer_unique_id          96096
# customer_zip_code_prefix    14994
# customer_city                4119
# customer_state                 27
# dtype: int64

## Checking % of Unique Values in each column
print(round(Customers.apply(lambda col: col.nunique()/len(col))*100,3))

# Result in console:
# customer_id                 100.000
# customer_unique_id           96.636
# customer_zip_code_prefix     15.078
# customer_city                 4.142
# customer_state                0.027
# dtype: float64

## Visualizations

# Visualization of total rows, missing values and unique values per column

plot_total_missing_unique_values(Customers,"Customers")

# ii.GEOLOCATION

## Checking Geolocation database
Geolocation.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1000163 entries, 0 to 1000162
# Data columns (total 5 columns):
#  #   Column                       Non-Null Count    Dtype  
# ---  ------                       --------------    -----  
#  0   geolocation_zip_code_prefix  1000163 non-null  int64  
#  1   geolocation_lat              1000163 non-null  float64
#  2   geolocation_lng              1000163 non-null  float64
#  3   geolocation_city             1000163 non-null  object 
#  4   geolocation_state            1000163 non-null  object 
# dtypes: float64(2), int64(1), object(2)
# memory usage: 38.2+ MB

## Checking Missing values
Geolocation.isnull().sum()

# Result in console:
# geolocation_zip_code_prefix    0
# geolocation_lat                0
# geolocation_lng                0
# geolocation_city               0
# geolocation_state              0
# dtype: int64

## Checking %Missing values
round(Geolocation.isnull().sum()/len(Geolocation) * 100,3)

# Result in console:
# geolocation_zip_code_prefix    0.0
# geolocation_lat                0.0
# geolocation_lng                0.0
# geolocation_city               0.0
# geolocation_state              0.0
# dtype: float64

## Checking Duplicates in all rows
Geolocation.duplicated().value_counts()

# Result in console:
# False    738332
# True     261831
# Name: count, dtype: int64

## Checking Duplicates in each column
print(Geolocation.apply(lambda col: col.duplicated().sum()))

# Result in console:
# geolocation_zip_code_prefix     981148
# geolocation_lat                 282800
# geolocation_lng                 282548
# geolocation_city                992152
# geolocation_state              1000136
# dtype: int64

## Checking %Duplicates in each column
print(round(Geolocation.apply(lambda col: col.duplicated().sum()/len(col))*100,3))

# Result in console:
# geolocation_zip_code_prefix    98.099
# geolocation_lat                28.275
# geolocation_lng                28.250
# geolocation_city               99.199
# geolocation_state              99.997
# dtype: float64

## Checking number of Unique Values in each column
print(Geolocation.apply(lambda col: col.nunique()))

# Result in console:
# geolocation_zip_code_prefix     19015
# geolocation_lat                717363
# geolocation_lng                717615
# geolocation_city                 8011
# geolocation_state                  27
# dtype: int64

## Checking % of Unique Values in each column
print(round(Geolocation.apply(lambda col: col.nunique()/len(col))*100,3))

# Result in console:
# geolocation_zip_code_prefix     1.901
# geolocation_lat                71.725
# geolocation_lng                71.750
# geolocation_city                0.801
# geolocation_state               0.003
# dtype: float64

## Visualizations

plot_total_missing_unique_values(Geolocation,"Geolocation")


# iii.ORDER_ITEMS

## Checking Order_Items database
Order_items.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 112650 entries, 0 to 112649
# Data columns (total 7 columns):
#  #   Column               Non-Null Count   Dtype  
# ---  ------               --------------   -----  
#  0   order_id             112650 non-null  object 
#  1   order_item_id        112650 non-null  int64  
#  2   product_id           112650 non-null  object 
#  3   seller_id            112650 non-null  object 
#  4   shipping_limit_date  112650 non-null  object 
#  5   price                112650 non-null  float64
#  6   freight_value        112650 non-null  float64
# dtypes: float64(2), int64(1), object(4)
# memory usage: 6.0+ MB

## Change formats to datetime
Order_items['shipping_limit_date']=pd.to_datetime(Order_items['shipping_limit_date'], errors='coerce')

## Checking Order_Items database after change of format
Order_items.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 112650 entries, 0 to 112649
# Data columns (total 9 columns):
#  #   Column                  Non-Null Count   Dtype         
# ---  ------                  --------------   -----         
#  0   order_id                112650 non-null  object        
#  1   order_item_id           112650 non-null  int64         
#  2   product_id              112650 non-null  object        
#  3   seller_id               112650 non-null  object        
#  4   shipping_limit_date     112650 non-null  datetime64[ns]
#  5   price                   112650 non-null  float64       
#  6   freight_value           112650 non-null  float64       
#  7   product_payment_value   112650 non-null  float64       
#  8   freight_to_price_ratio  112650 non-null  float64       
# dtypes: datetime64[ns](1), float64(4), int64(1), object(3)
# memory usage: 7.7+ MB

## Checking Missing values
Order_items.isnull().sum()

# Result in console:
# order_id               0
# order_item_id          0
# product_id             0
# seller_id              0
# shipping_limit_date    0
# price                  0
# freight_value          0
# dtype: int64

## Checking %Missing values
round(Order_items.isnull().sum()/len(Order_items) * 100,2)

# Result in console:
# order_id               0.0
# order_item_id          0.0
# product_id             0.0
# seller_id              0.0
# shipping_limit_date    0.0
# price                  0.0
# freight_value          0.0
# dtype: float64

## Checking Duplicates in all rows
Order_items.duplicated().value_counts()

# Result in console:
# False    112650
# Name: count, dtype: int64

## Checking Duplicates in each column
print(Order_items.apply(lambda col: col.duplicated().sum()))

# Result in console:
# order_id                13984
# order_item_id          112629
# product_id              79699
# seller_id              109555
# shipping_limit_date     19332
# price                  106682
# freight_value          105651
# dtype: int64

## Checking %Duplicates in each column
print(round(Order_items.apply(lambda col: col.duplicated().sum()/len(col))*100,3))

# Result in console:
# order_id               12.414
# order_item_id          99.981
# product_id             70.749
# seller_id              97.253
# shipping_limit_date    17.161
# price                  94.702
# freight_value          93.787
# dtype: float64

## Checking number of Unique Values in each column
print(Order_items.apply(lambda col: col.nunique()))

# Result in console:
# order_id               98666
# order_item_id             21
# product_id             32951
# seller_id               3095
# shipping_limit_date    93318
# price                   5968
# freight_value           6999
# dtype: int64

## Checking % of Unique Values in each column
print(round(Order_items.apply(lambda col: col.nunique()/len(col))*100,3))

# Result in console:
# order_id               87.586
# order_item_id           0.019
# product_id             29.251
# seller_id               2.747
# shipping_limit_date    82.839
# price                   5.298
# freight_value           6.213
# dtype: float64

## Visualizations

# Visualization of total rows, missing values and unique values per column

plot_total_missing_unique_values(Order_items,"Order_items")

# iv.ORDER_PAYMENTS

## Checking Order_Payments database
Order_Payments.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 103886 entries, 0 to 103885
# Data columns (total 5 columns):
#  #   Column                Non-Null Count   Dtype  
# ---  ------                --------------   -----  
#  0   order_id              103886 non-null  object 
#  1   payment_sequential    103886 non-null  int64  
#  2   payment_type          103886 non-null  object 
#  3   payment_installments  103886 non-null  int64  
#  4   payment_value         103886 non-null  float64
# dtypes: float64(1), int64(2), object(2)
# memory usage: 4.0+ MB

## Checking Missing values
Order_Payments.isnull().sum()

# Result in console:
# order_id                0
# payment_sequential      0
# payment_type            0
# payment_installments    0
# payment_value           0
# dtype: int64

## Checking %Missing values
round(Order_Payments.isnull().sum()/len(Order_Payments) * 100,3)

# Result in console:
# order_id                0.0
# payment_sequential      0.0
# payment_type            0.0
# payment_installments    0.0
# payment_value           0.0
# dtype: float64

## Checking Duplicates in all rows
Order_Payments.duplicated().value_counts()

# Result in console:
# False    103886
# Name: count, dtype: int64

## Checking Duplicates in each column
print(Order_Payments.apply(lambda col: col.duplicated().sum()))

# Result in console:
# order_id                  4446
# payment_sequential      103857
# payment_type            103881
# payment_installments    103862
# payment_value            74809
# dtype: int64

## Checking %Duplicates in each column
print(round(Order_Payments.apply(lambda col: col.duplicated().sum()/len(col))*100,3))

# Result in console:
# order_id                 4.280
# payment_sequential      99.972
# payment_type            99.995
# payment_installments    99.977
# payment_value           72.011
# dtype: float64

## Checking number of Unique Values in each column
print(Order_Payments.apply(lambda col: col.nunique()))

# Result in console:
# order_id                99440
# payment_sequential         29
# payment_type                5
# payment_installments       24
# payment_value           29077
# dtype: int64

## Checking % of Unique Values in each column
print(round(Order_Payments.apply(lambda col: col.nunique()/len(col))*100,3))

# Result in console:
# order_id                95.720
# payment_sequential       0.028
# payment_type             0.005
# payment_installments     0.023
# payment_value           27.989
# dtype: float64

## Visualizations

# Visualization of total rows, missing values and unique values per column

plot_total_missing_unique_values(Order_Payments,"Order_Payments")

# v.ORDER_REVIEWS

## Checking Order_Reviews database
Order_Reviews.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 89999 entries, 0 to 89998
# Data columns (total 7 columns):
#  #   Column                   Non-Null Count  Dtype 
# ---  ------                   --------------  ----- 
#  0   review_id                89999 non-null  object
#  1   order_id                 89999 non-null  object
#  2   review_score             89999 non-null  int64 
#  3   review_comment_title     10595 non-null  object
#  4   review_comment_message   37570 non-null  object
#  5   review_creation_date     89999 non-null  object
#  6   review_answer_timestamp  89999 non-null  object
# dtypes: int64(1), object(6)
# memory usage: 4.8+ MB

## Change formats to datetime
Order_Reviews['review_creation_date']=pd.to_datetime(Order_Reviews['review_creation_date'], errors='coerce')
Order_Reviews['review_answer_timestamp']=pd.to_datetime(Order_Reviews['review_answer_timestamp'], errors='coerce')

## Checking Order_Reviews database after changes
Order_Reviews.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 89999 entries, 0 to 89998
# Data columns (total 7 columns):
#  #   Column                   Non-Null Count  Dtype         
# ---  ------                   --------------  -----         
#  0   review_id                89999 non-null  object        
#  1   order_id                 89999 non-null  object        
#  2   review_score             89999 non-null  int64         
#  3   review_comment_title     10595 non-null  object        
#  4   review_comment_message   37570 non-null  object        
#  5   review_creation_date     89999 non-null  datetime64[ns]
#  6   review_answer_timestamp  89999 non-null  datetime64[ns]
# dtypes: datetime64[ns](2), int64(1), object(4)
# memory usage: 4.8+ MB

## Checking Missing values
Order_Reviews.isnull().sum()

# Result in console:
# review_id                      0
# order_id                       0
# review_score                   0
# review_comment_title       79404
# review_comment_message     52429
# review_creation_date           0
# review_answer_timestamp        0
# dtype: int64

## Checking %Missing values
round(Order_Reviews.isnull().sum()/len(Order_Reviews) * 100,3)

# Result in console:
# review_id                   0.000
# order_id                    0.000
# review_score                0.000
# review_comment_title       88.228
# review_comment_message     58.255
# review_creation_date        0.000
# review_answer_timestamp     0.000
# dtype: float64

## Checking Duplicates in all rows
Order_Reviews.duplicated().value_counts()

# Result in console:
# False    89999
# Name: count, dtype: int64

## Checking Duplicates in each column
print(Order_Reviews.apply(lambda col: col.duplicated().sum()))

# Result in console:
# review_id                    677
# order_id                     447
# review_score               89994
# review_comment_title       85775
# review_comment_message     56674
# review_creation_date       89363
# review_answer_timestamp     8602
# dtype: int64

## Checking %Duplicates in each column
print(round(Order_Reviews.apply(lambda col: col.duplicated().sum()/len(col))*100,3))

# Result in console:
# review_id                   0.752
# order_id                    0.497
# review_score               99.994
# review_comment_title       95.307
# review_comment_message     62.972
# review_creation_date       99.293
# review_answer_timestamp     9.558
# dtype: float64

## Checking number of Unique Values in each column
print(Order_Reviews.apply(lambda col: col.nunique()))

# Result in console:
# review_id                  89322
# order_id                   89552
# review_score                   5
# review_comment_title        4223
# review_comment_message     33324
# review_creation_date         636
# review_answer_timestamp    81397
# dtype: int64

## Checking % of Unique Values in each column
print(round(Order_Reviews.apply(lambda col: col.nunique()/len(col))*100,3))

# Result in console:
# review_id                  99.248
# order_id                   99.503
# review_score                0.006
# review_comment_title        4.692
# review_comment_message     37.027
# review_creation_date        0.707
# review_answer_timestamp    90.442
# dtype: float64

## Visualizations

## Visualization of total rows, missing values and unique values per column

plot_total_missing_unique_values(Order_Reviews,"Order_Reviews")


# vi. ORDER_STATUS

## Checking Order_Status database
Order_Status.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 99441 entries, 0 to 99440
# Data columns (total 8 columns):
#  #   Column                         Non-Null Count  Dtype 
# ---  ------                         --------------  ----- 
#  0   order_id                       99441 non-null  object
#  1   customer_id                    99441 non-null  object
#  2   order_status                   99441 non-null  object
#  3   order_purchase_timestamp       99441 non-null  object
#  4   order_approved_at              99281 non-null  object
#  5   order_delivered_carrier_date   97658 non-null  object
#  6   order_delivered_customer_date  96476 non-null  object
#  7   order_estimated_delivery_date  99441 non-null  object
# dtypes: object(8)
# memory usage: 6.1+ MB

## Change formats to datetime
Order_Status['order_purchase_timestamp']=pd.to_datetime(Order_Status['order_purchase_timestamp'], errors='coerce')
Order_Status['order_approved_at']=pd.to_datetime(Order_Status['order_approved_at'], errors='coerce')
Order_Status['order_delivered_carrier_date']=pd.to_datetime(Order_Status['order_delivered_carrier_date'], errors='coerce')
Order_Status['order_delivered_customer_date']=pd.to_datetime(Order_Status['order_delivered_customer_date'], errors='coerce')
Order_Status['order_estimated_delivery_date']=pd.to_datetime(Order_Status['order_estimated_delivery_date'], errors='coerce')

## Checking Order_Status database after changes
Order_Status.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 99441 entries, 0 to 99440
# Data columns (total 8 columns):
#  #   Column                         Non-Null Count  Dtype         
# ---  ------                         --------------  -----         
#  0   order_id                       99441 non-null  object        
#  1   customer_id                    99441 non-null  object        
#  2   order_status                   99441 non-null  object        
#  3   order_purchase_timestamp       99441 non-null  datetime64[ns]
#  4   order_approved_at              99281 non-null  datetime64[ns]
#  5   order_delivered_carrier_date   97658 non-null  datetime64[ns]
#  6   order_delivered_customer_date  96476 non-null  datetime64[ns]
#  7   order_estimated_delivery_date  99441 non-null  datetime64[ns]
# dtypes: datetime64[ns](5), object(3)
# memory usage: 6.1+ MB

## Checking Missing values
Order_Status.isnull().sum()

# Result in console:
# order_id                            0
# customer_id                         0
# order_status                        0
# order_purchase_timestamp            0
# order_approved_at                 160
# order_delivered_carrier_date     1783
# order_delivered_customer_date    2965
# order_estimated_delivery_date       0
# dtype: int64

## Checking %Missing values
round(Order_Status.isnull().sum()/len(Order_Status) * 100,3)

# Result in console:
# order_id                         0.000
# customer_id                      0.000
# order_status                     0.000
# order_purchase_timestamp         0.000
# order_approved_at                0.161
# order_delivered_carrier_date     1.793
# order_delivered_customer_date    2.982
# order_estimated_delivery_date    0.000
# dtype: float64

## Checking Duplicates in all rows
Order_Status.duplicated().value_counts()

# Result in console:
# False    99441
# Name: count, dtype: int64

## Checking Duplicates in each column
print(Order_Status.apply(lambda col: col.duplicated().sum()))

# Result in console:
# order_id                             0
# customer_id                          0
# order_status                     99433
# order_purchase_timestamp           566
# order_approved_at                 8707
# order_delivered_carrier_date     18422
# order_delivered_customer_date     3776
# order_estimated_delivery_date    98982
# dtype: int64


## Checking %Duplicates in each column
print(round(Order_Status.apply(lambda col: col.duplicated().sum()/len(col))*100,3))

# Result in console:
# order_id                          0.000
# customer_id                       0.000
# order_status                     99.992
# order_purchase_timestamp          0.569
# order_approved_at                 8.756
# order_delivered_carrier_date     18.526
# order_delivered_customer_date     3.797
# order_estimated_delivery_date    99.538
# dtype: float64

## Checking number of Unique Values in each column
print(Order_Status.apply(lambda col: col.nunique()))

# Result in console:
# order_id                         99441
# customer_id                      99441
# order_status                         8
# order_purchase_timestamp         98875
# order_approved_at                90733
# order_delivered_carrier_date     81018
# order_delivered_customer_date    95664
# order_estimated_delivery_date      459
# dtype: int64

## Checking % of Unique Values in each column
print(round(Order_Status.apply(lambda col: col.nunique()/len(col))*100,3))

# Result in console:
# order_id                         100.000
# customer_id                      100.000
# order_status                       0.008
# order_purchase_timestamp          99.431
# order_approved_at                 91.243
# order_delivered_carrier_date      81.473
# order_delivered_customer_date     96.202
# order_estimated_delivery_date      0.462
# dtype: float64

## Visualizations

## Visualization of total rows, missing values and unique values per column

plot_total_missing_unique_values(Order_Status,"Order_Status")

# vii. PRODUCTS

## Checking Products database
Products.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 32951 entries, 0 to 32950
# Data columns (total 9 columns):
#  #   Column                      Non-Null Count  Dtype  
# ---  ------                      --------------  -----  
#  0   product_id                  32951 non-null  object 
#  1   product_category_name       32341 non-null  object 
#  2   product_name_lenght         32341 non-null  float64
#  3   product_description_lenght  32341 non-null  float64
#  4   product_photos_qty          32341 non-null  float64
#  5   product_weight_g            32949 non-null  float64
#  6   product_length_cm           32949 non-null  float64
#  7   product_height_cm           32949 non-null  float64
#  8   product_width_cm            32949 non-null  float64
# dtypes: float64(7), object(2)
# memory usage: 2.3+ MB

## Checking Missing values
Products.isnull().sum()

# Result in console:
# product_id                      0
# product_category_name         610
# product_name_lenght           610
# product_description_lenght    610
# product_photos_qty            610
# product_weight_g                2
# product_length_cm               2
# product_height_cm               2
# product_width_cm                2
# dtype: int64

## Checking %Missing values
round(Products.isnull().sum()/len(Products) * 100,3)

# Result in console:
# product_id                    0.000
# product_category_name         1.851
# product_name_lenght           1.851
# product_description_lenght    1.851
# product_photos_qty            1.851
# product_weight_g              0.006
# product_length_cm             0.006
# product_height_cm             0.006
# product_width_cm              0.006
# dtype: float64

## Checking Duplicates in all rows
Products.duplicated().value_counts()

# Result in console:
# False    32951
# Name: count, dtype: int64

## Checking Duplicates in each column
print(Products.apply(lambda col: col.duplicated().sum()))

# Result in console:
# product_id                        0
# product_category_name         32877
# product_name_lenght           32884
# product_description_lenght    29990
# product_photos_qty            32931
# product_weight_g              30746
# product_length_cm             32851
# product_height_cm             32848
# product_width_cm              32855
# P_Flg_NA_Category             32949
# P_Flg_NA_Photos               32949
# P_Flg_NA_Dimentions_Weight    32950
# P_Flg_NA                      32950
# dtype: int64

## Checking %Duplicates in each column
print(round(Products.apply(lambda col: col.duplicated().sum()/len(col))*100,3))

# Result in console:
# product_id                     0.000
# product_category_name         99.775
# product_name_lenght           99.797
# product_description_lenght    91.014
# product_photos_qty            99.939
# product_weight_g              93.308
# product_length_cm             99.697
# product_height_cm             99.687
# product_width_cm              99.709
# dtype: float64

## Checking number of Unique Values in each column
print(Products.apply(lambda col: col.nunique()))

# Result in console:
# product_id                    32951
# product_category_name            73
# product_name_lenght              66
# product_description_lenght     2960
# product_photos_qty               19
# product_weight_g               2204
# product_length_cm                99
# product_height_cm               102
# product_width_cm                 95
# dtype: int64

## Checking % of Unique Values in each column
print(round(Products.apply(lambda col: col.nunique()/len(col))*100,2))

# Result in console:
# product_id                    32951
# product_category_name            73
# product_name_lenght              66
# product_description_lenght     2960
# product_photos_qty               19
# product_weight_g               2204
# product_length_cm                99
# product_height_cm               102
# product_width_cm                 95
# dtype: int64

## Visualizations

# Visualization of total rows, missing values and unique values per column

plot_total_missing_unique_values(Products,"Products")

# viii.SELLERS

## Checking Sellers database
Sellers.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 3095 entries, 0 to 3094
# Data columns (total 4 columns):
#  #   Column                  Non-Null Count  Dtype 
# ---  ------                  --------------  ----- 
#  0   seller_id               3095 non-null   object
#  1   seller_zip_code_prefix  3095 non-null   int64 
#  2   seller_city             3095 non-null   object
#  3   seller_state            3095 non-null   object
# dtypes: int64(1), object(3)
# memory usage: 96.8+ KB

## Checking Missing values
Sellers.isnull().sum()

# Result in console:
# seller_id                 0
# seller_zip_code_prefix    0
# seller_city               0
# seller_state              0
# dtype: int64

## Checking %Missing values
round(Sellers.isnull().sum()/len(Sellers) * 100,3)

# Result in console:
# seller_id                 0.0
# seller_zip_code_prefix    0.0
# seller_city               0.0
# seller_state              0.0
# dtype: float64

## Creating columns to mark Missing values
# There are no missing values

## Checking Duplicates in all rows
Sellers.duplicated().value_counts()

# Result in console:
# False    3095
# Name: count, dtype: int64

## Checking Duplicates in each column
print(Sellers.apply(lambda col: col.duplicated().sum()))

# Result in console:
# seller_id                        0
# seller_zip_code_prefix         849
# seller_city                   2484
# seller_state                  3072
# dtype: int64

## Checking number of Unique Values in each column
print(Sellers.apply(lambda col: col.nunique()))

# Result in console:
# seller_id                 3095
# seller_zip_code_prefix    2246
# seller_city                611
# seller_state                23
# dtype: int64

## Checking % of Unique Values in each column
print(round(Sellers.apply(lambda col: col.nunique()/len(col))*100,3))

# Result in console:
# seller_id                 100.000
# seller_zip_code_prefix     72.569
# seller_city                19.742
# seller_state                0.743
# dtype: float64

## Visualizations

# Visualization of total rows, missing values and unique values per column

plot_total_missing_unique_values(Sellers,"Sellers")

# PRODUCT CATEGORY

# II. MERGING DATA

#  Consolidation of databases to connect databases

# Consolidating Order_Payments to have 1 row per order_id
Order_Payments2 = Order_Payments.groupby(
    ["order_id"]
).agg(
    payment_sequential=("payment_sequential", "count"),
    payment_installments=("payment_installments", "mean"),
    payment_value=("payment_value", "sum"),
    payment_type_count=("payment_type", "nunique")
).reset_index()

# Consolidating Order_Reviews keeping the latest value for each order_id based on review_creation_date
Order_Reviews2 = Order_Reviews.loc[
    Order_Reviews.groupby('order_id')['review_creation_date'].idxmax()
]

# Creating a first database from Order_items -- 112650
database1= Order_items

# Connecting database1 to Products -- 112650
database1=database1.merge(Products, how='left', left_on='product_id', right_on='product_id')

# Connecting database1 to Sellers -- 112650
database1=database1.merge(Sellers, how='left', left_on='seller_id', right_on='seller_id')

# Connecting database1 to Order_Status -- 112650
database1=database1.merge(Order_Status, how='left', left_on='order_id', right_on='order_id')

# Connecting Order_Payments2 to database1 -- 112650
database1=database1.merge(Order_Payments2, how='left', left_on='order_id', right_on='order_id')

# Connecting database1 to Order_Reviews2 -- 112650
database1=database1.merge(Order_Reviews2, how='left', left_on='order_id', right_on='order_id')

# Connecting database1 to Customers
database1=database1.merge(Customers, how='left', left_on='customer_id', right_on='customer_id')

# III. CREATING NEW FEATURES FOR MODELLING

# Loading database Product_Categories
Product_Categories=pd.read_csv("refined_product_categories.csv")

# Connecting database1 to Product_Categories
Final_database=database1.merge(Product_Categories, how='left', left_on='product_category_name', right_on='product_category_name')

# Total_purchase_count as the total number of purchases of the customer
Final_database['total_purchase_count'] = Final_database.groupby('customer_unique_id')['order_id'].transform('count')

# Product payment value as the sum of price and freight value
Final_database['product_payment_value']=Final_database['price']+Final_database['freight_value']

# Freight_to_price_ratio as the ratio of Freight over Price
Final_database['freight_to_price_ratio'] = Final_database['freight_value'] / Final_database['price']

# Days from Creation of the review to Answer
Final_database['diff_review_creation_answer_days']=(
    Final_database['review_answer_timestamp']-Final_database['review_creation_date']
    ).dt.days

# Days from Purchase to Approved
Final_database['diff_approved_purchased'] = (
    Final_database['order_approved_at'] - Final_database['order_purchase_timestamp']
    ).dt.days

# Days from Estimated Delivery to Customer Delivered
Final_database['diff_customerdelivered_estimated'] = (
    Final_database['order_delivered_customer_date'] - Final_database['order_estimated_delivery_date']
    ).dt.days

# Days from Carrier to Customer
Final_database['diff_customerdelivered_deliveredcarrier'] = (
    Final_database['order_delivered_customer_date'] - Final_database['order_delivered_carrier_date']
    ).dt.days

# Days from Purchase to Customer Delivered
Final_database['diff_customerdelivered_purchase'] = (
    Final_database['order_delivered_customer_date'] - Final_database['order_purchase_timestamp']
    ).dt.days

# Days from Purchase to Delivered Carrier
Final_database['diff_deliveredcarrier_purchase'] = (
    Final_database['order_delivered_carrier_date'] - Final_database['order_purchase_timestamp']
    ).dt.days

# Working days from Purchased to Approved
Final_database['diff_approved_purchased_wd'] = Final_database.apply(
    lambda row: calculate_working_days(row['order_purchase_timestamp'], row['order_approved_at']),
    axis=1
)

# Working days from Carrier to Customer Delivered
Final_database['diff_customerdelivered_deliveredcarrier_wd'] = Final_database.apply(
    lambda row: calculate_working_days(row['order_delivered_carrier_date'], row['order_delivered_customer_date']),
    axis=1
)

# Working days from Purchase to Delivered Carrier
Final_database['diff_deliveredcarrier_purchase_wd'] = Final_database.apply(
    lambda row: calculate_working_days(row['order_purchase_timestamp'], row['order_delivered_carrier_date']),
    axis=1
)

# Creating index Order_id + Product_id
Final_database['order_id_product_id'] = Final_database['order_id'] +'-'+ Final_database['product_id'] 

# iv. FINAL EDA

# Checking Final_database
Final_database.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 112650 entries, 0 to 112649
# Data columns (total 54 columns):
#  #   Column                                      Non-Null Count   Dtype         
# ---  ------                                      --------------   -----         
#  0   order_id                                    112650 non-null  object        
#  1   order_item_id                               112650 non-null  int64         
#  2   product_id                                  112650 non-null  object        
#  3   seller_id                                   112650 non-null  object        
#  4   shipping_limit_date                         112650 non-null  datetime64[ns]
#  5   price                                       112650 non-null  float64       
#  6   freight_value                               112650 non-null  float64       
#  7   product_category_name                       111047 non-null  object        
#  8   product_name_lenght                         111047 non-null  float64       
#  9   product_description_lenght                  111047 non-null  float64       
#  10  product_photos_qty                          111047 non-null  float64       
#  11  product_weight_g                            112632 non-null  float64       
#  12  product_length_cm                           112632 non-null  float64       
#  13  product_height_cm                           112632 non-null  float64       
#  14  product_width_cm                            112632 non-null  float64       
#  15  seller_zip_code_prefix                      112650 non-null  int64         
#  16  seller_city                                 112650 non-null  object        
#  17  seller_state                                112650 non-null  object        
#  18  customer_id                                 112650 non-null  object        
#  19  order_status                                112650 non-null  object        
#  20  order_purchase_timestamp                    112650 non-null  datetime64[ns]
#  21  order_approved_at                           112635 non-null  datetime64[ns]
#  22  order_delivered_carrier_date                111456 non-null  datetime64[ns]
#  23  order_delivered_customer_date               110196 non-null  datetime64[ns]
#  24  order_estimated_delivery_date               112650 non-null  datetime64[ns]
#  25  payment_sequential                          112647 non-null  float64       
#  26  payment_installments                        112647 non-null  float64       
#  27  payment_value                               112647 non-null  float64       
#  28  payment_type_count                          112647 non-null  float64       
#  29  review_id                                   101539 non-null  object        
#  30  review_score                                101539 non-null  float64       
#  31  review_comment_title                        12323 non-null   object        
#  32  review_comment_message                      43553 non-null   object        
#  33  review_creation_date                        101539 non-null  datetime64[ns]
#  34  review_answer_timestamp                     101539 non-null  datetime64[ns]
#  35  customer_unique_id                          112650 non-null  object        
#  36  customer_zip_code_prefix                    112650 non-null  int64         
#  37  customer_city                               112650 non-null  object        
#  38  customer_state                              112650 non-null  object        
#  39  product_category_name_english               111023 non-null  object        
#  40  Category                                    111023 non-null  object        
#  41  total_purchase_count                        112650 non-null  int64         
#  42  product_payment_value                       112650 non-null  float64       
#  43  freight_to_price_ratio                      112650 non-null  float64       
#  44  diff_review_creation_answer_days            101539 non-null  float64       
#  45  diff_approved_purchased                     112635 non-null  float64       
#  46  diff_customerdelivered_estimated            110196 non-null  float64       
#  47  diff_customerdelivered_deliveredcarrier     110195 non-null  float64       
#  48  diff_customerdelivered_purchase             110196 non-null  float64       
#  49  diff_deliveredcarrier_purchase              111456 non-null  float64       
#  50  diff_approved_purchased_wd                  112635 non-null  float64       
#  51  diff_customerdelivered_deliveredcarrier_wd  110195 non-null  float64       
#  52  diff_deliveredcarrier_purchase_wd           111456 non-null  float64       
#  53  order_id_product_id                         112650 non-null  object        
# dtypes: datetime64[ns](8), float64(25), int64(4), object(17)
# memory usage: 46.4+ MB

## Checking Missing values
Final_database.isnull().sum()

# Result in console:
# order_id                                           0
# order_item_id                                      0
# product_id                                         0
# seller_id                                          0
# shipping_limit_date                                0
# price                                              0
# freight_value                                      0
# product_category_name                           1603
# product_name_lenght                             1603
# product_description_lenght                      1603
# product_photos_qty                              1603
# product_weight_g                                  18
# product_length_cm                                 18
# product_height_cm                                 18
# product_width_cm                                  18
# seller_zip_code_prefix                             0
# seller_city                                        0
# seller_state                                       0
# customer_id                                        0
# order_status                                       0
# order_purchase_timestamp                           0
# order_approved_at                                 15
# order_delivered_carrier_date                    1194
# order_delivered_customer_date                   2454
# order_estimated_delivery_date                      0
# payment_sequential                                 3
# payment_installments                               3
# payment_value                                      3
# payment_type_count                                 3
# review_id                                      11111
# review_score                                   11111
# review_comment_title                          100327
# review_comment_message                         69097
# review_creation_date                           11111
# review_answer_timestamp                        11111
# customer_unique_id                                 0
# customer_zip_code_prefix                           0
# customer_city                                      0
# customer_state                                     0
# product_category_name_english                   1627
# Category                                        1627
# total_purchase_count                               0
# product_payment_value                              0
# freight_to_price_ratio                             0
# diff_review_creation_answer_days               11111
# diff_approved_purchased                           15
# diff_customerdelivered_estimated                2454
# diff_customerdelivered_deliveredcarrier         2455
# diff_customerdelivered_purchase                 2454
# diff_deliveredcarrier_purchase                  1194
# diff_approved_purchased_wd                        15
# diff_customerdelivered_deliveredcarrier_wd      2455
# diff_deliveredcarrier_purchase_wd               1194
# order_id_product_id                                0
# dtype: int64

## Checking %Missing values
round(Final_database.isnull().sum()/len(Final_database) * 100,3)

# Result in console:
# order_id                                       0.000
# order_item_id                                  0.000
# product_id                                     0.000
# seller_id                                      0.000
# shipping_limit_date                            0.000
# price                                          0.000
# freight_value                                  0.000
# product_category_name                          1.423
# product_name_lenght                            1.423
# product_description_lenght                     1.423
# product_photos_qty                             1.423
# product_weight_g                               0.016
# product_length_cm                              0.016
# product_height_cm                              0.016
# product_width_cm                               0.016
# seller_zip_code_prefix                         0.000
# seller_city                                    0.000
# seller_state                                   0.000
# customer_id                                    0.000
# order_status                                   0.000
# order_purchase_timestamp                       0.000
# order_approved_at                              0.013
# order_delivered_carrier_date                   1.060
# order_delivered_customer_date                  2.178
# order_estimated_delivery_date                  0.000
# payment_sequential                             0.003
# payment_installments                           0.003
# payment_value                                  0.003
# payment_type_count                             0.003
# review_id                                      9.863
# review_score                                   9.863
# review_comment_title                          89.061
# review_comment_message                        61.338
# review_creation_date                           9.863
# review_answer_timestamp                        9.863
# customer_unique_id                             0.000
# customer_zip_code_prefix                       0.000
# customer_city                                  0.000
# customer_state                                 0.000
# product_category_name_english                  1.444
# Category                                       1.444
# total_purchase_count                           0.000
# product_payment_value                          0.000
# freight_to_price_ratio                         0.000
# diff_review_creation_answer_days               9.863
# diff_approved_purchased                        0.013
# diff_customerdelivered_estimated               2.178
# diff_customerdelivered_deliveredcarrier        2.179
# diff_customerdelivered_purchase                2.178
# diff_deliveredcarrier_purchase                 1.060
# diff_approved_purchased_wd                     0.013
# diff_customerdelivered_deliveredcarrier_wd     2.179
# diff_deliveredcarrier_purchase_wd              1.060
# order_id_product_id                            0.000
# dtype: float64

## Checking Duplicates in all rows
Final_database.duplicated().value_counts()

# Result in console:
# False    112650
# Name: count, dtype: int64

## Checking Duplicates in each column
print(Final_database.apply(lambda col: col.duplicated().sum()))

# Result in console:
# order_id                                       13984
# order_item_id                                 112629
# product_id                                     79699
# seller_id                                     109555
# shipping_limit_date                            19332
# price                                         106682
# freight_value                                 105651
# product_category_name                         112576
# product_name_lenght                           112583
# product_description_lenght                    109689
# product_photos_qty                            112630
# product_weight_g                              110445
# product_length_cm                             112550
# product_height_cm                             112547
# product_width_cm                              112554
# seller_zip_code_prefix                        110404
# seller_city                                   112039
# seller_state                                  112627
# customer_id                                    13984
# order_status                                  112643
# order_purchase_timestamp                       14538
# order_approved_at                              22475
# order_delivered_carrier_date                   31632
# order_delivered_customer_date                  16985
# order_estimated_delivery_date                 112200
# payment_sequential                            112629
# payment_installments                          112598
# payment_value                                  84804
# payment_type_count                            112647
# review_id                                      24232
# review_score                                  112644
# review_comment_title                          108456
# review_comment_message                         79856
# review_creation_date                          112016
# review_answer_timestamp                        32027
# customer_unique_id                             17230
# customer_zip_code_prefix                       97674
# customer_city                                 108540
# customer_state                                112623
# product_category_name_english                 112578
# Category                                      112639
# total_purchase_count                          112630
# product_payment_value                          83501
# freight_to_price_ratio                         65988
# diff_review_creation_answer_days              112448
# diff_approved_purchased                       112629
# diff_customerdelivered_estimated              112451
# diff_customerdelivered_deliveredcarrier       112504
# diff_customerdelivered_purchase               112503
# diff_deliveredcarrier_purchase                112586
# diff_approved_purchased_wd                    112634
# diff_customerdelivered_deliveredcarrier_wd    112535
# diff_deliveredcarrier_purchase_wd             112601
# order_id_product_id                            10225
# dtype: int64

## Checking %Duplicates in each column
print(round(Final_database.apply(lambda col: col.duplicated().sum()/len(col))*100,3))

# Result in console:
# order_id                                      12.414
# order_item_id                                 99.981
# product_id                                    70.749
# seller_id                                     97.253
# shipping_limit_date                           17.161
# price                                         94.702
# freight_value                                 93.787
# product_category_name                         99.934
# product_name_lenght                           99.941
# product_description_lenght                    97.372
# product_photos_qty                            99.982
# product_weight_g                              98.043
# product_length_cm                             99.911
# product_height_cm                             99.909
# product_width_cm                              99.915
# seller_zip_code_prefix                        98.006
# seller_city                                   99.458
# seller_state                                  99.980
# customer_id                                   12.414
# order_status                                  99.994
# order_purchase_timestamp                      12.905
# order_approved_at                             19.951
# order_delivered_carrier_date                  28.080
# order_delivered_customer_date                 15.078
# order_estimated_delivery_date                 99.601
# payment_sequential                            99.981
# payment_installments                          99.954
# payment_value                                 75.281
# payment_type_count                            99.997
# review_id                                     21.511
# review_score                                  99.995
# review_comment_title                          96.277
# review_comment_message                        70.889
# review_creation_date                          99.437
# review_answer_timestamp                       28.431
# customer_unique_id                            15.295
# customer_zip_code_prefix                      86.706
# customer_city                                 96.352
# customer_state                                99.976
# product_category_name_english                 99.936
# Category                                      99.990
# total_purchase_count                          99.982
# product_payment_value                         74.124
# freight_to_price_ratio                        58.578
# diff_review_creation_answer_days              99.821
# diff_approved_purchased                       99.981
# diff_customerdelivered_estimated              99.823
# diff_customerdelivered_deliveredcarrier       99.870
# diff_customerdelivered_purchase               99.870
# diff_deliveredcarrier_purchase                99.943
# diff_approved_purchased_wd                    99.986
# diff_customerdelivered_deliveredcarrier_wd    99.898
# diff_deliveredcarrier_purchase_wd             99.957
# order_id_product_id                            9.077
# dtype: float64

## Checking number of Unique Values in each column
print(Final_database.apply(lambda col: col.nunique()))

# Result in console:
# order_id                                       98666
# order_item_id                                     21
# product_id                                     32951
# seller_id                                       3095
# shipping_limit_date                            93318
# price                                           5968
# freight_value                                   6999
# product_category_name                             73
# product_name_lenght                               66
# product_description_lenght                      2960
# product_photos_qty                                19
# product_weight_g                                2204
# product_length_cm                                 99
# product_height_cm                                102
# product_width_cm                                  95
# seller_zip_code_prefix                          2246
# seller_city                                      611
# seller_state                                      23
# customer_id                                    98666
# order_status                                       7
# order_purchase_timestamp                       98112
# order_approved_at                              90174
# order_delivered_carrier_date                   81017
# order_delivered_customer_date                  95664
# order_estimated_delivery_date                    450
# payment_sequential                                20
# payment_installments                              51
# payment_value                                  27845
# payment_type_count                                 2
# review_id                                      88417
# review_score                                       5
# review_comment_title                            4193
# review_comment_message                         32793
# review_creation_date                             633
# review_answer_timestamp                        80622
# customer_unique_id                             95420
# customer_zip_code_prefix                       14976
# customer_city                                   4110
# customer_state                                    27
# product_category_name_english                     71
# Category                                          10
# total_purchase_count                              20
# product_payment_value                          29149
# freight_to_price_ratio                         46662
# diff_review_creation_answer_days                 201
# diff_approved_purchased                           20
# diff_customerdelivered_estimated                 198
# diff_customerdelivered_deliveredcarrier          145
# diff_customerdelivered_purchase                  146
# diff_deliveredcarrier_purchase                    63
# diff_approved_purchased_wd                        15
# diff_customerdelivered_deliveredcarrier_wd       114
# diff_deliveredcarrier_purchase_wd                 48
# order_id_product_id                           102425
# dtype: int64

## Checking % of Unique Values in each column
print(round(Final_database.apply(lambda col: col.nunique()/len(col))*100,3))

# Result in console:
# order_id                                      87.586
# order_item_id                                  0.019
# product_id                                    29.251
# seller_id                                      2.747
# shipping_limit_date                           82.839
# price                                          5.298
# freight_value                                  6.213
# product_category_name                          0.065
# product_name_lenght                            0.059
# product_description_lenght                     2.628
# product_photos_qty                             0.017
# product_weight_g                               1.957
# product_length_cm                              0.088
# product_height_cm                              0.091
# product_width_cm                               0.084
# seller_zip_code_prefix                         1.994
# seller_city                                    0.542
# seller_state                                   0.020
# customer_id                                   87.586
# order_status                                   0.006
# order_purchase_timestamp                      87.095
# order_approved_at                             80.048
# order_delivered_carrier_date                  71.919
# order_delivered_customer_date                 84.921
# order_estimated_delivery_date                  0.399
# payment_sequential                             0.018
# payment_installments                           0.045
# payment_value                                 24.718
# payment_type_count                             0.002
# review_id                                     78.488
# review_score                                   0.004
# review_comment_title                           3.722
# review_comment_message                        29.111
# review_creation_date                           0.562
# review_answer_timestamp                       71.569
# customer_unique_id                            84.705
# customer_zip_code_prefix                      13.294
# customer_city                                  3.648
# customer_state                                 0.024
# product_category_name_english                  0.063
# Category                                       0.009
# total_purchase_count                           0.018
# product_payment_value                         25.876
# freight_to_price_ratio                        41.422
# diff_review_creation_answer_days               0.178
# diff_approved_purchased                        0.018
# diff_customerdelivered_estimated               0.176
# diff_customerdelivered_deliveredcarrier        0.129
# diff_customerdelivered_purchase                0.130
# diff_deliveredcarrier_purchase                 0.056
# diff_approved_purchased_wd                     0.013
# diff_customerdelivered_deliveredcarrier_wd     0.101
# diff_deliveredcarrier_purchase_wd              0.043
# order_id_product_id                           90.923
# dtype: float64

# Visualisations

# Correlations

## IV. DATA CLEANING AND PREPARATION

# Checking number of rows per Order_Status in Final_database
pd.DataFrame({'Q': Final_database.groupby('order_status').size(), '%': round(Final_database.groupby('order_status').size()/len(Final_database),2)})

# Result in console:
#                    Q     %
# order_status              
# approved           3  0.00
# canceled         542  0.00
# delivered     110197  0.98
# invoiced         359  0.00
# processing       357  0.00
# shipped         1185  0.01
# unavailable        7  0.00

# Filter to only include rows where 'order_status' is 'delivered'
Final_database = Final_database[Final_database['order_status'] == 'delivered']

# Checking number of rows in Final_database
Final_database.shape

# Result in console:
# Out[138]: (110197, 68)

# Specifying the columns to drop
columns_to_drop = ['order_status', 'review_id', 'order_purchase_timestamp', 'order_approved_at',
                   'order_delivered_carrier_date', 'order_delivered_customer_date',
                   'payment_installments', 'order_item_id', 'shipping_limit_date',
                   'shipping_limit_month', 'shipping_limit_year', 'product_weight_g',
                   'product_length_cm', 'product_height_cm', 'product_width_cm',
                   'review_comment_title','review_comment_message', 'payment_value',
                   'order_id','product_id', 'seller_id', 'shipping_limit_period', 
                   'seller_zip_code_prefix', 'seller_city', 'customer_id',
                   'review_creation_date', 'review_answer_timestamp',
                   'customer_unique_id', 'customer_zip_code_prefix', 'customer_city',
                   'payment_sequential','order_estimated_delivery_date','product_category_name_english',
                   'product_category_name']

# Dropping the specified columns
Final_database = Final_database.drop(columns=columns_to_drop, errors='ignore')

# Reset index after filtering
Final_database = Final_database.reset_index(drop=True)

## Checking Final_database
Final_database.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 110197 entries, 0 to 110196
# Data columns (total 23 columns):
#  #   Column                                      Non-Null Count   Dtype  
# ---  ------                                      --------------   -----  
#  0   price                                       110197 non-null  float64
#  1   freight_value                               110197 non-null  float64
#  2   product_payment_value                       110197 non-null  float64
#  3   freight_to_price_ratio                      110197 non-null  float64
#  4   product_name_lenght                         108660 non-null  float64
#  5   product_description_lenght                  108660 non-null  float64
#  6   product_photos_qty                          108660 non-null  float64
#  7   seller_state                                110197 non-null  object 
#  8   diff_approved_purchased                     110182 non-null  float64
#  9   diff_customerdelivered_estimated            110189 non-null  float64
#  10  diff_customerdelivered_deliveredcarrier     110188 non-null  float64
#  11  diff_customerdelivered_purchase             110189 non-null  float64
#  12  diff_deliveredcarrier_purchase              110195 non-null  float64
#  13  diff_approved_purchased_wd                  110182 non-null  float64
#  14  diff_customerdelivered_deliveredcarrier_wd  110188 non-null  float64
#  15  diff_deliveredcarrier_purchase_wd           110195 non-null  float64
#  16  payment_type_count                          110194 non-null  float64
#  17  review_score                                99352 non-null   float64
#  18  diff_review_creation_answer_days            99352 non-null   float64
#  19  customer_state                              110197 non-null  object 
#  20  Category                                    108638 non-null  object 
#  21  total_purchase_count                        110197 non-null  int64  
#  22  order_id_product_id                         110197 non-null  object 
# dtypes: float64(18), int64(1), object(4)
# memory usage: 19.3+ MB

## Checking Missing values
Final_database.isnull().sum()

# Result in console:
# price                                             0
# freight_value                                     0
# product_payment_value                             0
# freight_to_price_ratio                            0
# product_name_lenght                            1537
# product_description_lenght                     1537
# product_photos_qty                             1537
# seller_state                                      0
# diff_approved_purchased                          15
# diff_customerdelivered_estimated                  8
# diff_customerdelivered_deliveredcarrier           9
# diff_customerdelivered_purchase                   8
# diff_deliveredcarrier_purchase                    2
# diff_approved_purchased_wd                       15
# diff_customerdelivered_deliveredcarrier_wd        9
# diff_deliveredcarrier_purchase_wd                 2
# payment_type_count                                3
# review_score                                  10845
# diff_review_creation_answer_days              10845
# customer_state                                    0
# Category                                       1559
# total_purchase_count                              0
# order_id_product_id                               0
# dtype: int64

## There aren't missing values

## Checking %Missing values
round(Final_database.isnull().sum()/len(Final_database) * 100,3)

# Result in console:
# price                                         0.00
# price                                         0.00
# freight_value                                 0.00
# product_payment_value                         0.00
# freight_to_price_ratio                        0.00
# product_name_lenght                           1.39
# product_description_lenght                    1.39
# product_photos_qty                            1.39
# seller_state                                  0.00
# diff_approved_purchased                       0.01
# diff_customerdelivered_estimated              0.01
# diff_customerdelivered_deliveredcarrier       0.01
# diff_customerdelivered_purchase               0.01
# diff_deliveredcarrier_purchase                0.00
# diff_approved_purchased_wd                    0.01
# diff_customerdelivered_deliveredcarrier_wd    0.01
# diff_deliveredcarrier_purchase_wd             0.00
# payment_type_count                            0.00
# review_score                                  9.84
# diff_review_creation_answer_days              9.84
# customer_state                                0.00
# Category                                      1.41
# total_purchase_count                          0.00
# order_id_product_id                           0.00
# dtype: float64

# Dropping rows with nulls values
Final_database=Final_database.dropna()

## Checking Final_database
Final_database.info()

# Result in console:
# <class 'pandas.core.frame.DataFrame'>
# Index: 108615 entries, 0 to 110196
# Data columns (total 23 columns):
#  #   Column                                      Non-Null Count   Dtype  
# ---  ------                                      --------------   -----  
#  0   price                                       108615 non-null  float64
#  1   freight_value                               108615 non-null  float64
#  2   product_payment_value                       108615 non-null  float64
#  3   freight_to_price_ratio                      108615 non-null  float64
#  4   product_name_lenght                         108615 non-null  float64
#  5   product_description_lenght                  108615 non-null  float64
#  6   product_photos_qty                          108615 non-null  float64
#  7   seller_state                                108615 non-null  object 
#  8   diff_approved_purchased                     108615 non-null  float64
#  9   diff_customerdelivered_estimated            108615 non-null  float64
#  10  diff_customerdelivered_deliveredcarrier     108615 non-null  float64
#  11  diff_customerdelivered_purchase             108615 non-null  float64
#  12  diff_deliveredcarrier_purchase              108615 non-null  float64
#  13  diff_approved_purchased_wd                  108615 non-null  float64
#  14  diff_customerdelivered_deliveredcarrier_wd  108615 non-null  float64
#  15  diff_deliveredcarrier_purchase_wd           108615 non-null  float64
#  16  payment_type_count                          108612 non-null  float64
#  17  review_score                                108615 non-null  float64
#  18  diff_review_creation_answer_days            97941 non-null   float64
#  19  customer_state                              108615 non-null  object 
#  20  Category                                    108615 non-null  object 
#  21  total_purchase_count                        108615 non-null  int64  
#  22  order_id_product_id                         108615 non-null  object 
# dtypes: float64(18), int64(1), object(4)
# memory usage: 19.9+ MB

# Exporting Final_database
Final_database.to_excel("Final_database.xlsx", index=False)

# List of numerical columns to normalise
columns_to_normalise = [
    "price", "freight_value","product_description_lenght", "product_photos_qty",
    "diff_approved_purchased", "diff_customerdelivered_estimated",
    "diff_customerdelivered_deliveredcarrier", "diff_customerdelivered_purchase",
    "diff_deliveredcarrier_purchase", "diff_approved_purchased_wd",
    "diff_customerdelivered_deliveredcarrier_wd", "diff_deliveredcarrier_purchase_wd",
    "price", "payment_type_count", "product_payment_value",
    "diff_review_creation_answer_days",
    "product_name_lenght","total_purchase_count"
    ]

# Initialise the scaler
scaler = StandardScaler()

# Normalise only the selected numerical columns
Final_database_norm=Final_database
Final_database_norm[columns_to_normalise] = scaler.fit_transform(Final_database[columns_to_normalise])


Final_database_norm = pd.get_dummies(Final_database_norm, columns=["Category"], prefix=["type_"], dtype='int')
Final_database_norm = pd.get_dummies(Final_database_norm, columns=["customer_state"], prefix=["cs_type_"], dtype='int')
Final_database_norm = pd.get_dummies(Final_database_norm, columns=["seller_state"], prefix=["ss_type_"], dtype='int')

# Normalise review_score
Final_database_norm['review_score'] = Final_database_norm['review_score'].apply(lambda x: 1 if x > 3 else 0)

# Setting index
Final_database_norm = Final_database_norm.set_index('order_id_product_id')

# Exporting Final_database_norm
#Final_database_norm.to_excel("Final_database_norm.xlsx", index=False)


## V. MODELLING

from sklearn.model_selection import train_test_split


# RESET THE DATA TO REMOVE THE EFFECT OF UNDERSAMPLING

# Seperating x and y
xvalues=Final_database_norm.drop(["review_score"],axis=1)
yvalue=Final_database_norm["review_score"]

# Splitting data into training and test
x_train, x_test, y_train, y_test = train_test_split(xvalues, yvalue, test_size = 0.3, random_state=4567, stratify=yvalue)
     

from imblearn.over_sampling import SMOTE

# Creating an SMOTE instance that will return 2x as many majority as minority class
# I.e. sampling_strategy=0.5 means minority class will be 50% of the majority class
smote = SMOTE(random_state=42, sampling_strategy=0.5)

# Note we only undersample the training data not the test data
x_train, y_train = smote.fit_resample(x_train, y_train)

# Getting the value countes by temporarily converting to a dataframe
pd.Series(y_train).value_counts()

import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.utils import resample
from xgboost import XGBClassifier as XGB
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import ConfusionMatrixDisplay as CM

# To ignore warnings below
import warnings
warnings.filterwarnings("ignore")

# Printing the shapes to check everything is OK
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# All splits check out and look perfect.


# HYPERPARAMETER MODEL

RF_algo = RF()
GBDT_algo = GBDT()
XGB_algo = XGB()

# Trying Bayesian optimisation

##RF
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import cross_val_score
def objective(params):
    # must be integer
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    clf = RF(**params, random_state=1234)
    scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro')
    return -np.mean(scores)
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 600, 1),  
    'max_depth': hp.quniform('max_depth', 2, 7, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1), 
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None])}
trials=Trials()
best_params = fmin(
    fn=objective,                # optimize objective
    space=space,                 # search the space
    algo=tpe.suggest,            #use tpe algo
    max_evals=15,                # maximum evaluation time
    trials=trials,               # record trials
    rstate=np.random.default_rng(1234)
)
print("Best Parameters:", best_params)

#Format the result of Hyperopt as the RandomForestClassifier parameter
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['max_features'] = ['sqrt', 'log2', None][best_params['max_features']]
RF_algo = RF(**best_params, random_state=1234)
RF_model=RF_algo.fit(x_train, y_train)

##GBDT
def objective(params2):
    params2['n_estimators'] = int(params2['n_estimators'])
    params2['max_depth'] = int(params2['max_depth'])
    clf = GBDT(**params2, random_state=1234)
    scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro') 
    return -np.mean(scores)
space2 = {
    'n_estimators': hp.quniform('n_estimators', 50, 600, 1), 
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.21),
    'criterion': hp.choice('criterion', ['friedman_mse', 'squared_error']),
    'max_depth': hp.quniform('max_depth', 3, 10, 1)}
trials2 = Trials()
best_params2 = fmin(
    fn=objective,                
    space=space2,                 
    algo=tpe.suggest,            
    max_evals=10,                
    trials=trials2,               
    rstate=np.random.default_rng(1234))
print("Best Parameters:", best_params2)
best_params2['n_estimators'] = int(best_params2['n_estimators'])
best_params2['max_depth'] = int(best_params2['max_depth'])
best_params2['criterion'] = ['friedman_mse', 'squared_error'][best_params2['criterion']]
GBDT_algo = GBDT(**best_params2, random_state=1234) 
GBDT_model=GBDT_algo.fit(x_train, y_train)


##XGB
def objective(params3):
    params3['n_estimators'] = int(params3['n_estimators'])
    params3['max_depth'] = int(params3['max_depth'])
    clf = XGB(**params3, use_label_encoder=False, random_state=1234)
    scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro')
    return -np.mean(scores)
space3 = {
    'n_estimators': hp.quniform('n_estimators', 50, 600, 1),
    'eta': hp.uniform('eta', 0.01, 5.0),                      
    'objective': hp.choice('objective', ['binary:logistic', 'binary:hinge']),  
    'max_depth': hp.quniform('max_depth', 3, 10, 1)}
trials3 = Trials()
best_params3 = fmin(
    fn=objective,               
    space=space3,               
    algo=tpe.suggest,          
    max_evals=15,             
    trials=trials3,           
    rstate=np.random.default_rng(1234))
print("Best Parameters:", best_params3)
best_params3['n_estimators'] = int(best_params3['n_estimators'])
best_params3['max_depth'] = int(best_params3['max_depth'])
best_params3['objective'] = ['binary:logistic', 'binary:hinge'][best_params3['objective']]
XGB_algo = XGB(**best_params, use_label_encoder=False, random_state=1234)
XGB_model=XGB_algo.fit(x_train, y_train)

models=[RF_model,GBDT_model,XGB_model]
names=['RF',"GBDT","XGB"]
for i in range(3):
  print(f"Model: {names[i]}")
  predict=models[i].predict(x_train)
  precision, recall, f1_score, _ = precision_recall_fscore_support(y_train, predict, average='macro')
  accuracy= accuracy_score(y_train, predict)
  print(f"Macro Precision: {precision}")
  print(f"Macro Recall: {recall}")
  print(f"Macro F1-score: {f1_score}")
  print(f"Accuracy: {accuracy}")
  print("\n")
  
# Final score 
for i in range(3):
  print(f"Model: {names[i]}")
  predict=models[i].predict(x_test)
  precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predict, average='macro')
  accuracy= accuracy_score(y_test, predict)
  print(f"Macro Precision: {precision}")
  print(f"Macro Recall: {recall}")
  print(f"Macro F1-score: {f1_score}")
  print(f"Accuracy: {accuracy}")
  print("\n")
  
# WITH HYPERPARAMETERS

# Creating a hyperparameter search function for re-usability
def random_search(algo, hyperparameters, x_train, y_train):
  # do the search using 5 folds/chunks
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2024,
                          scoring='precision_macro', n_iter=10, refit=True)

  # Passing the data to fit/train
  clf.fit(x_train, y_train)
  return clf.best_params_

# Below are the 3 models

# Random Forest (RF)
RF_tuned_parameters = {
    'n_estimators': randint(200, 400),  # More trees to capture patterns
    'max_depth': randint(8, 20),        # Deeper trees
    'max_features': [None, 0.8, 'sqrt']}

# Gradient Boosted Decision Trees (GBDT)
GBDT_tuned_parameters = {
    'n_estimators': randint(100, 350),
    'learning_rate': uniform(loc=0.01, scale=0.8),
    'criterion': ['friedman_mse', 'squared_error'],
    'max_depth': randint(3, 8),
    'min_samples_split': randint(10, 20),
    'subsample': uniform(loc=0.6, scale=0.4)}

# Extreme Gradient Boosting (XGBoost)
XGB_tuned_parameters = {
    'n_estimators': randint(100, 350),
    'eta': uniform(loc=0.01, scale=0.8),
    'objective': ['binary:logistic', 'binary:hinge'],
    'max_depth': randint(3, 8),
    'alpha': uniform(loc=0.01, scale=0.3),
    'lambda': uniform(loc=0.5, scale=1),
    'subsample': uniform(loc=0.3, scale=0.4),
    'colsample_bytree': uniform(loc=0.3, scale=0.4)}

RF_best_params = random_search(RF_algo, RF_tuned_parameters, x_train, y_train)
GBDT_best_params = random_search(GBDT_algo, GBDT_tuned_parameters, x_train, y_train)
XGB_best_params = random_search(XGB_algo, XGB_tuned_parameters, x_train, y_train)

#Training the model
RF_algo = RF(**RF_best_params) 
RF_model = RF_algo.fit(x_train, y_train)

GBDT_algo = GBDT(**GBDT_best_params)
GBDT_model = GBDT_algo.fit(x_train, y_train)

XGB_algo = XGB(**XGB_best_params)
XGB_model = XGB_algo.fit(x_train, y_train)

# Scoring the models
models = [RF_model, GBDT_model, XGB_model] 
names = ['Random Forest', 'GBDT', 'XGBDT']

for i in range(3):
  print(f"Model: {names[i]}")

  # Predicting based on training data
  predict = models[i].predict(x_test)

  # Calculating precision, recall, and F1-score
  precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predict, average='macro')
  accuracy = accuracy_score(y_test, predict) 
  print(f"Macro Precision: {precision}")
  print(f"Macro Recall: {recall}")
  print(f"Macro F1-score: {f1_score}")
  print(f"Accuracy: {accuracy}")
  print("\n")
  
  # Checking accuracy on test data
accuracy = accuracy_score(y_test, predict)
print(f"Accuracy: {accuracy}")

print(classification_report(y_test, predict))

plt.show()

# Random Forest
print("Random Forest Confusion Matrix")
predict = RF_model.predict(x_test)
print(CM.from_predictions(y_test, predict))


# GBDT
print("GBDT Confusion Matrix")
predict = GBDT_model.predict(x_test)
print(CM.from_predictions(y_test, predict))


