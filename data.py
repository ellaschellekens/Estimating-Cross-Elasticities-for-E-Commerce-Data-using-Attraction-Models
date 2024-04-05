import matplotlib.pyplot as plt
import pandas as pd
import scipy
import fathon
from fathon import fathonUtils as fu
import numpy as np
import csv
from google.cloud import bigquery
from statsmodels.tsa.stattools import grangercausalitytests

bq_client = bigquery.Client(project="")

#Collect the 85% most sold products

def read_file_to_string(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def save_data(producttype):

    query=read_file_to_string('/Users/eschellekens/Desktop/Projects/ella/Queries/'+str(producttype)+'.sql')
    df = bq_client.query(query).to_dataframe()
    print("test")

    n_wasmiddel= df['global_id'].unique()
    n_sales=df.groupby('global_id')['daily_sales'].aggregate('sum')
    n_sales_sorted=n_sales.sort_values(ascending=False)

    n_sales_sorted_df=n_sales_sorted.to_frame()
    n_sales_sorted_df=n_sales_sorted_df.reset_index()
    n_sales_sorted_df['cumsum_sales'] = n_sales_sorted_df['daily_sales'].cumsum()

    total_sales=n_sales_sorted.sum()
    eightyfive_percent_sales=0.85*total_sales


    most_sold=[]
    i=0
    while n_sales_sorted_df['cumsum_sales'][i]<= eightyfive_percent_sales:
        most_sold.append(n_sales_sorted_df['global_id'][i])
        i+=1

    n_most_sold =len(most_sold)
    df_use=df.loc[df['global_id'].isin(most_sold)]
    #df.to_csv('csv_'+str(producttype)+'_16_april.csv' , sep=';')
    df_use.to_csv('csv_'+str(producttype)+'_30_april_85.csv' , sep=';')
    return n_wasmiddel, n_most_sold, df_use



total_wasmiddel, frequent_wasmiddel, df_wasmiddel=save_data('wasmiddel')
total_wasverzachter, frequent_wasverzachter, df_wasverzachter=save_data('wasverzachter')
total_vlek, frequent_vlek, df_vlek=save_data('vlekverwijderaar')

