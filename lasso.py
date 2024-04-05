
import pandas as pd
import numpy as np



def read_file_to_string(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

df = pd.read_csv('C/simulated_sales_jan20_offer.csv', sep=';')
df = df.drop(df.columns[0], axis=1)  # one column is added because of csv file
df.rename({'promo_dummy': 'promotion'}, axis=1, inplace=True)# add so the variable names across data sets is the same
df.rename({'global_id': 'product_id'}, axis=1, inplace=True)
df.rename({'offer_dummy': 'no_offer'}, axis=1, inplace=True)
df['product_info']=df['product_id']

#calculate shares
sum_date = df.groupby('date_placed', dropna=False)['sales'].sum('sum')
df_sum = sum_date.to_frame().reset_index()
df_sum.rename({'sales': 'sum_sales'}, axis=1, inplace=True)
df_total = pd.merge(df, df_sum, how='outer')
df_total = df_total.drop_duplicates()

df_total['share'] = df_total['sales'] / df_total['sum_sales']
df_total.loc[df_total['no_offer'] == 1, 'price'] = 0

#change data formad to wide
df_wide = df_total.pivot(index='date_placed', columns='product_id',
                         values=['share', 'price', 'sales', 'promotion', 'no_offer', 'product_info'])
df_wide = df_wide.reset_index()




from sklearn.linear_model import Lasso
price_lasso=df_wide['price']


dummy_promo=df_wide['promotion']
sales=df_wide['sales']
shares=df_wide['share']

# run lasso on sales and prices
model=Lasso(alpha=40, max_iter=10000, random_state=12345,selection='random')
model.fit(price_lasso,sales) 
beta_lasso=model.coef_
intercept_lasso=model.intercept_
sparse=model.sparse_coef_

#save lasso results
df_affected=pd.DataFrame(columns=['product_affected','affected_by','value','pred'],)

for prod in range(len(beta_lasso)):
    for other in range(len(beta_lasso)):
        if prod != other and (beta_lasso[prod][other] !=0.0 or beta_lasso[prod][other] !=-0.0) :
            list_row=[prod,other,beta_lasso[prod][other],1]
            df_affected.loc[len(df_affected)] = list_row

predictions=df_affected[['product_affected','affected_by','pred']]
predictions.to_csv('prediction_lasso_offer_40.csv', sep=',')

true_values=pd.read_csv('/Simulation/dict_cross_elas2.csv', sep=';')
actuals = pd.DataFrame(columns=['product_affected', 'affected_by', 'elasticity'])
for index,row in true_values.iterrows():
   first = int(row[0].split(',')[0].replace("(", '').replace("'", ""))
   second = int(row[0].split(',')[1].replace(")",'').replace("'",""))
   value=row[1]
   actuals.loc[index]=[first,second,value]

#upload lasso predictions to sql and use sql to determine f1 score
from cumulo_context import context
context.bq_client.pandas_dataframe_to_bq(
            df=predictions,
            full_table_id='.eschellekens.hackathon_predictions',
            schema=[],
            write_disposition="WRITE_TRUNCATE")


#load f1 score from sql
def read_file_to_string(path: str) -> str:
    with open(path, "r") as f:
        return f.read()
query=read_file_to_string('/Queries/test.sql')
df_output = bq_client.query(query).to_dataframe()
