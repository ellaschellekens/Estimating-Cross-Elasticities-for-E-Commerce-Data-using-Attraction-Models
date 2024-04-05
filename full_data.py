from google.cloud import bigquery
import pandas as pd
import numpy as np
import math

bq_client = bigquery.Client(project="")

def read_file_to_string(path: str) -> str:
    with open(path, "r") as f:
        return f.read()
query=read_file_to_string('/Queries/wasmiddel_attributes.sql')
df = bq_client.query(query).to_dataframe() #df contains the info(substance etc) about detergent
df['global_id']=df['global_id'].astype('int')

#daily sales info per detergent
wasmiddel=pd.read_csv('/Data/csv_wasmiddel_30_april_85.csv', sep=';')
wasmiddel_sort=wasmiddel.sort_values(['global_id','date_placed'], ascending= [True,True])
wasmiddel_full=wasmiddel_sort.complete(('global_id'),'date_placed')
wasmiddel_full_sort=wasmiddel_full.sort_values(['global_id','date_placed'], ascending= [True,True])

sales_nan= wasmiddel_sort[wasmiddel_sort['daily_price'] <= 500] #delete unrealistic prices
sales_nan=sales_nan.rename(columns={"daily_sales":'sales'})
sales_nan=sales_nan.rename(columns={"daily_price":'price'})
sales_nan=sales_nan.rename(columns={"global_id":'product_id'})
sales_nan['no_offer'] = 0


from cumulo_context import context
context.bq_client.pandas_dataframe_to_bq(
            df=sales_nan,
            full_table_id='.eschellekens.sales_nan',
            schema=[],
            write_disposition="WRITE_TRUNCATE")


#Some attributes are missing, and some data is incorrect, need to update all of this information
#to do this, we need to make a list of the global_ide of the 85% most sold products
n_wasmiddel = wasmiddel_full_sort.groupby('global_id')['daily_sales'].aggregate('sum')
n_wasmiddel_sorted = n_wasmiddel.sort_values(ascending=False)
n_wasmiddel_sorted_df = n_wasmiddel_sorted.to_frame()
n_wasmiddel_sorted_df = n_wasmiddel_sorted_df.reset_index()

list_wasmiddel=n_wasmiddel_sorted_df['global_id'].values.tolist() 
df_use=df.loc[df['global_id'].isin(list_wasmiddel)]  
df_use=df_use.reset_index(drop=True)
inds = pd.isnull(df_use).any(1).to_numpy().nonzero() #get row number of missing values
#fill in missing values
df_use['number_of_washes'][54]=102
df_use['suitable_for_type'][54]='Gekleurde was'
df_use['kind_of_detergent'][77]='Hoofdwas'
df_use['suitable_for_type'][77]='Gekleurde was'
df_use['kind_of_detergent'][80]='Hoofdwas'
df_use['substance'][80]='Vloeibaar'
df_use['suitable_for_type'][80]='Gekleurde was'
df_use['number_of_washes'][81]=129
df_use['kind_of_detergent'][87]='Hoofdwas'
df_use['substance'][87]='Capsule'
df_use['suitable_for_type'][87]='Gekleurde was'
df_use['kind_of_detergent'][141]='Wasadditieven'
df_use['number_of_washes'][141]=1
df_use['substance'][141]='Poeder'
df_use['suitable_for_type'][141]='Donkere was'
df_use['kind_of_detergent'][162]='Hoofdwas'
df_use['substance'][162]='Capsule'
df_use['kind_of_detergent'][163]='Hoofdwas'
df_use['substance'][163]='Capsule'
df_use['suitable_for_type'][163]='Gekleurde was'
df_use['substance'][174]='Vloeibaar'
df_use['number_of_washes'][140]=156

#some attributes are wrong and need to be changed:
df_use['suitable_for_type'] = df_use['suitable_for_type'].replace('Alle kleuren was', 'Gekleurde was')
df_use['suitable_for_type'][7]='Witte was'
df_use['suitable_for_type'][10]='Witte was'
df_use['suitable_for_type'][25]='Witte was'
df_use['suitable_for_type'][27]='Gekleurde was'
df_use['suitable_for_type'][30]='Gekleurde was'
df_use['suitable_for_type'][33]='Gekleurde was'
df_use['substance'][54]='Capsule'
df_use['suitable_for_type'][58]='Witte was'
df_use['suitable_for_type'][69]='Witte was'
df_use['suitable_for_type'][73]='Witte was'
df_use['suitable_for_type'][75]='Witte was'
df_use['suitable_for_type'][76]='Witte was'
df_use['substance'][77]='Capsule'
df_use['suitable_for_type'][78]='Gekleurde was'
df_use['suitable_for_type'][91]='Gekleurde was'
df_use['suitable_for_type'][99]='Gekleurde was'
df_use['suitable_for_type'][105]='Witte was'
df_use['suitable_for_type'][118]='Gekleurde was'
df_use['suitable_for_type'][119]='Gekleurde was'
df_use['suitable_for_type'][136]='Gekleurde was'
df_use['suitable_for_type'][138]='Gekleurde was'
df_use['substance'][148]='Capsule'
df_use['kind_of_detergent'][152]='Wasadditieven'
df_use['number_of_washes'][153]=261
df_use['suitable_for_type'][154]='Gekleurde was'
df_use['suitable_for_type'][156]='Gekleurde was'
df_use['suitable_for_type'][157]='Gekleurde was'
df_use['suitable_for_type'][160]='Baby' #niet zeker is voor baby/kinder
df_use['suitable_for_type'][170]='Baby' #niet zeker is voor baby/kinder
df_use['suitable_for_type'][5]='Baby'
df_use['suitable_for_type'][169]='Gekleurde was'
df_use['suitable_for_type'][174]='Overige'#strijkwater
df_use['suitable_for_type'][2]='Overige'#doorloopdoekjes
df_use['suitable_for_type'][141]='Overige'#verf
df_use['suitable_for_type'][187]='Multi'#mix wit/kleur/zwart pak
df_use['suitable_for_type'][153]='Multi'
df_use['suitable_for_type'][154]='Multi'
df_use['suitable_for_type'][206]='Multi'
df_use['suitable_for_type'][234]='Multi'
df_use['suitable_for_type'][263]='Multi'
df_use['kind_of_detergent'][263]='Hoofdwas'

df_use['substance'][209]='Vloeibaar'
df_use['substance'][225]='Vloeibaar'
df_use['substance'][226]='Vloeibaar'
df_use['substance'][251]='Vloeibaar'
df_use['substance'][255]='Vloeibaar'
df_use['number_of_washes'][260]=120
df_use['kind_of_detergent'][267]='Wasadditieven'
df_use['substance'][267]='Vloeibaar'


#these contain detergent and softener.
df_use['kind_of_detergent'][197]='Combi'
df_use['kind_of_detergent'][198]='Combi'
df_use['kind_of_detergent'][199]='Combi'
df_use['kind_of_detergent'][200]='Combi'
df_use['kind_of_detergent'][201]='Combi'
df_use['kind_of_detergent'][203]='Combi'
df_use['kind_of_detergent'][204]='Combi'
df_use['kind_of_detergent'][206]='Combi'
df_use['kind_of_detergent'][210]='Combi'
df_use['kind_of_detergent'][231]='Combi'
df_use['kind_of_detergent'][234]='Combi'
df_use['kind_of_detergent'][237]='Combi'
df_use['kind_of_detergent'][238]='Combi'
df_use['kind_of_detergent'][239]='Combi'
df_use['kind_of_detergent'][242]='Combi'
df_use['kind_of_detergent'][247]='Combi'
df_use['kind_of_detergent'][249]='Combi'
df_use['kind_of_detergent'][250]='Combi'
df_use['kind_of_detergent'][263]='Combi'
df_use['kind_of_detergent'][265]='Combi'
df_use['kind_of_detergent'][266]='Combi'

df_use['number_of_washes'][275]=45
df_use['number_of_washes'][276]=45
df_use['suitable_for_type'][287]='Baby'
df_use['suitable_for_type'][288]='Baby'
df_use['suitable_for_type'][290]='Gekleurde was'
df_use['suitable_for_type'][300]='Witte was'
df_use['suitable_for_type'][301]='Multi'
df_use['suitable_for_type'][305]='Multi'
df_use['suitable_for_type'][307]='Gekleurde was'
df_use['substance'][315]='Capsule'
df_use['suitable_for_type'][320]='Gekleurde was'


# one brand has multiple brand ids
brand_id_reus='12345'
df_use['brand_id'][8]=brand_id_reus #spelling error
mask = df_use['partyName'].str.contains('Reus', regex=True)
df_use.loc[mask, 'brand_id'] = '12345'



#combine the sales info per day with the information about substance etc
wasmiddel_use=pd.merge(wasmiddel_full_sort,df_use, how='left', on='global_id')
wasmiddel_use= wasmiddel_use.iloc[: , 1:]
wasmiddel_use['number_of_washes']=wasmiddel_use['number_of_washes'].astype('float')
wasmiddel_use['price_per_wash']=wasmiddel_use['daily_price']/wasmiddel_use['number_of_washes']
wasmiddel_use['rev_per_wash']=wasmiddel_use['price_per_wash']*wasmiddel_use['daily_sales']
wasmiddel_use=wasmiddel_use.replace({np.nan: None})
wasmiddel_use['action_subtype']=wasmiddel_use['action_subtype'].fillna(0).map(lambda x: 1 if x != 0 else 0) #change action subtype to 0 and 1
wasmiddel_use['sales_promo']=wasmiddel_use['daily_sales']*wasmiddel_use['action_subtype']
wasmiddel_use.loc[wasmiddel_use['daily_price'] > 500, 'daily_price'] = None



wasmiddel_use['no_offer']= wasmiddel_use['daily_price'].isna().astype(int)
wasmiddel_use=wasmiddel_use.drop(columns=['price_per_wash','rev_per_wash','sales_promo','number_of_washes'])
wasmiddel_use=wasmiddel_use.rename(columns={"daily_sales":'sales'})
wasmiddel_use=wasmiddel_use.rename(columns={"daily_price":'price'})
wasmiddel_use=wasmiddel_use.rename(columns={"global_id":'product_id'})
wasmiddel_use['sales']=wasmiddel_use['sales'].astype(float)
wasmiddel_use['price']=wasmiddel_use['price'].astype(float)

for i in range(len(wasmiddel_use)):
    if math.isnan(wasmiddel_use['sales'][i]):
        wasmiddel_use['sales'][i] = 0
    if math.isnan(wasmiddel_use['price'][i]):
        wasmiddel_use['price'][i] = 0

from cumulo_context import context
context.bq_client.pandas_dataframe_to_bq(
            df=wasmiddel_use,
            full_table_id='.eschellekens.sales_full',
            schema=[],
            write_disposition="WRITE_TRUNCATE")
