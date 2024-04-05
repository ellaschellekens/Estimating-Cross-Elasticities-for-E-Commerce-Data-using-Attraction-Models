from google.cloud import bigquery
import pandas as pd
import numpy as np
import math
import itertools
import janitor
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
df_use['brand_id'][8]=brand_id_reus # brand has a spelling error
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

#group detergents with same brand/substance etc
#df_wasmiddel_combined=wasmiddel_use.groupby(['brand_id','suitable_for_type','substance','kind_of_detergent'])
#list_wasmiddel_combined=list(df_wasmiddel_combined)


def sum_with_none(values):
    if all(value is None for value in values):
        return None
    return np.sum(values)

#need  combined sales and revenue data of the grouped products
list_test=[]
for i in range(len(list_wasmiddel_combined)):
    temp=list_wasmiddel_combined[i][1].groupby(['date_placed']).agg(
    rev_per_wash_sum=('rev_per_wash', sum_with_none),
    av_price=('price_per_wash',np.nanmean),
    daily_sales_sum=('daily_sales', sum_with_none),
    sales_promo_sum=('sales_promo', sum_with_none),
    mean_action_subtype=('action_subtype',np.average), #action subtype should be weighted average
    product_info=('brand_id', lambda x:x+' '+list_wasmiddel_combined[i][0][1]+' '+list_wasmiddel_combined[i][0][2]+' '+list_wasmiddel_combined[i][0][3]+' '+list_wasmiddel_combined[i][1]['partyName'].iloc[1] ))
    list_test.append(temp)


#also need the weighted average price, however if sales are zero, price should be the normal average
# also add a new product id
for i in range(len(list_test)):
    list_test[i]['price']=None
    list_test[i]['promotion']=None
    for j in range(len(list_test[i])):
        if list_test[i]['rev_per_wash_sum'][j] ==0.0:
            list_test[i]['price'][j]=list_test[i]['av_price'][j]
        else:
            list_test[i]['price'][j]=list_test[i]['rev_per_wash_sum'][j]/list_test[i]['daily_sales_sum'][j]
        if list_test[i]['daily_sales_sum'][j] == 0.0 or np.isnan(list_test[i]['daily_sales_sum'][j]):
            list_test[i]['promotion'][j] = list_test[i]['mean_action_subtype'][j]
        else:
            list_test[i]['promotion'][j]=list_test[i]['sales_promo_sum'][j]/list_test[i]['daily_sales_sum'][j]
    list_test[i]['product_id']=i
    list_test[i]['type']=1
    #need the average price and sales if there is no promo for when there is no offer
    list_test[i]['no_promo_sale'],list_test[i]['no_promo_price']=list_test[i].groupby('promotion').agg(no_promo_sale=('daily_sales_sum',np.nanmean),no_promo_price=('price',np.nanmean)).iloc[0]
    list_test[i]=list_test[i].reset_index()

new_wasmiddel=pd.concat([df for df in list_test],ignore_index=True)

nan_count=new_wasmiddel['price'].isna().sum()
percent_nan=nan_count/len(new_wasmiddel)



#Interpolate mean values for sales and price
new_wasmiddel_use=new_wasmiddel.drop(columns=['rev_per_wash_sum','av_price','sales_promo_sum','mean_action_subtype'],axis=1)
#new_wasmiddel_use=new_wasmiddel_use.drop(columns=['Unnamed: 0','level_0'])
new_wasmiddel_use=new_wasmiddel_use.rename(columns={"daily_sales_sum":'sales'})
test= len(new_wasmiddel_use[(new_wasmiddel_use["price"].isna()) & new_wasmiddel_use['promotion']>0.0 ]) #0 so not possible to be in promotion but no offer
new_wasmiddel_use['no_offer']= new_wasmiddel_use['price'].isna().astype(int)



# save data





