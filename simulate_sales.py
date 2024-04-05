import numpy as np
import pandas as pd
import random
import math
import csv


np.random.seed(440)
n_products = 20
length_time_series = 850
price_level = 30
std_price = 2
chance_of_no_offer = 0.09
chance_of_promo = 0.45
list_all_products = []
own_elasticity=[]

for product in range(n_products):
    dict_product = {}
    if product == 0:  # market leader
        sales_level = np.random.normal(400, 10) 
    else:
        sales_level = np.random.normal(40,10) #baseline sales
    avg_price = np.random.normal(price_level, 5) #baseline price
    price_elasticity= np.random.uniform(1, 4)
    own_elasticity.append(price_elasticity)
    #make list to save variables
    product_id_list = []
    sales_product = []
    no_offer_dummies = []
    time_points = []
    price_points = []
    avg_price_points = []
    promo_dummies=[]
    elasticity_repeat=[]
    prev_price=avg_price
    
    for t in range(length_time_series):
        product_id_list.append(str(product))
        time_points.append(t)  
        #define the price at time t
        price_point = np.random.normal(avg_price, std_price) 
        price_point=0.8*price_point+0.2*prev_price
        elasticity_repeat.append(price_elasticity)

        # 2 products are introduced after 1 year
        if (0 < product <= 2 ) and t<=365:
            sales_on_t = 0
            no_offer_dummies.append(1)
            price_points.append(np.NaN)
        elif np.random.random() <= chance_of_no_offer: #check if product is offer at time t
            sales_on_t = 0
            no_offer_dummies.append(1)
            price_points.append(np.NaN)
        else:
            sales_on_t = sales_level
            no_offer_dummies.append(0)
            price_points.append(price_point)
            prev_price=price_point

        if not math.isnan(price_points[-1]) and np.random.random() <= chance_of_promo: #check if there is a promotion
            discount = random.randint(10, 60) / 100
            price_point = (1 - discount) * price_point
            promo_dummies.append(1)
            price_points[-1]=price_point
        else:
            promo_dummies.append(0)


        #calculate the sales
        sales_on_t += sales_on_t * -(price_elasticity) * ((price_point-avg_price)/avg_price) 

        sales_on_t=max(sales_on_t,0)
        sales_product.append(sales_on_t)
        avg_price_points.append(avg_price)

    #save all data
    dict_product["global_id"] = product_id_list
    dict_product["date_placed"] = time_points
    dict_product["offer_dummy"] = no_offer_dummies
    dict_product["sales"] = sales_product
    dict_product["price"] = price_points
    dict_product["avg_price"] = avg_price_points
    dict_product['promo_dummy'] = promo_dummies
    dict_product['elasticity'] = elasticity_repeat
    list_all_products.append(pd.DataFrame(dict_product))

df_all_products = pd.concat(list_all_products)

#define correlated products
correlation = {}
all_products=list(range(0,n_products))
for product in range(n_products):
    random_number = random.randint(0, 3)
    other_products=[x for x in all_products if x != product]
    correlation[str(product)] = random.sample(other_products, random_number) #sample which other products affect the current product, and put in dictionary

cross_elasticity={}
#define cross elasticities for correlated products
for product in df_all_products["global_id"].unique():
    if product in correlation.keys():
        correlated_products = correlation[product]
        sales_change = np.zeros(int(len(df_all_products)/n_products))
        for cor_p in correlated_products:
            df_cor_prod = df_all_products.query(f"global_id =='{cor_p}'")
            t_points_affected = df_cor_prod.query("(price < 0.95 * avg_price) | (price < 0.95 * avg_price)")
           
            if np.random.random() <= 0.5: #cross elasticity can be both positive and negative
                elasticity = np.random.uniform(0,own_elasticity[int(product)]/3) # cross elasticity should be smaller than own elasticity
            else:
                elasticity = -( np.random.uniform(0, own_elasticity[int(product)]))
            cross_elasticity[(product,cor_p)]=elasticity

            #change sales because of cross elasticity
            for time in t_points_affected['date_placed']:
                delta_price=t_points_affected['price'][time]-t_points_affected['avg_price'][time]
                avg_price=t_points_affected['avg_price'][time]
                current_product=df_all_products.query(f"global_id =='{product}'")
                current_sales=current_product['sales'][time]
                sales_change[time]=sales_change[time]+(elasticity*delta_price*current_sales/avg_price)

        df_all_products.loc[df_all_products['global_id']==str(product), 'sales' ]+=sales_change

df_all_products['sales'][df_all_products['sales']<0]=0



#save data
df_own_elasticity=pd.DataFrame(own_elasticity)
with open('dict_cross_elas_jan20_offer.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    for key, value in cross_elasticity.items():
        writer.writerow([key, value])

df_all_products.to_csv('simulated_sales_jan20_offer.csv' , sep=';')
df_own_elasticity.to_csv('own_elasticity_jan20_offer.csv', sep=';')

#save data to big query
