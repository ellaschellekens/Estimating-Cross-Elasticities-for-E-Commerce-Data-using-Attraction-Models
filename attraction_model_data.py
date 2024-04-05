import pandas as pd
import numpy as np
import csv
# from numpy.linalg import inv, matrix_power
from scipy.linalg import block_diag, inv
from scipy import sparse
import statsmodels.api as sm
from itertools import chain
from numpy.linalg import cholesky
from scipy.stats import norm
from google.cloud import bigquery
import re
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

bq_client = bigquery.Client(project="")

# initialise the dataset
brand_data = False
product_data = False
combined_data = False
simulated_data = True



def read_file_to_string(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


if brand_data:
    df = pd.read_csv('/Data/brand.csv', sep=';')
    df = df.drop(df.columns[0], axis=1)  # one column is added because of csv file

elif combined_data:
    df = pd.read_csv('/Data/combined.csv', sep=';')
    df = df.drop(df.columns[0], axis=1)  # one column is added because of csv file

elif product_data:
    query = ' SELECT * from `.eschellekens.sales_full`'
    df = bq_client.query(query).to_dataframe()
    df.rename({'action_subtype': 'promotion'}, axis=1, inplace=True)
    df = df.assign(product_info=df.partyName.astype(str) + ', ' +
                                df.substance.astype(str) + ', ' +
                                df.kind_of_detergent.astype(str) + ', ' +
                                df.suitable_for_type.astype(str))
else:
    df = pd.read_csv('/Simulation/simulated_sales.csv', sep=';')
    df = df.drop(df.columns[0], axis=1)  # one column is added because of csv file
    df.rename({'promo_dummy': 'promotion'}, axis=1,
              inplace=True)  #make sure variable names are equal across datasets
    df.rename({'global_id': 'product_id'}, axis=1, inplace=True)
    df.rename({'offer_dummy': 'no_offer'}, axis=1, inplace=True)
    df['product_info'] = df['product_id']

#Calculate the shares of each product
sum_date = df.groupby('date_placed', dropna=False)['sales'].sum('sum')
df_sum = sum_date.to_frame().reset_index()
df_sum.rename({'sales': 'sum_sales'}, axis=1, inplace=True)
df_total = pd.merge(df, df_sum, how='outer')
df_total = df_total.drop_duplicates()

df_total['share'] = df_total['sales'] / df_total['sum_sales']
if brand_data:
    df_total = df_total[df_total['product_id'] != 18]  # is paint
    delete_items=[18]
if simulated_data:
    df_total = df_total[df_total['product_id'] != 20]
    delete_items=[20]
    # remove one product
if combined_data:
    df_total = df_total[df_total['product_id'] != 67]  # is paint
    delete_items=[67]

#split price into price and promotion price variable
df_total['promo_price'] = 0.0
df_total.loc[df_total['promotion'] >= 0.5, 'promo_price'] = df_total['price']
df_total.loc[df_total['promotion'] >= 0.5, 'price'] = 0.0
df_total.loc[df_total['no_offer'] == 1, 'price'] = 0

#change the data format to wide
df_wide = df_total.pivot(index='date_placed', columns='product_id',
                         values=['share', 'price', 'promo_price', 'sales', 'promotion', 'no_offer', 'product_info'])
df_wide = df_wide.reset_index()

#add monthly dummies for empirical data
if not simulated_data:
    df_wide['date_placed'] = pd.to_datetime(df_wide['date_placed'])
    # Create 'month' column
    df_wide['month'] = df_wide['date_placed'].dt.month
    # Create dummy variables for each month
    dummy_variables = pd.get_dummies(df_wide['month'], prefix='month')

    df_wide['feb'] = dummy_variables['month_2']
    df_wide['mar'] = dummy_variables['month_3']
    df_wide['apr'] = dummy_variables['month_4']
    df_wide['may'] = dummy_variables['month_5']
    df_wide['jun'] = dummy_variables['month_6']
    df_wide['jul'] = dummy_variables['month_7']
    df_wide['aug'] = dummy_variables['month_8']
    df_wide['sep'] = dummy_variables['month_9']
    df_wide['oct'] = dummy_variables['month_10']
    df_wide['nov'] = dummy_variables['month_11']
    df_wide['dec'] = dummy_variables['month_12']



def ols(y, X, m, H):
    # Compute OLS betas
    var_cov = inv(X.T @ H.T @ H @ X)
    beta = var_cov @ (X.T @ H.T @ H @ y)
    # Estimate covariance of errors
    e = y - np.dot(X, beta)
    ee = e.to_numpy()
    e2 = np.reshape(ee, (m, -1))
    e3 = e2.transpose()
    S = np.dot(e3.T, e3) / e3.shape[0]

    return {"beta": beta, "S": S, 'residuals': e3, 'var_cov': var_cov}


def fgls(y, X, m, H, eps=1e-3):
    start = ols(y, X, m, H)
    S = start["S"]
    t = X.shape[0] // m
    prev = None
    count = 0
    while True:
        omega = np.kron(inv(S), np.eye(t))
        step = X.T @ H.T @ omega @ H @ X
        var_cov = inv(step)
        beta = var_cov @ (X.T @ H.T @ omega @ H @ y)
        e = y - np.dot(X, beta)
        ee = e.to_numpy()
        e2 = np.reshape(ee, (m, -1))
        e3 = e2.transpose()
        S = np.dot(e3.T, e3) / e3.shape[0]

        sse = np.sum(e3 ** 2)
        count = count + 1
       
        if prev is not None:
            dif = sse - prev
            print("count:", count, "dif:", dif)
        if (prev is not None and abs(sse - prev) < eps) or count > 305:
            break

        prev = sse

    return {"beta": beta, "S": S, "residuals": e3, "var_cov": var_cov}

#split into train and test data
train = df_wide[:730]
test = df_wide[730:850]
test = test.reset_index()

# need to identify products without promotions
promotions = train['promo_price']
zero_promo = promotions.columns[(promotions.eq(0.0)).all()]
price = train['price']
zero_price = price.columns[(price.eq(0.0)).all()]
promo_test = test['promo_price']
zero_promo_test = promo_test.columns[(promo_test.eq(0.0)).all()]
price_test = test['price']
zero_price_test = price_test.columns[(price_test.eq(0.0)).all()]
zeros = list(set(chain(zero_promo, zero_price)))
zeros.sort()
train_full = train
test_full = test

for var in train.columns:
    if var[1] in zeros:
        train = train.drop(columns=[var], axis=1)
        test = test.drop(columns=[var], axis=1)

#need to define the transformation matrix H
price = train['price']
price = pd.DataFrame(price)
f = len(zeros)  # number of products without promotions
M = train['share']
m = len(train_full['share'].columns)  # number of products 
n = len(M)  # time
ident = np.identity(m - f)
values = np.full((m - f, m - f), 1 / (m+1))
ident2 = np.identity(n)

h = ident - values
H = np.kron(h, ident2)
H = H.astype(np.float32)

#The monthly dummies
if not simulated_data:
    dummies_train = train.iloc[:, -11:]
    dummies_test = test.iloc[:, -11:]

#need to define the X  containing all relevant variables 
X_var = pd.DataFrame()
X_var_test = pd.DataFrame()
list_of_x = []
list_of_x_test = []
products = set(df_total['product_id'].unique().tolist())
products = products - set(zeros)  # all products 
for prod in products:
    ones = np.ones(len(train))
    ones_df = pd.DataFrame(ones, columns=['Constant' + str(prod)]) #create the intercept
    X_var = pd.concat([X_var, ones_df], axis=1)
    if not simulated_data:
        X_var = pd.concat([X_var, dummies_train], axis=1) #add monthly dummies
    X_var = pd.concat([X_var, train['price'][prod]], axis=1) #add price variables
    X_var = pd.concat([X_var, train['promo_price'][prod]], axis=1) #add promotion price variables
    if train['no_offer'][prod].sum() != 0:
        X_var = pd.concat([X_var, train['no_offer'][prod]], axis=1)  # add no offer dummy
        #give names to all columns
        if not simulated_data:
            X_var.columns = ['constant' + str(prod), 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
                             'nov',
                             'dec', 'price' + str(prod),
                             'promo_price' + str(prod), 'no_offer' + str(prod)]
        else:
            X_var.columns = ['constant' + str(prod), 'price' + str(prod),
                             'promo_price' + str(prod), 'no_offer' + str(prod)]
    else:
        if not simulated_data:
            X_var.columns = ['constant' + str(prod), 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
                             'nov',
                             'dec', 'price' + str(prod),
                             'promo_price' + str(prod)]
        else:
            X_var.columns = ['constant' + str(prod), 'price' + str(prod),
                             'promo_price' + str(prod)]

    list_of_x.append(X_var) 
    X_var = pd.DataFrame()
    #also need to define X for the test data
    if not simulated_data:
        ones_test = np.ones(len(test))
        ones_test_df = pd.DataFrame(ones_test, columns=['Constant' + str(prod)])
        X_var_test = pd.concat([X_var_test, ones_test_df], axis=1)
        X_var_test = pd.concat([X_var_test, dummies_test], axis=1)
        X_var_test = pd.concat([X_var_test, test['price'][prod]], axis=1)
        X_var_test = pd.concat([X_var_test, test['promo_price'][prod]], axis=1)
        if train['no_offer'][prod].sum() != 0: #moet train zijn niet test, want als train 0 is dan geen beta
            X_var_test = pd.concat([X_var_test, test['no_offer'][prod]], axis=1)
            X_var_test.columns = ['constant' + str(prod), 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
                                  'nov', 'dec', 'price' + str(prod),
                                  'promo_price' + str(prod), 'no_offer' + str(prod)]
        else:
            X_var_test.columns = ['constant' + str(prod), 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
                                  'nov', 'dec', 'price' + str(prod),
                                  'promo_price' + str(prod)]
        list_of_x_test.append(X_var_test)
        X_var_test = pd.DataFrame()

X_block = block_diag(*list_of_x) #save X as the block matrix
X = X_block.astype(np.float32)

epsilon = 1e-100
M = M.replace(0.0, epsilon)
log_Mt = np.log(M.T)
y = log_Mt.stack() 

#calculate the first step attraction model
model_ols = ols(y, X, m - f, H)
model_fgls = fgls(y, X, m - f, H)
beta = model_fgls["beta"]
resid = model_fgls["residuals"]
S = model_fgls['S']
var_cov_old = model_fgls['var_cov']
se_old = np.sqrt(np.diagonal(var_cov_old))

test_statistic_old = beta / se_old
critical_value = 1.960
significant_old = np.abs(test_statistic_old) > critical_value

# part 2, determine which variables should be added to the model
alpha = 0.05
variables = []
variables_test = []
j = 0
for prod in products:
    list_copy = list_of_x.copy()
    del list_copy[j]  
    #make a new variable that contains the price and promotion price of all other products
    X_new1 = pd.concat([*list_copy], axis=1)
    X_new = X_new1.astype(float)
    X_new = X_new.drop(X_new.filter(regex='constant').columns, axis=1)  # delete the constants
    X_new = X_new.drop(X_new.filter(regex='no_offer').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='feb').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='mar').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='apr').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='may').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='jun').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='jul').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='aug').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='sep').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='oct').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='nov').columns, axis=1)
    X_new = X_new.drop(X_new.filter(regex='dec').columns, axis=1)

    list_copy_test = list_of_x_test.copy()
    del list_copy_test[j] 
    X_new1_test = pd.concat([*list_copy_test], axis=1)
    X_new_test = X_new1_test.astype(float)
    X_new_test = X_new_test.drop(X_new.filter(regex='constant').columns, axis=1)  # delete the constants
    X_new_test = X_new_test.drop(X_new.filter(regex='no_offer').columns, axis=1)
    X_prod_test = list_of_x_test[j]

    residual = resid[:, j]
    #regress the price and promotion prices on the residual
    model = sm.OLS(residual, X_new)
    result = model.fit()
    
    #determine which variables are significant
    p_values = result.pvalues.to_numpy()
    p_test = np.less(p_values, alpha)
    p_values = np.vstack((p_values, p_test))
    p_values = pd.DataFrame(p_values)
    names = X_new.columns
    p_values.columns = names
    p_values = p_values.add_prefix(str(prod))
    X_new = X_new.add_prefix(str(prod))
    X_new_test = X_new_test.add_prefix(str(prod))
    X_prod = list_of_x[
        j]  r
    X_prod = X_prod.add_prefix(str(prod))
    
    # Add the products that are significant, so the second row in pvalues is one
    for i in range(len(p_values.columns)):
        if p_values.iat[1, i] == 1.0:
            temp = X_new[p_values.columns[i]].to_numpy()
            temp_df = pd.DataFrame(temp, columns=[p_values.columns[i]])
            X_prod = pd.concat([X_prod, temp_df], axis=1)

            temp_test = X_new_test[p_values.columns[i]].to_numpy()
            temp_test_df = pd.DataFrame(temp_test, columns=[p_values.columns[i]])
            X_prod_test = pd.concat([X_prod_test, temp_test_df], axis=1)
            # X_prod = np.append(X_prod, temp_df, axis=1) #x_prod now contains variables that are correlated with the residual

    j = j + 1
    variables.append(X_prod)  # list of all new variables for train
    X_prod = pd.DataFrame()
    variables_test.append(X_prod_test)  # list of all new variables for test
    X_prod_test = pd.DataFrame()

# part 3, full model with all variables that are found significant in step 2
X_block_final = block_diag(*variables)
X_final = X_block_final.astype(np.float32)

model_ols_new = ols(y, X_final, m - f, H)
model_fgls_new = fgls(y, X_final, m - f, H)
beta_new = model_fgls_new["beta"]
resid_new = model_fgls_new["residuals"]
var_cov = model_fgls_new['var_cov']
S = model_fgls_new['S']




# determine if variables are significant
se = np.sqrt(np.diagonal(var_cov))
test_statistic = beta_new / se
critical_value = 1.960
significant = np.abs(test_statistic) > critical_value
sum_sig = np.sum(significant)
per_sig = sum_sig / (len(significant) - 24 * 11)
# combine the variable names with beta values and significance
column_names = []
for df in variables:
    column_names.extend(df.columns)
beta_significant = np.vstack((beta_new, significant))
beta_significant = np.vstack((beta_significant, se))
beta_df = pd.DataFrame(beta_significant)
beta_df.columns = column_names

column_groups = {}
for column in beta_df.columns:
    prefix = re.match(r'^(\d+)', column).group(1)
    if prefix not in column_groups:
        column_groups[prefix] = []
    column_groups[prefix].append(column)

#  Create smaller DataFrames and store them in a dictionary
smaller_dataframes = {}
for prefix, columns in column_groups.items():
    smaller_dataframes[prefix] = beta_df[columns]

# beta_df.to_csv('betas_sig_se_brands_attraction_attraction_model.csv', sep=';') save beta values

# part 4, calculate the elasticities
price_columns = beta_df.filter(regex='.*price.*')
del_insignificant = price_columns.columns[price_columns.iloc[1] == 0]  # check which variables are insignificant
price_columns = price_columns.drop(columns=del_insignificant)  # delete insignificant variables
price_columns = price_columns.iloc[[0]]
rows = cols = max(df_total['product_id']) + 1  # number of products

#define the correct size for the matrix with the estimates
B_price = np.zeros((rows, cols), dtype=float)
B_promo = np.zeros((rows, cols), dtype=float)

#add the estimates to the estimate matrix
for row in range(rows):
    for col in range(cols):
        column_name = f"{row}price{col}"
        if column_name in price_columns.columns:
            B_price[row, col] = price_columns[column_name].iloc[0]

for row in range(rows):
    for col in range(cols):
        column_name = f"{row}promo_price{col}"
        if column_name in price_columns.columns:
            B_promo[row, col] = price_columns[column_name].iloc[0]

B_price = pd.DataFrame(B_price)
B_promo = pd.DataFrame(B_promo)
#delete columns of products that are not evaluated
B_price = B_price.drop(zeros+delete_items)
B_promo = B_promo.drop(zeros+delete_items)
B_price = B_price.drop(columns=zeros+delete_items)
B_promo = B_promo.drop(columns=zeros+delete_items)

share = train['share']
avg_share = np.mean(share, axis=0)
price = train['price']  
avg_price = np.mean(price, axis=0)
promo_price = train['promo_price']
avg_promo_price = np.mean(promo_price, axis=0)

I = np.identity(len(products))
J = np.ones(len(products))
Ds = np.diag(avg_share)
Dx_price = np.diag(avg_price)
Dx_promo = np.diag(avg_promo_price)

B_price = B_price.values
B_promo = B_promo.values

#calculate the price elasticity
E_price = (I - J @ Ds) @ B_price @ Dx_price
mask = B_price == 0
E_price[mask] = 0

#calculate the promo price elasticities
E_promo = (I - J @ Ds) @ B_promo @ Dx_promo
mask_pro = B_promo == 0
E_promo[mask_pro] = 0

E_price = pd.DataFrame(E_price)
E_promo = pd.DataFrame(E_promo)

#gave names to columns and rows
E_price.columns=products
E_promo.columns=products

E_price.index=products
E_promo.index=products

E_price.to_csv('E_price_att_att_combined_data.csv', sep=';')
E_promo.to_csv('E_promo_att_att_combined_data.csv', sep=';')



# part 5, forecasting
X_test = block_diag(*variables_test)
X_test = X_test.astype(np.float32)


M_test=test['share']

yhat = (np.dot(X_test, beta_new)).astype(float) #estimate attraction
yhat1 = yhat.reshape(m - f, -1).T
np.random.seed(12345)


P = cholesky(S)
M_sim = np.zeros((yhat1.shape[0], m - f))
n = 10000

for i in range(n):
    z = np.dot(P, np.random.normal(size=((m - f) * yhat1.shape[0],)).reshape(-1, m - f).T)  # draw from normal distrubution and multiply by P
    A_it = np.exp(yhat1 + z.T)  # moet die Z keer 0.5?
    denom = np.sum(A_it, axis=1)
    denom_shape= denom[:, np.newaxis]
    s= A_it/denom_shape
    M_sim += s

M_sim /= n

M_testnp=M_test.to_numpy()

RMSE_sim = np.sqrt(np.mean((M_testnp - M_sim) ** 2, axis=0).astype(float))



#make a plot of the shares
# Set up the plot grid (2x2)
num_rows = 8  # You can adjust the number of rows as per your preference
num_cols = 10

# Set up the plot grid
import matplotlib.pyplot as plt

# Set up the plot grid
plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Loop through each column and plot the data
for b in range(80):
    plt.subplot(num_rows, num_cols, b + 1)
    share = np.column_stack((M_testnp[:, b],  M_sim[:, b]))
    plt.plot(share, marker='o')
    plt.title(f"Column {b}")
    if b == 0:
        plt.legend(["true", "sim"])

plt.show()


