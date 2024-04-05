

This repository contains the code for my thesis:
Abstract: This research investigates the effect of discounts, particularly price promotions, on consumer purchasing behaviour within the domain of laundry detergents for Bol.com. Employing three models: attraction-attraction, lasso-attraction, and correlation-attraction model, we dissect the effects of promotions on not only the promoted items but also similar products within the same product category. We explore whether explicitly labelling price reductions as promotions have a greater impact compared to simply adjusting prices. By combining empirical data and simulated analyses, this study explains consumer responses to discounts in online retail, offering potential insights for strategic decision-making and future research in leveraging promotional strategies within e-commerce.

# Attraction_attraction_model.py
    This file contains the code for the attraction-attraction model.

# Correlation_attraction_model.py
    This file contains the code for the correlation-attraction model.

# Lasso.py
    This file contains the code for the lasso regression.

# Simulate_sales.py
    This file contains the code to make the simulated sales.

# Wasmiddel.sql
    This file contains the query needed to load the sales data of the detergents.

# Attributes.sql
    This file contains the query needed to load the attribute data of the detergents.

# Data.py
    This file uses the sales data made in wasmiddel.sql to define the 85% most sold products.

# Combine_wasmiddel.py and full_data.py
    These files combine the sales information from data.py and the information about the attributes from attributes.sql. The     combine_wasmiddel.py file also combines products of the same brand or products that are considered to be similar for the     combined data set.

# Simulated_correlations.sql and Product_correlations.sql
    These files are used to calculate the correlations between the price and sales of products for the simulated and             empirical data, respectively. They differ because the date structure in these data sets is different.

# Test.sql
    This file is used to calculate the f1, precision and recall score. Make sure the correct data is loaded since this is        used for all three models
