WITH WasmiddelId AS (
  SELECT
    global_id,
    brand_id
  FROM ``
  WHERE chunk_name ='Wasmiddel'),


daily_sales_table AS(
  SELECT
    global_id,
    date_placed,
    sum(sales) as sales_per_day,
    sum(sales * price)/sum(sales) as sales_price
  FROM ``
  JOIN WasmiddelId
  USING (global_id)
  GROUP BY global_id, date_placed
  ORDER BY date_placed),


 daily_offer_table AS(
  SELECT
    global_id,
    execution_date AS date_placed,
    price as offer_price
  FROM ``
  JOIN WasmiddelId
  USING (global_id)
  WHERE execution_date BETWEEN '2021-01-01' AND '2023-04-30'
    AND country= 'NL'
  ORDER BY execution_date),

 promo_table AS(
  SELECT distinct
    global_id,
    action_subtype,
    date_placed
 FROM ``
 WHERE action_type='Discount'
 AND country ='NL'
 AND date_placed BETWEEN '2021-01-01' AND '2023-04-30' ),

sales_table AS(
  SELECT
    global_id,
    date_placed,
  COALESCE(sales_per_day,0) as daily_sales,
  COALESCE(sales_price, offer_price) as daily_price
  FROM daily_offer_table
  LEFT JOIN daily_sales_table
  USING (global_id, date_placed)
  Order BY date_placed)

  SELECT
    global_id,
    date_placed,
    daily_sales,
    daily_price,
    action_subtype
    FROM sales_table
    LEFT JOIN promo_table
    USING(global_id, date_placed)
    Order BY date_placed
