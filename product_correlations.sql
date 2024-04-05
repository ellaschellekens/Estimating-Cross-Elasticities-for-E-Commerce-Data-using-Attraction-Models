CREATE OR REPLACE TABLE `.eschellekens.brand_corr`
AS
WITH base AS(
  SELECT
    product_id,
    date_placed,
    IF(AVG(sales) OVER (PARTITION BY product_id ORDER BY unix_date(DATE(date_placed)) RANGE BETWEEN 14 PRECEDING AND 14 FOLLOWING)!=0,(sales - (AVG(sales) OVER (PARTITION BY product_id ORDER BY unix_date(DATE(date_placed)) RANGE BETWEEN 14 PRECEDING AND 14 FOLLOWING))) / (AVG(sales) OVER (PARTITION BY product_id ORDER BY unix_date(DATE(date_placed)) RANGE BETWEEN 14 PRECEDING AND 14 FOLLOWING)),-1) AS sales,
    (price - (AVG(price) OVER (PARTITION BY product_id ORDER BY unix_date(DATE(date_placed)) RANGE BETWEEN 14 PRECEDING AND 14 FOLLOWING))) / (AVG(price) OVER (PARTITION BY product_id ORDER BY unix_date(DATE(date_placed)) RANGE BETWEEN 14 PRECEDING AND 14 FOLLOWING)) AS price_diff
  FROM `.eschellekens.sales_brand`
  WHERE no_offer = 0
),
join_data AS(
  SELECT
    base.product_id AS product_affected,
    data_influencing.product_id AS affected_by,
    date_placed,
    base.sales AS affected_sales,
    data_influencing.price_diff AS affected_by_price
  FROM base
  JOIN base AS data_influencing
  USING(date_placed)
  WHERE data_influencing.price_diff < 0.95
  AND ABS(base.price_diff) < 0.05
  ),
calc_correlation AS(
SELECT
  product_affected,
  affected_by,
  COUNT(*) as n_obs,
  CORR(affected_sales, affected_by_price) AS correlation,
FROM join_data
GROUP BY product_affected, affected_by
)
SELECT *
FROM calc_correlation
WHERE product_affected != affected_by
AND abs(correlation) > 0.2
and n_obs > 120
order by abs(correlation)
