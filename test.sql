WITH actuals AS(
  SELECT
    CONCAT(product_affected, "_",affected_by) AS outcome,
    1 AS actual,
  FROM `thesis-413011.predicted_price.actual_affected_offer` /* make sure to load the correct data*/
),

predictions AS(
  SELECT
    CONCAT(product_affected, "_",affected_by) AS outcome,
    1 AS pred
  FROM `thesis-413011.predicted_price.affected_price_offer` /* make sure to load the correct data*/
),

p_r as (SELECT
  SAFE_DIVIDE(
                  SUM(
                    CASE
                      WHEN actual = 1 AND pred = 1 THEN 1
                      ELSE 0
                    END
                  ),
                  SUM(pred)
               ) AS precision,
  SAFE_DIVIDE(
                  SUM(
                    CASE
                      WHEN actual = 1 AND pred = 1 THEN 1
                      ELSE 0
                    END
                  ),
                  SUM(actual)
               ) AS recall,
FROM actuals
FULL OUTER JOIN predictions
USING(outcome))

select precision, recall,
safe_divide(2*(precision*recall),(precision+recall)) as f1
from p_r
