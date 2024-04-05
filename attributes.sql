CREATE OR REPLACE TABLE `.eschellekens.attributen_final`
AS

WITH WasmiddelId AS (
  SELECT
    global_id,
    brand_id
  FROM ` `
  WHERE chunk_name ='Wasmiddel'),

   party_name AS(
      SELECT
      partyId,
      partyName
      FROM``
   ),

 attributen_suitable AS(
    SELECT
    globalId,
    att_value.value as suitable_for_type
    FROM ``,
    unnest(attributes) as att,
    unnest(att.values) as att_value
    WHERE att.name in ('Suitable for Type of Laundry')),

 attributen_kind AS(
    SELECT
    globalId,
    att_value.value as kind_of_detergent
    FROM ``,
    unnest(attributes) as att,
    unnest(att.values) as att_value
    WHERE att.name in ('Kind of Laundry Detergent')),

 attributen_substance AS(
    SELECT
    globalId,
    att_value.value as substance
    FROM ``,
    unnest(attributes) as att,
    unnest(att.values) as att_value
    WHERE att.name in ('Substance')),

 attributen_number AS(
    SELECT
    globalId,
    att_value.value as number_of_washes
    FROM ``,
    unnest(attributes) as att,
    unnest(att.values) as att_value
    WHERE att.name in ('Number of Wash Treatments'))


SELECT
  global_id,
  brand_id,
  attributen_kind.kind_of_detergent,
  attributen_number.number_of_washes,
  attributen_substance.substance,
  attributen_suitable.suitable_for_type,
  party_name.partyName
FROM WasmiddelId as was
left join attributen_kind
on was.global_id=attributen_kind.globalId
left join attributen_number
on was.global_id=attributen_number.globalId
left join attributen_substance
on was.global_id=attributen_substance.globalId
left join attributen_suitable
on was.global_id=attributen_suitable.globalId
left join party_name
on was.brand_id=CAST(party_name.partyId as STRING)
order by global_id
