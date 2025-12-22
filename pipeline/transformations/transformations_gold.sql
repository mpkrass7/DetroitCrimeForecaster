
CREATE OR REPLACE MATERIALIZED VIEW incidents_forecasting_gold

    with aggregate_data as (
        SELECT 
            council_district,
            priority,
            called_at_hour,
            count(*) as call_count
        FROM incidents_silver
        GROUP BY ALL
    ),
    districts_priorities as (select distinct council_district, priority from incidents_silver),
    min_max_date as (
        SELECT 
            MIN(called_at_hour) as min_hour, 
            MAX(called_at_hour) as max_hour 
        FROM incidents_silver
    ),
    interpolate_df as (
        SELECT 
            council_district,
            priority,
            explode(
                sequence(CAST(min_hour AS TIMESTAMP), CAST(max_hour AS TIMESTAMP), INTERVAL 1 HOUR)
                ) AS called_at_hour
        FROM min_max_date
        CROSS JOIN districts_priorities
    )
    SELECT
        i.council_district, 
        i.priority,
        i.called_at_hour,
        coalesce(a.call_count, 0) as call_count
    FROM interpolate_df as i
    LEFT JOIN aggregate_data as a
    ON i.called_at_hour = a.called_at_hour
    AND i.council_district = a.council_district
    AND i.priority = a.priority
    WHERE a.called_at_hour < current_date
;