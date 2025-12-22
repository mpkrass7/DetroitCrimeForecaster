
CREATE OR REPLACE MATERIALIZED VIEW basic_prophet_forecasts
    with horizon as (
        SELECT 
            date_add(max(called_at_hour), 3) as forecast_horizon
        FROM incidents_forecasting_gold
    )
    SELECT
        council_district,
        priority,
        called_at_hour,
        round(call_count_forecast,2) as call_count_forecast
        FROM AI_FORECAST(
        TABLE(incidents_forecasting_gold),
        horizon => (SELECT forecast_horizon FROM horizon),
        time_col => 'called_at_hour',
        value_col => 'call_count',
        group_col => array('council_district', 'priority'),
        parameters => '{"global_floor": 0}'
    )