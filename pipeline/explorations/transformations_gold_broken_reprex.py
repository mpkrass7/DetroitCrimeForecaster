import datetime as dt
import logging

from pyspark import pipelines as dp
from pyspark.sql.functions import col, explode, expr, lit, sequence

latest_allowed_date = (dt.datetime.now() - dt.timedelta(days=1)).date()

logger = logging.getLogger(__name__)


def _aggregate_data(df):
    return (
        df.groupBy("council_district", "priority", "called_at_hour")
        .count()
        .withColumnRenamed("count", "call_count")
    )


def _create_interpolation_data(df):
    """
    Create a dataframe with all possible combinations of dates, districts and priorities.
    """

    min_date, max_date = df.selectExpr(
        "min(called_at_hour)", "max(called_at_hour)"
    ).first()

    min_date = dt.datetime.strftime(min_date, "%Y-%m-%d %H:%M:%S")
    max_date = dt.datetime.strftime(max_date, "%Y-%m-%d %H:%M:%S")
    districts = df.select("council_district").distinct()
    priorities = df.select("priority").distinct()

    date_range = spark.range(1).select(
        explode(
            sequence(
                lit(min_date).cast("timestamp"),
                lit(max_date).cast("timestamp"),
                expr("interval 1 hour"),
            )
        ).alias("called_at_hour")
    )
    return date_range.crossJoin(districts).crossJoin(priorities)


@dp.materialized_view
def incidents_forecasting_gold():
    """
    Aggregate data by hour, district and priority.

    Fill in missing dates via lots of interpolation
    """

    df_silver = spark.read.table("incidents_silver")

    df_aggregate = _aggregate_data(df_silver)

    logger.info(f"Aggregated data has {df_aggregate.count()} rows")

    interpolate_df = _create_interpolation_data(df_aggregate)

    return (
        interpolate_df.join(
            df_aggregate,
            on=["council_district", "priority", "called_at_hour"],
            how="left",
        )
        .fillna(0)
        .filter(col("called_at_hour").cast("date") <= latest_allowed_date)
        .orderBy("council_district", "priority", "called_at_hour")
    )
