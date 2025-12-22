from pyspark import pipelines as dp
from pyspark.sql.functions import col, concat, expr, lit


@dp.table
def incidents_silver():
    """
    Basic data cleaning operations

    Filter null council districts and priorities
    Format dates
    """

    return (
        spark.readStream.table("incidents_bronze")
        .filter(col("priority").between("1","5"))
        .filter(col("council_district").between(1, 7))
        .withColumn("council_district", concat(lit("D"), col("council_district").cast("int").cast("string")))
        .withColumn("priority", concat(lit("P"), col("priority")))
        .withColumn("called_at_timestamp", (col("called_at") / 1000).cast("timestamp"))
        .withColumn("called_at_date", col("called_at_timestamp").cast("date"))
        .withColumn("called_at_hour", expr("date_trunc('HOUR', called_at_timestamp)"))
    )