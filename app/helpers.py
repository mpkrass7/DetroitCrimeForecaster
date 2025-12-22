import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config, oauth_service_principal
from dotenv import load_dotenv
from logzero import logger
from plotly.subplots import make_subplots

load_dotenv()

# Running locally using streamlit run
logger.info("Running helpers locally")
st.session_state.DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
st.session_state.DATABRICKS_CLIENT_ID = os.getenv("DATABRICKS_CLIENT_ID")
st.session_state.DATABRICKS_CLIENT_SECRET = os.getenv("DATABRICKS_CLIENT_SECRET")
DATABRICKS_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
st.session_state.DATABRICKS_HTTP_PATH = f"/sql/1.0/warehouses/{DATABRICKS_WAREHOUSE_ID}"

w = WorkspaceClient(
    host=st.session_state.DATABRICKS_HOST,
    client_id=st.session_state.DATABRICKS_CLIENT_ID,
    client_secret=st.session_state.DATABRICKS_CLIENT_SECRET
    )

plotly_default_colors = px.colors.qualitative.Plotly[:7]
districts = [f"District {i}" for i in range(1, 8)]
DISTRICT_COLOR_MAPPING = dict(zip(districts, plotly_default_colors))
DISTRICT_COLOR_MAPPING["Aggregated Districts"] = plotly_default_colors[0]


def build_layout():
    container0 = st.container()
    container1 = st.container()
    container2 = st.container()
    stats_column, map_column = container2.columns([1, 4])
    container3 = st.container()
    container4 = st.container()
    return (
        container0,
        container1,
        container2,
        stats_column,
        map_column,
        container3,
        container4,
    )


def credential_provider():
  config = Config(
    host          = f"https://{st.session_state.DATABRICKS_HOST}",
    client_id     = st.session_state.DATABRICKS_CLIENT_ID,
    client_secret = st.session_state.DATABRICKS_CLIENT_SECRET)
  return oauth_service_principal(config)


@st.cache_data()
def pull_tables():
    CATALOG, SCHEMA,  = "mk_fiddles", "detroit_911"
    DATA_TABLE = "incidents_forecasting_gold"
    PREDICTION_TABLE = "basic_prophet_forecasts"


    logger.info("Connecting to Databricks Tables...")
    # Create the connection to the Databricks database.
    with sql.connect(
        server_hostname      = st.session_state.DATABRICKS_HOST,
        http_path            = st.session_state.DATABRICKS_HTTP_PATH,
        credentials_provider = credential_provider) as cnx:
 
        # Create a Cursor
        cur = cnx.cursor()
        logger.info("Pulling 911 calls over past 2 months...")

        # Count out of date records.
        query = f"""
            SELECT *  
            FROM {CATALOG}.{SCHEMA}.{DATA_TABLE} as a 
            WHERE a.called_at_hour >= (
                SELECT ADD_MONTHS(MAX(b.called_at_hour),-2) FROM {CATALOG}.{SCHEMA}.{DATA_TABLE} as b
                )
                """
        cur.execute(query)
        scoring_data = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        df_scoring_data = pd.DataFrame(scoring_data, columns=columns)
        latest_call = df_scoring_data["called_at_hour"].max()
        logger.info("Pulling Predictions...")

        query = f"""
            SELECT * FROM {CATALOG}.{SCHEMA}.{PREDICTION_TABLE}
            WHERE called_at_hour > '{latest_call}'
            """

        cur.execute(query)

        predictions_data = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        df_predictions_data = pd.DataFrame(predictions_data, columns=columns)

        # logger.info(df_scoring_data.head())

        cnx.close()
        return (
            df_scoring_data,
            df_predictions_data,
        )


def clean_data(scoring_data: pd.DataFrame, predictions_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rename columns and format data types for scoring and prediction dataframes.
    """

    df_scoring_data = scoring_data.copy()
    df_predictions_data = predictions_data.copy()

    for df in [df_scoring_data, df_predictions_data]:
        df['council_district'] = "District " + df["council_district"].str[-1]
        df['priority'] = df["priority"].str[-1].astype(int)
        df.rename(
            columns={
                "called_at_hour": "Call Hour",
                "call_count": "Incidents",
                "call_count_forecast": "Incidents",
                "council_district": "District",
                "priority": "Priority",
            },
            inplace=True,
        )
    
    return df_scoring_data, df_predictions_data


def show_some_stats(actuals: pd.DataFrame, stats_box1: DeltaGenerator, stats_box2: DeltaGenerator):
    """
    Show some key statistics about the latest data.
    """
    latest_call = actuals["Call Hour"].max()
    actuals_last_two_weeks = (
        actuals.loc[lambda x: x["Call Hour"] >= latest_call - dt.timedelta(days=14)]
        .assign(
            high_priority_incidents=lambda x: np.where(x["Priority"].isin([4, 5]), 1, 0)
        )
        .assign(this_week=lambda x: x["Call Hour"] > latest_call - dt.timedelta(days=7))
    )
    actuals_last_week = actuals_last_two_weeks.loc[lambda x: x["this_week"] == True]
    actuals_prior_week = actuals_last_two_weeks.loc[lambda x: x["this_week"] == False]
    stats_box1.metric(
        label="Incidents in Last Week",
        value="{:,}".format(actuals_last_week["Incidents"].sum()),
        delta=f"{round((actuals_last_week['Incidents'].sum()/actuals_prior_week['Incidents'].sum() - 1) * 100 , 1)}%",
    )
    stats_box2.metric(
        label="High Priority Incidents in Last Week",
        value="{:,}".format(actuals_last_week["high_priority_incidents"].sum()),
        delta=f"{round((actuals_last_week['high_priority_incidents'].sum()/actuals_prior_week['high_priority_incidents'].sum() - 1) * 100, 1)}%",
    )

    return


def aggregate_plot_data(
    df_scoring_data: pd.DataFrame, df_predictions_data: pd.DataFrame, time_aggregation: str, aggregate_districts=True
):

    time_aggregation = "6-hours" if time_aggregation == 2 else time_aggregation
    agg_columns = ["Call Hour"] if aggregate_districts else ["Call Hour", "District"]
    hour_mapping = {
        "12-hours": "12h",
        "6-hours": "6h",
        "3-hours": "3h",
        "hourly": "h",
    }
    format_scoring_data, format_prediction_data = (
        df_scoring_data.copy(),
        df_predictions_data.copy(),
    )
    format_scoring_data["Call Hour"] = format_scoring_data["Call Hour"].dt.round(
        hour_mapping[time_aggregation], ambiguous=False
    )
    format_prediction_data["Call Hour"] = format_prediction_data["Call Hour"].dt.round(
        hour_mapping[time_aggregation], ambiguous=False
    )

    format_scoring_data = (
        format_scoring_data.groupby(agg_columns)["Incidents"].sum().reset_index()
    )
    format_prediction_data = (
        format_prediction_data.groupby(agg_columns)["Incidents"].sum().reset_index()
    )

    if aggregate_districts:
        format_scoring_data = format_scoring_data.assign(
            District="Aggregated Districts"
        )
        format_prediction_data = format_prediction_data.assign(
            District="Aggregated Districts"
        )

    return format_scoring_data, format_prediction_data


def plot_lines(
    df_scoring_data: pd.DataFrame,
    df_predictions_data: pd.DataFrame,
    time_aggregation:str="hourly",
    aggregate_districts:bool=False,
):

    scoring_data_format, prediction_data_format = aggregate_plot_data(
        df_scoring_data, df_predictions_data, time_aggregation, aggregate_districts
    )

    fig = go.Figure()
    for _, i in enumerate(scoring_data_format.District.unique().tolist()):
        score_temp = scoring_data_format.loc[lambda x: x.District == i]
        pred_temp = prediction_data_format.loc[lambda x: x.District == i]
        fig.add_trace(
            go.Scatter(
                x=score_temp["Call Hour"],
                y=score_temp["Incidents"],
                name=i,
                opacity=0.9,
                legendgroup=i,
                line=dict(color=DISTRICT_COLOR_MAPPING[i], width=3),
                # mode="lines+markers",
                line_shape="spline",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pred_temp["Call Hour"],
                y=pred_temp["Incidents"],
                name=f"forecast {i}",
                opacity=0.75,
                legendgroup=i,
                line=dict(color=DISTRICT_COLOR_MAPPING[i], width=3, dash="dashdot"),
                line_shape="spline",
            )
        )

        for trace in fig["data"]:
            trace["line_shape"] = "spline"
            if "forecast" in trace["name"]:
                trace["showlegend"] = False

    fig = fig.update_layout(
        plot_bgcolor="white",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
        ),
        margin=dict(t=10, l=0, b=10, r=10),
        xaxis_title="Hour of Call",
        yaxis_title="Number of Incidents",
        width=1000,
        legend=dict(yanchor="top", y=0.95, xanchor="center", x=1.08),
        # height=600,
    )
    fig = fig.update_yaxes(title_font={"size": 20})
    fig = fig.update_xaxes(title_text="Call Hour", title_font={"size": 20})
    fig["layout"]["yaxis"]["fixedrange"] = True

    return fig


def plot_map(actuals: pd.DataFrame, coordinates_dataframe: pd.DataFrame) -> go.Figure:

    latest_call = actuals["Call Hour"].max()
    actuals_last_week = (
        actuals.loc[lambda x: x["Call Hour"] > latest_call - dt.timedelta(days=7)]
        .assign(
            high_priority_incidents=lambda x: np.where(x["Priority"].isin([4, 5]), 1, 0)
        )
        .groupby("District")[["Incidents", "high_priority_incidents"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "Incidents": "Incidents in last 7 Days",
                "high_priority_incidents": "High Priority Incidents in last 7 Days",
            }
        )
        .assign(district_color=lambda x: x.District.map(DISTRICT_COLOR_MAPPING))
    )
    actuals_plus_aggregation = actuals.merge(actuals_last_week, on="District")

    json_coordinates = json.loads(coordinates_dataframe.to_json())
    for i in range(len(json_coordinates["features"])):
        new_id = json_coordinates["features"][i]["properties"]["District"][1]
        json_coordinates["features"][i]["id"] = f"District {new_id}"
    district_colors = actuals_plus_aggregation.district_color.unique()
    fig = px.choropleth_mapbox(
        data_frame=actuals_plus_aggregation,
        geojson=json_coordinates,
        locations="District",
        color="District",
        color_discrete_sequence=district_colors,
        mapbox_style="carto-positron",
        hover_data=[
            "District",
            "Incidents in last 7 Days",
            "High Priority Incidents in last 7 Days",
        ],
        zoom=9.6,
        opacity=0.3,
        center={"lat": 42.3584, "lon": -83.0858},
    )
    # fig = fig.update_traces(
    #     hovertemplate="%{District} Incidents: %{Incidents in Last Week}",
    # )
    fig.update_layout(
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        margin=dict(t=10, l=0, b=10, r=10),
    )

    return fig


def format_data_for_area_chart(
    df: pd.DataFrame,
) -> pd.DataFrame:
    melt_df = df.assign(
        Priority=lambda x: "Priority " + x.Priority.astype(str),
        call_hour=lambda x: x["Call Hour"].astype(str),
    )
    melt_df.columns = [i.upper() for i in melt_df.columns]
    mdf_total = (
        melt_df.sort_values(by=["CALL_HOUR", "PRIORITY", "DISTRICT"])
        .reset_index(drop=True)
        .groupby(["CALL_HOUR", "PRIORITY"])["INCIDENTS"]
        .sum()
        .reset_index()
    )

    mdf_total["pct_total"] = mdf_total.groupby(["CALL_HOUR"])["INCIDENTS"].transform(
        lambda x: np.round(x / x.sum() * 100, 2)
    )
    return mdf_total


def plot_area_chart(melted_df: pd.DataFrame, melted_predict_df: pd.DataFrame) -> go.Figure:

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        column_widths=[0.6],
        #     row_heights=[0.4, 0.6],
        specs=[[{"type": "scatter"}], [{"type": "scatter"}]],
    )

    priority_colors = [
        f"#{i}" for i in ["6a040f", "d62828", "f77f00", "fcbf49", "eae2b7"][::-1]
    ]

    for count, i in enumerate(melted_df.PRIORITY.unique().tolist()):
        mdf = melted_df.loc[lambda x: x.PRIORITY == i]
        mpdf = melted_predict_df.loc[lambda x: x.PRIORITY == i].loc[
            lambda x: x.CALL_HOUR >= max(mdf.CALL_HOUR)
        ]
        fig = fig.add_trace(
            go.Scatter(
                x=mdf["CALL_HOUR"],
                y=mdf["pct_total"],
                name=i,
                opacity=1,
                line=dict(width=0.5, color=priority_colors[count]),
                stackgroup="one",
                legendgroup=i,
            ),
            row=1,
            col=1,
        )
        fig = fig.add_trace(
            go.Scatter(
                x=mpdf["CALL_HOUR"],
                y=mpdf["pct_total"],
                name=f"Forecast: {i}",
                opacity=0.5,
                line=dict(width=0.5, color=priority_colors[count]),
                stackgroup="two",
                legendgroup=i,
            ),
            row=1,
            col=1,
        )

        fig = fig.add_trace(
            go.Scatter(
                x=mdf["CALL_HOUR"],
                y=mdf["INCIDENTS"],
                name=i,
                line=dict(width=0.5, color=priority_colors[count]),
                stackgroup="one",
                legendgroup=i,
            ),
            row=2,
            col=1,
        )
        fig = fig.add_trace(
            go.Scatter(
                x=mpdf["CALL_HOUR"],
                y=mpdf["INCIDENTS"],
                name=f"Forecast: {i}",
                opacity=0.5,
                line=dict(width=0.5, color=priority_colors[count]),
                stackgroup="two",
                legendgroup=i,
            ),
            row=2,
            col=1,
        )

    for trace in fig["data"]:
        trace["line_shape"] = "spline"
        if "Forecast" in trace["name"]:
            trace["showlegend"] = False

    for trace in fig["data"]:
        if trace["yaxis"] == "y2":
            trace["showlegend"] = False

    fig = fig.update_layout(
        plot_bgcolor="white",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
        ),
        margin=dict(t=10, l=0, b=10, r=10),
        xaxis_title="Hour of Call",
        yaxis_title="Number of Incidents",
        height=800,
        width=1200,
        legend=dict(yanchor="top", y=0.6, xanchor="center", x=1.08),
    )

    fig = fig.update_xaxes(title_text="Call Hour", title_font={"size": 20})

    # A bunch of layout configurations
    fig["layout"]["xaxis"]["title"] = None
    fig["layout"]["xaxis2"]["title"] = {"font": {"size": 18}, "text": "Call Hour"}
    fig["layout"]["yaxis"]["ticksuffix"] = "%"
    fig["layout"]["yaxis"]["range"] = [-0.5, 100]
    fig["layout"]["yaxis"]["title"] = {
        "font": {"size": 16},
        "text": "Percent of\n Total Incdients",
    }
    fig["layout"]["yaxis"]["fixedrange"] = True
    fig["layout"]["yaxis2"]["fixedrange"] = True
    # fig['layout']['yaxis']
    fig["layout"]["yaxis2"]["title"] = {
        "font": {"size": 16},
        "text": "Number of Incidents",
    }
    return fig
