import datetime as dt
import os

import geopandas as gpd
from logzero import logger
import streamlit as st

import helpers

GEO_DATA_LOC = "data/City_Council_Districts"

GEO_DATA_LOC = os.path.join(os.path.dirname(__file__), GEO_DATA_LOC)

PAGE_CONFIG = {
        "page_title": "Detroit 911 Calls",
        "layout": "wide",
        "page_icon": "oncoming_police_car",
    }

st.set_page_config(**PAGE_CONFIG)


# padding = 0

plot_config = {
    "displayModeBar": False,
    }


j_df = gpd.read_file(GEO_DATA_LOC).to_crs({"init": "epsg:4326"})

df_scoring_data, df_predictions_data = helpers.pull_snowflake_tables(dt.date.today())

df_scoring_data, df_predictions_data = helpers.clean_data(
    df_scoring_data, df_predictions_data
)

message_bar = st.empty()
st.title("Detroit 911 Call Forecaster")

district_options = df_scoring_data.District.unique().tolist()
priority_options = df_scoring_data.Priority.astype(int).unique().tolist()

line_chart = st.empty()
stats_column, map_column = st.columns([1, 4])
stats_chart1 = stats_column.empty()
stats_chart2 = stats_column.empty()
map_chart = map_column.empty()

area_charts = st.empty()

default_districts = ["District 1"]
default_priorities = list(range(1, 6))
default_time_aggregation_index = 2

default_scoring_data = df_scoring_data.loc[
    lambda x: (x.District.isin(default_districts))
    & (x.Priority.astype(int).isin(default_priorities))
]
default_predictions_data = df_predictions_data.loc[
    lambda x: (x.District.isin(default_districts))
    & (x.Priority.isin(default_priorities))
]

with st.sidebar.form(key="my_form"):

    districts = st.multiselect(
        "Select District(s) of Interest",
        district_options,
        default=default_districts,
    )
    priorities = st.multiselect(
        "Select Priority(s) of Interest",
        priority_options,
        default=default_priorities,
    )
    aggregate_districts = st.selectbox(
        "Aggregate Districts?",
        [False, True],
        format_func=lambda x: "Y" if x else "N",
    )
    time_aggregation = st.selectbox(
        "Select Time Aggregation",
        ["hourly", "3-hours", "6-hours", "12-hours"],
        index=default_time_aggregation_index,
    )

    pressed = st.form_submit_button("Plot Forecast")

    st.write(
        """If the data is not up to date, try a fresh pull from 
        the database by clicking the clear cache button below."""
    )
    clear_cache = st.form_submit_button("Clear Cache")

expander = st.sidebar.expander("What is this?")
expander.write(
    """
    This app is a tool for forecasting the number of 911 calls in the Detroit area.
    Users can pick a district of interest and a time aggregation. The app will then 
    show a two day forecast of the number of 911 calls in the district.
    """
)

# chart, _, map, _ = st.columns([3, 2, 3, 2])
if pressed == 0:
    line_chart.plotly_chart(
        helpers.plot_lines(
            df_scoring_data=default_scoring_data,
            df_predictions_data=default_predictions_data,
            time_aggregation=default_time_aggregation_index,
            aggregate_districts=False,
        ),
        config=plot_config,
        use_container_width=True,
    )
    helpers.show_some_stats(
        default_scoring_data, default_predictions_data, stats_chart1, stats_chart2
    )
    map_chart.plotly_chart(
        helpers.plot_map(default_scoring_data, default_predictions_data, j_df),
        config=plot_config,
        use_container_width=True,
    )

    area_charts.plotly_chart(
        helpers.plot_area_chart(
            helpers.format_data_for_area_chart(default_scoring_data),
            helpers.format_data_for_area_chart(default_predictions_data),
        ),
        config=plot_config,
        use_container_width=True,
    )


if pressed:

    filter_scoring_data = df_scoring_data.loc[
        lambda x: (x.District.isin(districts))
        & (x.Priority.astype(int).isin(priorities))
    ]
    filter_predictions_data = df_predictions_data.loc[
        lambda x: (x.District.isin(districts)) & (x.Priority.isin(priorities))
    ]
    
    line_chart.plotly_chart(
        helpers.plot_lines(
            df_scoring_data=filter_scoring_data,
            df_predictions_data=filter_predictions_data,
            time_aggregation=time_aggregation,
            aggregate_districts=aggregate_districts,
        ),
        config= plot_config,
        use_container_width=True,
    )
    helpers.show_some_stats(
        filter_scoring_data, filter_predictions_data, stats_chart1, stats_chart2
    )
    map_chart.plotly_chart(
        helpers.plot_map(filter_scoring_data, filter_predictions_data, j_df),
        config = plot_config,
        use_container_width=True,
    )
    area_charts.plotly_chart(
        helpers.plot_area_chart(
            helpers.format_data_for_area_chart(filter_scoring_data),
            helpers.format_data_for_area_chart(filter_predictions_data)
        ),
        config=plot_config,
        use_container_width=True,
    )
if clear_cache:
    st.legacy_caching.clear_cache()
    message_bar.info("Cache cleared.")


