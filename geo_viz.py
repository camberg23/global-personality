# Code revision to make it work
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import copy
from datetime import datetime
from collections import Counter as count
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import streamlit as st

# Trait name mapping
trait_names = {
    'o': 'Openness',
    'c': 'Conscientiousness',
    'e': 'Extraversion',
    'a': 'Agreeableness',
    'n': 'Neuroticism'
}

traits = ['o', 'c', 'e', 'a', 'n']
us_or_global = st.radio('Data Scope:', ['US only', 'Global'])

def plot_us_trait_location(state_or_city, trait):
    if state_or_city == 'State':
        data_state_renamed = pd.read_csv("us_state_viz.csv")

        # Adjust the plotting code to use the renamed DataFrame and columns
        for trait, full_trait_name in trait_names.items():
            fig = px.choropleth(data_state_renamed, 
                                locations="State_Abbrev", 
                                locationmode="USA-states",
                                color=full_trait_name,
                                hover_name="State",
                                hover_data=[full_trait_name, 'Count', trait + "_std"],  # Order matters
                                color_continuous_scale='Viridis',
                                scope="usa",
                                title=f"US Map of Average {full_trait_name} Score by State")

            fig.update_traces(hovertemplate=f"<b>%{{hovertext}}, {full_trait_name}:</b><br>" +
                              "<br>" +
                              f"<b>Average: %{{customdata[0]:.2f}}</b><br>" +   # Index based on order in hover_data
                              f"Standard Dev.: %{{customdata[2]:.2f}}<br>" +   # Index based on order in hover_data
                              "User count: %{customdata[1]}"                 # Index based on order in hover_data
            )
            st.plotly_chart(fig)
    else:
        cluster_aggregates = pd.read_csv("us_city_viz.csv")
        # Map the traits to their full names for the color column
        for trait, full_name in trait_names.items():
            cluster_aggregates[full_name] = cluster_aggregates[trait]

        # Step 3: Plotting
        for trait, full_name in trait_names.items():
            fig = px.scatter_geo(cluster_aggregates, 
                                 locationmode='USA-states', 
                                 scope='usa',
                                 lat='Latitude',
                                 lon='Longitude',
                                 size='Count',
                                 color=full_name,
                                 hover_name='City',
                                 hover_data={full_name: True, 'Count': True, f"{trait}_std": True},
        #                          color_continuous_scale='Viridis',
                                 title=f"Bubble Map of {full_name} by Clustered US Cities",
                                 size_max=40
                                )

            fig.update_traces(
                hovertemplate=(
                    f"<b>%{{hovertext}} {full_name}:</b><br>" +
                    "<b>Average: %{customdata[0]:.2f}</b><br>" +
                    "Standard Dev.: %{customdata[2]:.2f}<br>" +
                    "User count: %{customdata[1]}")
            )


            fig.update_geos(center=dict(lat=38.0902, lon=-95.7129))
            st.plotly_chart(fig)


def plot_globe_trait_location(level, trait):
    # Function to aggregate data based on the selected level (Country or City)
    def generate_map_v2(trait, level, threshold_users=500):
        if level == "Country":
            data = pd.read_csv('country_data.csv')
        else:
            data = pd.read_csv('city_data.csv')
        full_trait_name = trait_names[trait]
        data[full_trait_name] = data[trait]

        if level == "Country":
            # Choropleth map for countries
            fig = px.choropleth(data, 
                                locations="Country", 
                                locationmode="country names",
                                color=full_trait_name,
                                hover_name="Country",
                                hover_data={"Count": True, trait: f"Mean {full_trait_name} Score", f"{trait}_std": f"Std Dev {full_trait_name}"},
    #                             color_continuous_scale=["black", "lightgreen"],
                                title=f"Map of Average {full_trait_name} Score by {level}")

            fig.update_traces(hovertemplate=f"<b>%{{hovertext}}, {full_trait_name}</b><br>Mean: %{{customdata[1]:.2f}}<br>Std Dev: %{{customdata[2]:.2f}}<br>Count: %{{customdata[0]}}")
            st.plotly_chart(fig)

        elif level == "City":
            # Clustering for cities
            kms_per_radian = 6371.0088
            epsilon = 50 / kms_per_radian
            city_counts_filtered = data[data['Count'] > threshold_users].copy()
            coords_global = city_counts_filtered[['Latitude', 'Longitude']].values
            db_global = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords_global))
            city_counts_filtered['Cluster'] = db_global.labels_

            clustered_data_global = city_counts_filtered[city_counts_filtered['Cluster'] != -1]
            cluster_aggregates_global = clustered_data_global.groupby('Cluster').agg({
                'Latitude': 'mean',
                'Longitude': 'mean',
                'Count': 'sum',
                'CityState': lambda x: x.value_counts().idxmax(),
                'Country': lambda x: x.value_counts().idxmax()
            }).reset_index()

            # Merging aggregated cluster data with trait data
            city_scores_global = data.groupby(['CityState', 'Country']).mean().reset_index()
            clustered_scores_global = cluster_aggregates_global.merge(city_scores_global, on=['CityState', 'Country'], how='left')
            clustered_scores_global.rename(columns={"Count_x": "Count", "Latitude_x": "Latitude", "Longitude_x": "Longitude"}, inplace=True)  # Renaming columns after merging

            # Bubble map for cities
            fig = px.scatter_geo(clustered_scores_global, 
                                 lat='Latitude',
                                 lon='Longitude',
                                 color=full_trait_name,
                                 hover_name='CityState',
                                 hover_data={trait: True, 'Count': True, 'Country': True, f"{trait}_std": f"Std Dev {full_trait_name}"},
    #                              color_continuous_scale='Viridis',
                                 title=f"World Map of {full_trait_name} by Clustered Major Cities")
            fig.update_traces(marker=dict(size=9))

            fig.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}, %{customdata[2]}</b><br>" +
                    f"Avg. {full_trait_name}: " + "%{customdata[0]:.2f}<br>" +
                    f"Std. Dev {full_trait_name}: " + "%{customdata[3]:.2f}<br>" +
                    "Count: %{customdata[1]}"
                )
            )

            # Add the extracted country boundaries to the cities' scatter map
            fig.update_geos(countrywidth=0.5, countrycolor="Black", showcountries=True)
            st.plotly_chart(fig)


#         fig.update_geos(center=dict(lat=38.0902, lon=-95.7129))
#         st.plotly_chart(fig)
    generate_map_v2(trait=trait_dropdown.value, level=level_radio.value)


if us_or_global == 'US only':
    state_or_city = st.radio('US Level:', ['State', 'City'])
    trait = st.selectbox('Trait:', list(trait_names.values()))
    plot_us_trait_location(state_or_city, trait)
    
else:
    level = st.radio('View by:', ['Country', 'City'])
    trait = st.selectbox('Trait:', list(trait_names.values()))
    plot_globe_trait_location(level, trait)
