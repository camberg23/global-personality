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
import openai

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

traits = list(trait_names.keys())

def plot_globe_trait_location(trait, level, scores, top_N=5, is_percentile=False):
        data = scores
        inv_trait_names = {v: k for k, v in trait_names.items()}

        trait_abbrev = inv_trait_names[trait]
        data[trait] = data[trait_abbrev]
        full_trait_name = trait

        if is_percentile:
                hover_data = {
                    "Count": True, 
                    trait: f"Percentile {trait}",  # Change here to display percentile
                }
                hover_template_country = f"<b>%{{hovertext}}, {trait}</b><br>Percentile: %{{customdata[1]:.2f}}<br>Count: %{{customdata[0]}}"
                hover_template_city = (
                    "<b>%{hovertext}, %{customdata[2]}</b><br>" +
                    f"Percentile {trait}: " + "%{customdata[0]:.2f}<br>" +
                    "Count: %{customdata[1]}"
                )
        else:
                hover_data = {
                    "Count": True, 
                    trait: f"Mean {trait}", 
                    f"{trait_abbrev}_std": f"Std Dev {trait}"
                }
                hover_template_country = f"<b>%{{hovertext}}, {trait}</b><br>Mean: %{{customdata[1]:.3f}}<br>Std Dev: %{{customdata[2]:.3f}}<br>Count: %{{customdata[0]}}"
                hover_template_city = (
                    "<b>%{hovertext}, %{customdata[2]}</b><br>" +
                    f"Avg. {trait}: " + "%{customdata[0]:.3f}<br>" +
                    f"Std. Dev {trait}: " + "%{customdata[3]:.3f}<br>" +
                    "Count: %{customdata[1]}"
                )

        if level == "Country view":
            # Choropleth map for countries
            fig = px.choropleth(data, 
                                locations="Country", 
                                locationmode="country names",
                                color=full_trait_name,
                                hover_name="Country",
                                hover_data={"Count": True, trait: f"Mean {full_trait_name} Score", f"{trait_abbrev}_std": f"Std Dev {full_trait_name}"},
                                color_continuous_scale=px.colors.sequential.Plasma)

            # fig.update_traces(hovertemplate=f"<b>%{{hovertext}}, {full_trait_name}</b><br>Mean: %{{customdata[1]:.3f}}<br>Std Dev: %{{customdata[2]:.3f}}<br>Count: %{{customdata[0]}}")
            fig.update_traces(hovertemplate=hover_template_country)
            fig.update_layout(width=1000, 
                             height=700,
                            title={
                            'text': f"Map of Average {full_trait_name} Score by Country",
                            'x': 0.5,
                            'y': 0.95,
                            'xanchor': 'center',
                            'font': {
                                'size': 30  # Adjust this value for desired font size
                                }
                            },
                             coloraxis_colorbar=dict(
                             lenmode="fraction", len=0.75,
                             yanchor="bottom", y=-0.1,
                             xanchor="center", x=0.5,
                             orientation="h"
                         ))
            st.plotly_chart(fig, use_container_width=True)

        elif level == "City view":
            # Clustering for cities
            city_counts = data.groupby('CityState').agg({'Count': 'sum'}).reset_index()
            
            # Filter to include only the top N cities by aggregated user count
            top_cities = city_counts.nlargest(top_N, 'Count')['CityState']
            
            # Filter your data to only include these top cities
            data = data[data['CityState'].isin(top_cities)]
            kms_per_radian = 6371.0088
            epsilon = 50 / kms_per_radian
            coords_global = data[['Latitude', 'Longitude']].values
            db_global = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords_global))
            data['Cluster'] = db_global.labels_

            clustered_data_global = data[data['Cluster'] != -1]
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
                                 hover_data={trait: True, 'Count': True, 'Country': True, f"{trait_abbrev}_std": f"Std Dev {full_trait_name}"},
                                 color_continuous_scale=px.colors.sequential.Plasma)
            fig.update_traces(marker=dict(size=9))
                
            # fig.update_traces(
            #     hovertemplate=(
            #         "<b>%{hovertext}, %{customdata[2]}</b><br>" +
            #         f"Avg. {full_trait_name}: " + "%{customdata[0]:.3f}<br>" +
            #         f"Std. Dev {full_trait_name}: " + "%{customdata[3]:.3f}<br>" +
            #         "Count: %{customdata[1]}"
            #     )
            # )

            fig.update_traces(hovertemplate=hover_template_city)
            # Add the extracted country boundaries to the cities' scatter map
            fig.update_geos(countrywidth=0.5, countrycolor="Black", showcountries=True)
            fig.update_layout(width=1000, 
                             height=700,
                            title={
                            'text': f"World Map of {full_trait_name} by Major Cities",
                            'x': 0.5,
                            'y': 0.95,
                            'xanchor': 'center',
                            'font': {
                                'size': 30  # Adjust this value for desired font size
                                }
                            },
                              coloraxis_colorbar=dict(
                             lenmode="fraction", len=0.75,
                             yanchor="bottom", y=-0.1,
                             xanchor="center", x=0.5,
                             orientation="h"
                 ))
            st.plotly_chart(fig, use_container_width=True)

def plot_us_trait_location(state_or_city, trait, scores,  top_N=100, is_percentile=False):
    inv_trait_names = {v: k for k, v in trait_names.items()}
    trait_abbrev = inv_trait_names[trait]

    if state_or_city == 'State view':
        data_state_renamed = scores
        full_trait_name = trait
        trait = inv_trait_names[trait]
        data_state_renamed[full_trait_name] = data_state_renamed[trait]

        fig = px.choropleth(data_state_renamed, 
                            locations="State_Abbrev", 
                            locationmode="USA-states",
                            color=full_trait_name,
                            hover_name="State",
                            hover_data=[trait, 'Count', trait + "_std"],
                            color_continuous_scale=px.colors.sequential.Plasma,
                            scope="usa")

        if is_percentile:
            hovertemplate = (f"<b>%{{hovertext}}, {full_trait_name}:</b><br>" +
                             f"<b>Percentile: %{{customdata[0]:.3f}}</b><br>" +
                             "User count: %{customdata[1]}")
        else:
            hovertemplate = (f"<b>%{{hovertext}}, {full_trait_name}:</b><br>" +
                             f"<b>Average: %{{customdata[0]:.3f}}</b><br>" +
                             f"Standard Dev.: %{{customdata[2]:.3f}}<br>" +
                             "User count: %{customdata[1]}")
            
        fig.update_traces(hovertemplate=hovertemplate)
        fig.update_layout(width=1000, 
                          height=700,
                          title={
                              'text': f"US Map of {full_trait_name} by State",
                              'x': 0.5,
                              'y': 0.95,
                              'xanchor': 'center',
                              'font': {'size': 30}
                          },
                            coloraxis_colorbar=dict(
                          lenmode="fraction", len=0.75,
                          yanchor="bottom", y=-0.1,
                          xanchor="center", x=0.5,
                          orientation="h"
                      ))
        st.plotly_chart(fig, use_container_width=True)

    else:  # City view
        cluster_aggregates = scores
        cluster_aggregates[trait] = cluster_aggregates[trait_abbrev]
        # CITIES N LARGEST
        cluster_aggregates = cluster_aggregates.nlargest(top_N, 'Count')
    
        fig = px.scatter_geo(cluster_aggregates, 
                             locationmode='USA-states', 
                             scope='usa',
                             lat='Latitude',
                             lon='Longitude',
                             size='Count',
                             color=trait,
                             hover_name='City',
                             hover_data={trait: True, 'Count': True, f"{trait_abbrev}_std": True},
                             custom_data=[trait, 'Count', f"{trait_abbrev}_std"],  # Add custom data here
                             color_continuous_scale=px.colors.sequential.Plasma,
                             size_max=60)
    
        if is_percentile:
            hovertemplate = (f"<b>%{{hovertext}} {trait}:</b><br>" +
                             f"<b>Percentile: %{{customdata[0]:.3f}}</b><br>" +  # Use it correctly here
                             f"User count: %{{customdata[1]:.0f}}")

        else:
            hovertemplate = (f"<b>%{{hovertext}} {trait}:</b><br>" +
                             f"<b>Average: %{{customdata[0]:.3f}}</b><br>" +  # And here
                             f"Standard Dev.: %{{customdata[2]:.3f}}<br>" +
                             f"User count: %{{customdata[1]:.0f}}")

        fig.update_traces(hovertemplate=hovertemplate)  # Don't forget to update the traces with the new hovertemplate

        fig.update_geos(center=dict(lat=38.0902, lon=-95.7129))
        fig.update_layout(width=1000, 
                          height=700,
                          title={
                              'text': f"US Bubble Map of {trait} by Major Cities",
                              'x': 0.5,
                              'y': 0.95,
                              'xanchor': 'center',
                              'font': {'size': 30}
                          },
                          coloraxis_colorbar=dict(
                          lenmode="fraction", len=0.75,
                          yanchor="bottom", y=-0.1,
                          xanchor="center", x=0.5,
                          orientation="h"
                      ))
        st.plotly_chart(fig, use_container_width=True)

def display_top_bottom_places(data, trait, scope, place_column, N=5, score_type="Normalized Scores"):
    """Display the top N and bottom N places based on the trait score."""
    trait_descriptions = {
        'Openness': 'open',
        'Conscientiousness': 'conscientious',
        'Extraversion': 'extraverted',
        'Agreeableness': 'agreeable',
        'Neuroticism': 'neurotic'
    }
    inv_trait_names = {v: k for k, v in trait_names.items()}
    full_name = trait
    trait = inv_trait_names[trait]
    description = trait_descriptions[full_name]

    original_scope = scope
    # Determine the scope based on place_column
    if place_column == "CityState":
        text_scope = "largest 1000 cities in the world"
    elif place_column == "City":
        text_scope = "largest 100 cities in the US"
    else:
        text_scope = scope
        
    # Sort the data based on the trait and take the top N and bottom N
    top_places = data.sort_values(by=trait, ascending=False).head(N)
    bottom_places = data.sort_values(by=trait, ascending=True).head(N)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<span style='font-size:1.4em;'><b>Highest {N} {original_scope} in {full_name}:</b></span>", unsafe_allow_html=True)
        for idx, (i, row) in enumerate(top_places.iterrows()):
            place_name = row[place_column]
            # Append country name if the scope is cities and global
            if 'Country' in data.columns and scope == 'cities':
                country_name = 'US' if row['Country'] == 'United States' else row['Country']
                place_name += f", {country_name}"
            if score_type == "Percentiles":
                st.markdown(f"<span style='font-size:1.2em;'>{idx+1}. <b>{place_name}</b>: more {description} than {row[trait]:.2f}% of {text_scope} (N={row['Count']} users)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='font-size:1.2em;'>{idx+1}. <b>{place_name}</b>: {row[trait]:.2f} ± {row[trait + '_std']:.2f} (N={row['Count']} users)</span>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<span style='font-size:1.4em;'><b>Lowest {N} {original_scope} in {full_name}:</b></span>", unsafe_allow_html=True)
        for idx, (i, row) in enumerate(bottom_places.iterrows()):
            place_name = row[place_column]
            # Append country name if the scope is cities and global
            if 'Country' in data.columns and scope == 'cities':
                country_name = 'US' if row['Country'] == 'United States' else row['Country']
                place_name += f", {country_name}"
            if score_type == "Percentiles":
                st.markdown(f"<span style='font-size:1.2em;'>{idx+1}. <b>{place_name}</b>: less {description} than {100 - row[trait]:.2f}% of {text_scope} (N={row['Count']} users)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='font-size:1.2em;'>{idx+1}. <b>{place_name}</b>: {row[trait]:.2f} ± {row[trait + '_std']:.2f} (N={row['Count']} users)</span>", unsafe_allow_html=True)

    return {
    'top': [row[place_column] for _, row in top_places.iterrows()],
    'bottom': [row[place_column] for _, row in bottom_places.iterrows()]
    }

def plot_comparison(scores1, scores2, std1, std2, label1, label2, count1, count2, traits, score_type, comparison_type):
    """Plot a side-by-side comparison of two entities over multiple traits."""
    
    # Organize data for grouped bar chart
    x_labels = traits
    y_values_plot_1 = [s if s > 0 else 1 for s in scores1]  # adjusting for plot
    y_values_plot_2 = [s if s > 0 else 1 for s in scores2]
    
    y_values_hover_1 = scores1  # original values for hover
    y_values_hover_2 = scores2

    # Set error bars to be invisible for percentiles
    error_visible = True if score_type == "Normalized Scores" else False

    # Create a grouped bar chart
    fig = go.Figure()

    # Set hovertemplate based on score_type
    if score_type == "Percentiles":
        hovertemplate1 = f"Higher than %{{customdata}}% of {comparison_type}<extra></extra>"
        hovertemplate2 = f"Higher than %{{customdata}}% of {comparison_type}<extra></extra>"
    else:
        hovertemplate1 = "Trait: %{x}<br>Score: %{customdata}<extra></extra>"
        hovertemplate2 = "Trait: %{x}<br>Score: %{customdata}<extra></extra>"

    # Bars for first entity
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_values_plot_1,
        customdata=y_values_hover_1,
        name=f"{label1} (n={count1:,})",
        error_y=dict(type='data', array=std1, visible=error_visible),
        marker_color='blue',
        hovertemplate=hovertemplate1
    ))

    # Bars for second entity
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_values_plot_2,
        customdata=y_values_hover_2,
        name=f"{label2} (n={count2:,})",
        error_y=dict(type='data', array=std2, visible=error_visible),
        marker_color='red',
        hovertemplate=hovertemplate2
    ))

    # Update layout for better visualization
    yaxis_title = "Scores (percentile)" if score_type == "Percentiles" else "Scores (normalized)"
    fig.update_layout(
        title={
        'text': f"Trait Comparison: <span style='color:blue'>{label1}</span> and <span style='color:red'>{label2}</span>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {
            'size': 24  # Adjust this value for desired font size
            }
        },
        xaxis_title="Traits",
        yaxis_title=yaxis_title,
        barmode='group',
        legend=dict(
            yanchor="top",
            y=1.15,
            xanchor="right",
            x=1
        ),
        font=dict(
            family="Roboto, monospace",
            size=18
        ),
        margin=dict(t=100)  # Add more space at the top
    )
    
    st.plotly_chart(fig, use_container_width=True)


def generate_personality_description(selected, percentiles, trait_names):    
    # Construct the initial system message
    system_message = """
                        You are a helpful assistant that provides a courteous and succinct summary of a location's overall personality blend based on Big Five personality traits percentiles.
                        Where applicable, try to seamlessly blend this information with concrete things that you know about the place to make it a harmonious, holistic, and accurate profile.
                        Always use relative language, as the information is based on percentiles, comparing the location's traits to the greater population.
                        Please note, while there are no 'good' or 'bad' personalities, it is generally considered desirable to be high in openness, consciousness, agreeableness, and extraversion, and low in neuroticism.
                        Please be a bit sensitive about this given a place's results.
                        YOU MUST LIMIT THIS OUTPUT TO ONE STRONG PARAGRAPH ONLY.
                        Finally: IN A SINGLE SENTENCE, give one interesting and/or unique fact about the location. FORMAT this ON A NEW LINE as "Bonus information about [location]:"
                        """

    # Construct user messages
    user_messages = [f"{selected} is in the {percentiles[trait]} percentile in the world for trait {trait_full}." for trait, trait_full in trait_names.items()]
    
    # Combine all messages
    messages = [{"role": "system", "content": system_message}]
    for msg in user_messages:
        messages.append({"role": "user", "content": msg})
    
    # Request a completion from the model
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract and return the model's response
    response = completion.choices[0].message['content']
    return response

def generate_list_explanation(places, trait, score_type):    
    # Construct the initial system message
    system_message = """
                        You are a helpful assistant that provides a courteous and succinct attempted explanation/educated guess of why particular places are rank highest and lowest for a particular Big Five trait.
                        Where applicable, try to seamlessly blend this information with concrete things that you know about the place to try to make sense of these rankings.
                        Always use relative language, as the information is based on percentiles, comparing the location's traits to the greater population.
                        Please note, while there are no 'good' or 'bad' personalities, it is generally considered desirable to be high in openness, consciousness, agreeableness, and extraversion, and low in neuroticism.
                        Please be a bit sensitive about this given a place's results.
                        YOU MUST LIMIT THIS OUTPUT TO ONE STRONG PARAGRAPH ONLY.
                        """

     # Construct user messages
    top_places = ', '.join(places['top'])
    bottom_places = ', '.join(places['bottom'])
    
    user_message = f"The highest ranking places for {trait} are {top_places}. The lowest ranking places for {trait} are {bottom_places}. Please provide a courteous and succinct attempted explanation/educated guess of why these particular places are ranking highest and lowest."
        
    # Combine all messages
    messages = [{"role": "system", "content": system_message}, {"role":"user", "content":user_message}]
    
    # Request a completion from the model
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # Extract and return the model's response
    response = completion.choices[0].message['content']
    return response

def generate_personality_comparison(selected1, selected2, percentiles1, percentiles2, trait_names, comparison_type):    
    # Construct the initial system message
    system_message = """
                        You are a helpful assistant that provides a courteous and succinct analysis and comparison of the percentile differences in Big Five traits between two places.
                        CRITICAL: Make sure you are giving an intuitive and helpful analysis of the differences rather than just restating the data.
                        Where applicable, blend this information with what you know about the place to make it a harmonious and accurate profile.
                        Always use relative language, as the information is based on percentiles, comparing the location's traits to the greater population.
                        Please note, while there are no 'good' or 'bad' personalities, it is generally considered desirable to be high in openness, consciousness, agreeableness, and extraversion, and low in neuroticism.
                        Please be a bit sensitive about this given places' results.
                        YOU MUST LIMIT OUTPUT TO ONE STRONG PARAGRAPH ONLY.
                        """

    # Construct user messages
    user_messages = []
    for trait, trait_full in trait_names.items():
        msg = f"{selected1} is in the {percentiles1[trait]} percentile and {selected2} is in the {percentiles2[trait]} percentile in the world for trait {trait_full.lower()}."
        user_messages.append(msg)
    
    # Combine all messages
    messages = [{"role": "system", "content": system_message}]
    for msg in user_messages:
        messages.append({"role": "user", "content": msg})
    
    # Request a completion from the model
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract and return the model's response
    response = completion.choices[0].message['content']
    return response

def plot_percentile(percentiles, trait_names_values, selected, comparison_type):
    """Plot the percentile scores as a horizontal bar chart."""
    
    # Handling the special case for 'US States'
    comparison_type = "US states" if comparison_type == "US States" else comparison_type.lower()

    # Mapping of trait names to adjectives
    trait_adj = {
        'Openness': 'open',
        'Conscientiousness': 'conscientious',
        'Extraversion': 'extraverted',
        'Agreeableness': 'agreeable',
        'Neuroticism': 'neurotic'
    }

    # Organize data for horizontal bar chart
    y_labels = list(trait_names_values.values())
    original_values = list(percentiles.values())
    x_values = [p if p >= 1 else 1 for p in original_values]  # Adjust x_values for plotting
    
    # Create a horizontal bar chart
    fig = go.Figure()

    for i, (y_label, x_value, original_value) in enumerate(zip(y_labels, x_values, original_values)):
        # Get the adjective for the current trait
        adj = trait_adj.get(y_label, 'unknown trait')
        
        # Bars for percentiles
        fig.add_trace(go.Bar(
            y=[y_label],  # make this a list to use in a loop
            x=[x_value],  # make this a list to use in a loop
            orientation='h',  # to make the bars horizontal
            text=[original_value],  # use the original value for the text
            textposition='inside',  # to position the text inside the bars
            marker_color='green',  # choosing a different color for distinction
            hovertemplate=(
                f"{selected} is more {adj} than {original_value:.2f}% of {comparison_type}<extra></extra>"
            ),
            showlegend=False  # do not show the legend
        ))

    # Update layout for better visualization
    fig.update_layout(
        title={
            'text': f"Personality Profile of {selected}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}  # Adjust this value for desired font size
        },
        yaxis_title="Traits",
        xaxis_title="Percentile",
        font=dict(
            family="Roboto, monospace",
            size=18
        ),
        margin=dict(t=100, l=200),  # Add more space at the top and left
        showlegend=False  # do not show the legend globally
    )
    
    return fig

def display_percentile(comparison_type, selected, data):
    if comparison_type == "Cities":
        selected_data = data[data['CityState'] == selected]
    elif comparison_type == "Countries":
        selected_data = data[data['Country'] == selected]
    elif comparison_type == "US States":
        selected_data = data[data['State'] == selected]
            
    trait_names = {'o': 'Openness', 'c': 'Conscientiousness', 'e': 'Extraversion', 'a': 'Agreeableness', 'n': 'Neuroticism'}
    percentiles = compute_percentile(data, selected_data, trait_names)
        
    fig = plot_percentile(percentiles, trait_names, selected, comparison_type)
    st.plotly_chart(fig, use_container_width=True)

    description = generate_personality_description(selected, percentiles, trait_names)
    st.write(f'**Personality profile of {selected}**:', description)

def compute_percentile(data, selected_data, trait_names):
    percentile_scores = {}
    for trait in trait_names:
        scores = data[trait].values
        if isinstance(selected_data, pd.DataFrame):
            if selected_data.empty:
                raise ValueError(f"No data found for the selected trait: {trait}")
            selected_data = selected_data.iloc[0]

        selected_score = selected_data[trait]
        
        # Exclude the current row's score when calculating the percentile
        scores = scores[scores != selected_score]
        percentile = 100 * len(scores[scores < selected_score]) / len(scores)
        percentile_scores[trait] = round(percentile, 2)
    return percentile_scores


def compute_percentiles_for_all(data, trait_names):
    new_data = data.copy()
    for i, row in new_data.iterrows():
        percentiles = compute_percentile(data.drop(index=i), row, trait_names)
        for trait, percentile in percentiles.items():
            new_data.at[i, trait] = percentile
    return new_data
