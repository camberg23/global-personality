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
import openai

import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import streamlit as st

def plot_globe_trait_location(trait, level, top_N=500):
        if level == "Country view":
            data = pd.read_csv('data/country_data.csv')
        else:
            data = pd.read_csv('data/city_data_fixed.csv')
            
        data = data[data['Count'] > THRESHOLD_USERS]  # Filter the data based on threshold_users for both countries and cities
    
        inv_trait_names = {v: k for k, v in trait_names.items()}

        trait_abbrev = inv_trait_names[trait]
        data[trait] = data[trait_abbrev]
        full_trait_name = trait

        if level == "Country view":
            # Choropleth map for countries
            fig = px.choropleth(data, 
                                locations="Country", 
                                locationmode="country names",
                                color=full_trait_name,
                                hover_name="Country",
                                hover_data={"Count": True, trait: f"Mean {full_trait_name} Score", f"{trait_abbrev}_std": f"Std Dev {full_trait_name}"},
                                color_continuous_scale=px.colors.sequential.Plasma)

            fig.update_traces(hovertemplate=f"<b>%{{hovertext}}, {full_trait_name}</b><br>Mean: %{{customdata[1]:.3f}}<br>Std Dev: %{{customdata[2]:.3f}}<br>Count: %{{customdata[0]}}")
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
                            })
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

            fig.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}, %{customdata[2]}</b><br>" +
                    f"Avg. {full_trait_name}: " + "%{customdata[0]:.3f}<br>" +
                    f"Std. Dev {full_trait_name}: " + "%{customdata[3]:.3f}<br>" +
                    "Count: %{customdata[1]}"
                )
            )

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
                            })
            st.plotly_chart(fig, use_container_width=True)

def plot_us_trait_location(state_or_city, trait, top_N=100):
    inv_trait_names = {v: k for k, v in trait_names.items()}
    trait_abbrev = inv_trait_names[trait]
    
    if state_or_city == 'State view':
        data_state_renamed = pd.read_csv("data/us_state_viz.csv")
        full_trait_name = trait
        trait = inv_trait_names[trait]
        data_state_renamed[full_trait_name] = data_state_renamed[trait]
        
        # Adjust the plotting code to use the renamed DataFrame and columns
        fig = px.choropleth(data_state_renamed, 
                            locations="State_Abbrev", 
                            locationmode="USA-states",
                            color=full_trait_name,
                            hover_name="State",
                            hover_data=[trait, 'Count', trait + "_std"],  # Order matters
                            color_continuous_scale=px.colors.sequential.Plasma,
                            scope="usa")

        fig.update_traces(hovertemplate=f"<b>%{{hovertext}}, {full_trait_name}:</b><br>" +
                          "<br>" +
                          f"<b>Average: %{{customdata[0]:.3f}}</b><br>" +   # Index based on order in hover_data
                          f"Standard Dev.: %{{customdata[2]:.3f}}<br>" +   # Index based on order in hover_data
                          "User count: %{customdata[1]}"                 # Index based on order in hover_data
        )
        fig.update_layout(width=1000, 
                          height=700,
                            title={
                            'text': f"US Map of {full_trait_name} by State",
                            'x': 0.5,
                            'y': 0.95,
                            'xanchor': 'center',
                            'font': {
                                'size': 30  # Adjust this value for desired font size
                                }
                            })
        st.plotly_chart(fig, use_container_width=True)
    else:
        cluster_aggregates = pd.read_csv("data/us_city_viz_improved.csv")
        cluster_aggregates[trait] = cluster_aggregates[trait_abbrev]

        # Step 3: Plotting
        fig = px.scatter_geo(cluster_aggregates, 
                             locationmode='USA-states', 
                             scope='usa',
                             lat='Latitude',
                             lon='Longitude',
                             size='Count',
                             color=trait,
                             hover_name='City',
                             hover_data={trait: True, 'Count': True, f"{trait_abbrev}_std": True},
                             color_continuous_scale=px.colors.sequential.Plasma,
                             size_max=60
                            )

        fig.update_traces(
            hovertemplate=(
                f"<b>%{{hovertext}} {trait}:</b><br>" +
                "<b>Average: %{customdata[0]:.3f}</b><br>" +
                "Standard Dev.: %{customdata[2]:.3f}<br>" +
                "User count: %{customdata[1]}")
        )


        fig.update_geos(center=dict(lat=38.0902, lon=-95.7129))
        fig.update_layout(width=1000, 
                          height=700,
                            title={
                            'text': f"US Bubble Map of {trait} by Major Cities",
                            'x': 0.5,
                            'y': 0.95,
                            'xanchor': 'center',
                            'font': {
                                'size': 30  # Adjust this value for desired font size
                                }
                            })
        st.plotly_chart(fig, use_container_width=True)

def display_top_bottom_places(data, trait, scope, place_column, N=5):
    """Display the top N and bottom N places based on the trait score."""
    inv_trait_names = {v: k for k, v in trait_names.items()}

    full_name = trait  # assign a default value to full_name here
    trait = inv_trait_names[trait]

    # Sort the data based on the trait and take the top N and bottom N
    top_places = data.sort_values(by=trait, ascending=False).head(N)
    bottom_places = data.sort_values(by=trait, ascending=True).head(N)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<span style='font-size:1.4em;'><b>Highest {N} {scope} in {full_name}:</b></span>", unsafe_allow_html=True)
        for idx, (i, row) in enumerate(top_places.iterrows()):
            place_name = row[place_column]
            # Append country name if the scope is cities and global
            if 'Country' in data.columns and scope == 'cities':
                country_name = 'US' if row['Country'] == 'United States' else row['Country']
                place_name += f", {country_name}"
            st.markdown(f"<span style='font-size:1.2em;'>{idx+1}. <b>{place_name}</b>: {row[trait]:.2f} ± {row[trait + '_std']:.2f}; N={row['Count']} users</span>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<span style='font-size:1.4em;'><b>Lowest {N} {scope} in {full_name}:</b></span>", unsafe_allow_html=True)
        for idx, (i, row) in enumerate(bottom_places.iterrows()):
            place_name = row[place_column]
            # Append country name if the scope is cities and global
            if 'Country' in data.columns and scope == 'cities':
                country_name = 'US' if row['Country'] == 'United States' else row['Country']
                place_name += f", {country_name}"
            st.markdown(f"<span style='font-size:1.2em;'>{idx+1}. <b>{place_name}</b>: {row[trait]:.2f} ± {row[trait + '_std']:.2f}; N={row['Count']} users</span>", unsafe_allow_html=True)

def plot_comparison(scores1, scores2, std1, std2, label1, label2, count1, count2, traits, score_type, comparison_type):
    """Plot a side-by-side comparison of two entities over multiple traits."""
    
    # Organize data for grouped bar chart
    x_labels = traits
    y_values_1 = [s if s > 0 else 1 for s in scores1]  # adding small value if percentile is 0
    y_values_2 = [s if s > 0 else 1 for s in scores2]

    # Set error bars to be invisible for percentiles
    error_visible = True if score_type == "Normalized Scores" else False

    # Create a grouped bar chart
    fig = go.Figure()

    # Set hovertemplate based on score_type
    if score_type == "Percentiles":
        hovertemplate1 = f"Higher than %{{y:.1f}}% of {comparison_type}<extra></extra>"
        hovertemplate2 = f"Higher than %{{y:.1f}}% of {comparison_type}<extra></extra>"
    else:
        hovertemplate1 = "Trait: %{x}<br>Score: %{y:.3f}<extra></extra>"
        hovertemplate2 = "Trait: %{x}<br>Score: %{y:.3f}<extra></extra>"

    # Bars for first entity
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_values_1,
        name=f"{label1} (n={count1:,})",
        error_y=dict(type='data', array=std1, visible=error_visible),
        marker_color='blue',
        hovertemplate=hovertemplate1
    ))

    # Bars for second entity
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_values_2,
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



def compute_percentile(data, selected_data, trait_names):
    percentile_scores = {}
    for trait in trait_names:
        scores = data[trait].values
        selected_score = selected_data[trait]
        percentile = 100 * len(scores[scores < selected_score]) / len(scores)
        percentile_scores[trait] = round(percentile, 2)
    return percentile_scores

def plot_percentile(percentiles, trait_names_values, selected):
    """Plot the percentile scores as a horizontal bar chart."""
    
    # Organize data for horizontal bar chart
    y_labels = list(trait_names_values.values())
    x_values = list(percentiles.values())

    # Create a horizontal bar chart
    fig = go.Figure()

    # Bars for percentiles
    fig.add_trace(go.Bar(
        y=y_labels,
        x=x_values,
        orientation='h',  # to make the bars horizontal
        text=x_values,
        textposition='inside',  # to position the text inside the bars
        marker_color='green',  # choosing a different color for distinction
        hovertemplate="Trait: %{y}<br>Percentile: %{x}%<extra></extra>"
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
    )
    
    return fig

def generate_personality_description(selected, percentiles, trait_names):    
    # Construct the initial system message
    system_message = """
                        You are a helpful assistant that provides a courteous and succinct summary of a location's overall personality blend based on Big Five personality traits percentiles.
                        Where applicable, blend this information with what you know about the place to make it a harmonious and accurate profile.
                        Always use relative language, as the information is based on percentiles, comparing the location's traits to the global population.
                        Please note, while there are no 'good' or 'bad' personalities, it is generally considered desirable to be high in openness, consciousness, agreeableness, and extraversion, and low in neuroticism.
                        Please be a bit sensitive about this given a place's results.
                        YOU MUST LIMIT OUTPUT TO ONE STRONG PARAGRAPH ONLY.
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

def generate_personality_comparison(selected1, selected2, percentiles1, percentiles2, trait_names, comparison_type):    
    # Construct the initial system message
    system_message = """
                        You are a helpful assistant that provides a courteous and succinct analysis and summary of the percentile differences in Big Five traits between two places.
                        Where applicable, blend this information with what you know about the place to make it a harmonious and accurate profile.
                        Always use relative language, as the information is based on percentiles, comparing the location's traits to the global population.
                        Please note, while there are no 'good' or 'bad' personalities, it is generally considered desirable to be high in openness, consciousness, agreeableness, and extraversion, and low in neuroticism.
                        Please be a bit sensitive about this given places' results.
                        YOU MUST LIMIT OUTPUT TO ONE STRONG PARAGRAPH ONLY.
                        """

    # Construct user messages
    user_messages = []
    for trait, trait_full in trait_names.items():
        msg = f"{selected1} is in the {percentiles1[trait]} percentile and {selected2} is in the {percentiles2[trait]} percentile in the world for trait {trait_full}."
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
        
def display_percentile(comparison_type, selected):
    if comparison_type == "Cities":
        data = pd.read_csv('data/top_1000_city_data.csv')
        selected_data = data[data['CityState'] == selected].iloc[0]
    else:
        data = pd.read_csv('data/country_data.csv')
        selected_data = data[data['Country'] == selected].iloc[0]

    trait_names = {'o': 'Openness', 'c': 'Conscientiousness', 'e': 'Extraversion', 'a': 'Agreeableness', 'n': 'Neuroticism'}
    percentiles = compute_percentile(data, selected_data, trait_names)

    description = generate_personality_description(selected, percentiles, trait_names)

    fig = plot_percentile(percentiles, trait_names, selected)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f'**Personality profile of {selected}**:', description)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Trait name mapping
trait_names = {
    'o': 'Openness',
    'c': 'Conscientiousness',
    'e': 'Extraversion',
    'a': 'Agreeableness',
    'n': 'Neuroticism'
}

traits = ['o', 'c', 'e', 'a', 'n']

THRESHOLD_USERS = 200

openai.organization = st.secrets['ORG']
openai.api_key = st.secrets['KEY']


st.set_page_config(layout="wide")

# Create a section title and space
st.title("Personality Atlas")
st.write("Explore and compare the Big Five personality traits across the globe using Truity's 4M person database!")

st.write("*Add some context and information here.*")
st.write("---")

col1, col2, col3, col4 = st.columns([1, 1, 1, 0.75])

with col1:
    us_or_global = st.selectbox('US or global:', ['Choose an option', 'US only', 'Global'])

# Conditionally display based on the first selection
if us_or_global == 'US only':
    with col2:
        state_or_city = st.selectbox('US scope:', ['Choose an option', 'State view', 'City view'])
elif us_or_global == 'Global':
    with col2:
        level = st.selectbox('Global scope:', ['Choose an option', 'Country view', 'City view'])

with col3:
    trait = st.selectbox('Big Five Trait:', ['Choose an option'] + list(trait_names.values()))

with col4:
    N = st.number_input('Top/bottom N to list:', min_value=1, max_value=50, value=5)

# Inside the main Streamlit code:
if st.button('Submit'):
    if us_or_global == 'US only' and trait != 'Choose an option' and state_or_city != 'Choose an option':
        if state_or_city == 'State view':
            state_scores = pd.read_csv('data/us_state_viz.csv')  # Load your state data here
            display_top_bottom_places(state_scores, trait, 'states', 'State', N)  # 'State' is the column name in state data
        elif state_or_city == 'City view':
            city_scores = pd.read_csv('data/us_city_viz_improved.csv')
            display_top_bottom_places(city_scores, trait, 'cities', 'City', N)
            
        plot_us_trait_location(state_or_city, trait)

    elif us_or_global == 'Global' and trait != 'Choose an option' and level != 'Choose an option':
        if level == "Country view":
            country_scores = pd.read_csv('data/country_data.csv')
            country_scores = country_scores[country_scores['Count'] > THRESHOLD_USERS]
            display_top_bottom_places(country_scores, trait, 'countries', 'Country', N)
            plot_globe_trait_location(trait, level)
        elif level == "City view":
            city_scores = pd.read_csv('data/top_1000_city_data.csv')
            city_scores = city_scores[city_scores['Count'] > THRESHOLD_USERS]
            display_top_bottom_places(city_scores, trait, 'cities', 'CityState', N)
            plot_globe_trait_location(trait, level)

# Create a section title and space
st.title("Personality profile of any location")
st.write("Get the average Big Five personality profiles of any location in our database.")
st.write("---")

# User Input
comparison_type = st.radio("Choose the type of place:", ["Cities", "Countries", "US States"], key='profile')

if comparison_type == "Cities":
    data = pd.read_csv('data/top_1000_city_data.csv')
    selected = st.selectbox("Select the city:", data['CityState'].unique(), key='profile_city')
elif comparison_type == "Countries":
    data = pd.read_csv('data/country_data.csv')
    selected = st.selectbox("Select the country:", data['Country'].unique(), key='profile_country')
else:  # Assuming comparison_type is "US States"
    data = pd.read_csv('data/us_state_viz.csv')
    selected = st.selectbox("Select the US state:", data['State'].unique(), key='profile_state')

if st.button('Submit', key='profile_button'):
        with st.spinner('Generating profile...'):
                display_percentile(comparison_type, selected)

# Create a section title and space
st.title("Population comparison tool")
st.write("Compare the average Big Five personality profiles of any two countries or cities.")
st.write("Note: there are almost always greater personality differences *within* a given location than *across* locations. Notice the large error bars (set score type to normalized scores), which signify significant trait diversity within each place.")
st.write("---")

# Select comparison type: City vs. City or Country vs. Country
comparison_type = st.radio("Would you like to compare cities or countries?", ["Cities", "Countries"])
# Handle City vs. City comparison
if comparison_type == "Cities":
    st.header("City Comparison")
    
    city_scores = pd.read_csv('data/top_1000_city_data.csv')  

    city_options = city_scores['CityState'] + ", " + city_scores['Country']

    default_city1_index = np.where(city_options == "New York, New York, United States")[0][0]
    default_city2_index = np.where(city_options == "Tokyo, Japan")[0][0]

    col1, col2, col3 = st.columns(3)
    city1_selected = col1.selectbox("Select the first city:", city_options, index=int(default_city1_index))
    city2_selected = col2.selectbox("Select the second city:", city_options, index=int(default_city2_index))
    score_type = col3.selectbox("Score Type:", ["Normalized Scores", "Percentiles"])

    city1_citystate, city1_country = city1_selected.rsplit(', ', 1)
    city2_citystate, city2_country = city2_selected.rsplit(', ', 1)

    city1_data = city_scores[(city_scores['CityState'] == city1_citystate) & (city_scores['Country'] == city1_country)].iloc[0]
    city2_data = city_scores[(city_scores['CityState'] == city2_citystate) & (city_scores['Country'] == city2_country)].iloc[0]
    
    percentiles1, percentiles2 = {}, {}
    if score_type == "Percentiles":
        percentiles1 = compute_percentile(city_scores, city1_data, trait_names)
        percentiles2 = compute_percentile(city_scores, city2_data, trait_names)
        city1_scores = list(percentiles1.values())
        city2_scores = list(percentiles2.values())
    else:
        city1_scores = [city1_data[trait] for trait in trait_names]
        city2_scores = [city2_data[trait] for trait in trait_names]

    city1_std = [city1_data[trait+'_std'] for trait in trait_names]
    city2_std = [city2_data[trait+'_std'] for trait in trait_names]

    city1_count = city_scores[(city_scores['CityState'] == city1_citystate) & (city_scores['Country'] == city1_country)]['Count'].values[0]
    city2_count = city_scores[(city_scores['CityState'] == city2_citystate) & (city_scores['Country'] == city2_country)]['Count'].values[0]

    if st.button('Submit', key='city_comparison_button'):
        with st.spinner('Generating comparison...'):
            plot_comparison(city1_scores, city2_scores, city1_std, city2_std, city1_selected, city2_selected, city1_count, city2_count, list(trait_names.values()), score_type, comparison_type.lower())
            if score_type == 'Percentiles':
                comparison_paragraph = generate_personality_comparison(city1_selected, city2_selected, percentiles1, percentiles2, trait_names, comparison_type)
                st.write(comparison_paragraph)


# Handle Country vs. Country comparison
elif comparison_type == "Countries":
    st.header("Country Comparison")
    country_scores = pd.read_csv('data/country_data.csv')

    country_scores = country_scores[country_scores['Count'] > THRESHOLD_USERS]
    
    default_country1_index = np.where(country_scores['Country'].unique() == "United States")[0][0]
    default_country2_index = np.where(country_scores['Country'].unique() == "Russia")[0][0]

    col1, col2, col3 = st.columns(3)
    country1_selected = col1.selectbox("Select the first country:", country_scores['Country'].unique(), index=int(default_country1_index))
    country2_selected = col2.selectbox("Select the second country:", country_scores['Country'].unique(), index=int(default_country2_index))
    score_type = col3.selectbox("Score Type:", ["Normalized Scores", "Percentiles"])

    country1_data = country_scores[country_scores['Country'] == country1_selected].iloc[0]
    country2_data = country_scores[country_scores['Country'] == country2_selected].iloc[0]
    
    percentiles1, percentiles2 = {}, {}
    if score_type == "Percentiles":
        percentiles1 = compute_percentile(country_scores, country1_data, trait_names)
        percentiles2 = compute_percentile(country_scores, country2_data, trait_names)
        country1_scores = list(percentiles1.values())
        country2_scores = list(percentiles2.values())
    else:
        country1_scores = [country1_data[trait] for trait in trait_names]
        country2_scores = [country2_data[trait] for trait in trait_names]

    country1_std = [country1_data[trait+'_std'] for trait in trait_names]
    country2_std = [country2_data[trait+'_std'] for trait in trait_names]

    country1_count = country_scores[country_scores['Country'] == country1_selected]['Count'].values[0]
    country2_count = country_scores[country_scores['Country'] == country2_selected]['Count'].values[0]

    if st.button('Submit', key='country_comparison_button'):
        with st.spinner('Generating comparison...'):
            plot_comparison(country1_scores, country2_scores, country1_std, country2_std, country1_selected, country2_selected, country1_count, country2_count, list(trait_names.values()), score_type, comparison_type.lower())
            if score_type == 'Percentiles':
                comparison_paragraph = generate_personality_comparison(country1_selected, country2_selected, percentiles1, percentiles2, trait_names, comparison_type)
                st.write(comparison_paragraph)
