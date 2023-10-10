from geo_viz_utils import * 

# Trait name mapping
trait_names = {
    'o': 'Openness',
    'c': 'Conscientiousness',
    'e': 'Extraversion',
    'a': 'Agreeableness',
    'n': 'Neuroticism'
}

traits = list(trait_names.keys())

THRESHOLD_USERS = 200

openai.organization = st.secrets['ORG']
openai.api_key = st.secrets['KEY']

st.set_page_config(page_title="Personality Atlas", layout="wide")

# # Adjusted Title and Logo using Flexbox
# st.markdown(
#     """
#     <div style="display: flex; align-items: center; justify-content: center;">
#         <h1 style='font-size:4em; margin-right: -35px;'>Personality Atlas <span style='font-size: 0.6em;'>by</span></h1>
#         <a href="https://www.truity.com/" target="_blank">
#             <img src="https://d31u95r9ywbjex.cloudfront.net/sites/all/themes/bootstrap_truity/images-new/truity_logo.png" style="width:150px; transform: translateY(-50px);">
#         </a>
#     </div>
#     """, 
#     unsafe_allow_html=True
# )
# Adjusted Logo and Title using Flexbox
# st.markdown(
#     """
#     <div style="display: flex; align-items: center; justify-content: center;">
#         <a href="https://www.truity.com/" target="_blank">
#             <img src="https://d31u95r9ywbjex.cloudfront.net/sites/all/themes/bootstrap_truity/images-new/truity_logo.png" style="width:155px; transform: translateY(-50px)">
#         </a>
#         <h1 style='font-size:3.5em; margin-left: 14px'>Personality Atlas</h1>
#     </div>
#     """, 
#     unsafe_allow_html=True
# )

with st.expander("**Welcome to Truity’s Personality Atlas! Click to read more about this project.**"):
    st.markdown("Welcome to Truity’s **Big Five Personality Atlas**, where you can explore and compare the [Big Five](https://en.wikipedia.org/wiki/Big_Five_personality_traits) personality traits from across the globe, powered by Truity's 4M person database.")
    
    st.markdown("This page is split up into three different tools: (1) an **interactive personality map**, (2) a **personality profile generator** by location, and (3) a **head-to-head comparison tool**.")
    
    st.markdown("You can take [Truity’s validated Big Five personality assessment here](https://www.truity.com/test/big-five-personality-test). The code and data used for generating these analyses is [publicly available](https://github.com/camberg23/global-personality).")

    st.markdown(
        """
        Longstanding evidence suggests that the Big Five is a valid measure of personality cross-culturally
        [[1](https://journals.sagepub.com/doi/10.1177/0022022198291009),[2](https://journals.sagepub.com/doi/abs/10.1177/0022022106297299)]. 
        See [here](https://en.wikipedia.org/wiki/Big_Five_personality_traits_and_culture) for an 
        overview of this research subarea.
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <i><small>One important caveat of this research: Our analysis involved converting 
        [user IP addresses](https://ip-api.com/) to real-world locations; especially for fine-grained city data, 
        IP addresses can be a slightly [noisy source](https://www.if-so.com/geo-targeting/) of geographic data. 
        We attempt to address this problem for cities by clustering data within a 
        [radius](https://github.com/camberg23/global-personality/blob/9a2dadbde2ab718fc3b18d0c621c1794580c9a84/geo_viz.py#L63) 
        that is larger than the typical margin of error typically associated with IP addresses.</small></i>
        """, 
        unsafe_allow_html=True
    )

st.write("---")
st.title("Interactive personality maps")
st.write("Make your settings choices and click submit to get an interactive map of the desired Big Five trait in the chosen geographical scope. Note: you can zoom into and out of the maps and mouseover areas for more information!")

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.75])

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
    trait = st.selectbox('Big Five Trait:', ['Choose an option', 'Display all traits'] + list(trait_names.values()))

with col4:
    score_type = st.selectbox("Score Type:", ["Choose an option", "Percentiles", "Normalized Scores"])

with col5:
    N = st.number_input('\uFF03 hi/lo:', min_value=0, max_value=50, value=5)

if st.button('Submit'):
    if trait == 'Display all traits':
        traits_to_display = list(trait_names.values())
    else:
        traits_to_display = [trait]

    for trait in traits_to_display:
        if us_or_global == 'US only' and trait != 'Choose an option' and state_or_city != 'Choose an option' and score_type != 'Choose an option':
            is_percentile = score_type == "Percentiles"
            if state_or_city == 'State view':
                scores = pd.read_csv('data/us_state_viz_improved.csv')
                if is_percentile:
                    scores = compute_percentiles_for_all(scores, trait_names)
                places = display_top_bottom_places(scores, trait, 'US states', 'State', N, score_type)
            elif state_or_city == 'City view':
                scores = pd.read_csv('data/us_city_viz_improved.csv')
                if is_percentile:
                    scores = compute_percentiles_for_all(scores, trait_names)
                places = display_top_bottom_places(scores, trait, 'US cities', 'City', N, score_type)
            
            with st.spinner("Generating a potential explanation of this ranking..."):
                explanation = generate_list_explanation(places, trait, score_type)
                st.write(explanation)
                    
            plot_us_trait_location(state_or_city, trait, scores, top_N=100, is_percentile=is_percentile)
    
        elif us_or_global == 'Global' and trait != 'Choose an option' and level != 'Choose an option' and score_type != 'Choose an option':
            is_percentile = score_type == "Percentiles"
            
            if level == "Country view":
                scores = pd.read_csv('data/country_data.csv')
                scores = scores[scores['Count'] > THRESHOLD_USERS]
                if is_percentile:
                    scores = compute_percentiles_for_all(scores, trait_names)
                places = display_top_bottom_places(scores, trait, 'countries', 'Country', N, score_type)
                
                with st.spinner("Generating a potential explanation of this ranking..."):
                    explanation = generate_list_explanation(places, trait, score_type)
                    st.write(explanation)

                plot_globe_trait_location(trait, level, scores, top_N=1000, is_percentile=is_percentile)

            elif level == "City view":
                scores = pd.read_csv('data/top_1000_city_data.csv')
                scores = scores[scores['Count'] > THRESHOLD_USERS]
                if is_percentile:
                    scores = compute_percentiles_for_all(scores, trait_names)
                places = display_top_bottom_places(scores, trait, 'cities', 'CityState', N, score_type)
                
                with st.spinner("Generating a potential explanation of this ranking..."):
                    explanation = generate_list_explanation(places, trait, score_type)
                    st.write(explanation)

                plot_globe_trait_location(trait, level, scores, top_N=1000, is_percentile=is_percentile)

# Create a section title and space
st.write("---")
st.write("---")
st.title("Personality profile of any location")
st.write("Get the average Big Five personality profiles of any location in our database.")

# Layout the top level containers
col1, col2 = st.columns((1, 1))  # Two columns of equal size

# User Input
with col1:
    comparison_type = st.radio("Choose the type of place:", ["Cities", "US States", "Countries"], key='profile')

is_button_pressed = False  # Initialize a flag to check if the button is pressed

with col2:
    if comparison_type == "Cities":
        data = pd.read_csv('data/top_1000_city_data.csv')
        city_options = data['CityState'] + ", " + data['Country']
        default_city_index = np.where(city_options == "New York, New York, United States")[0][0]
        selected = st.selectbox("Select the city:", city_options, key='profile_city', index=int(default_city_index))
        selected, _ = selected.rsplit(', ', 1)
    elif comparison_type == "Countries":
        data = pd.read_csv('data/country_data.csv')
        default_country_index = np.where(data['Country'] == "United States")[0][0]
        selected = st.selectbox("Select the country:", data['Country'].unique(), key='profile_country', index=int(default_country_index))
    else:  # Assuming comparison_type is "US States"
        data = pd.read_csv('data/us_state_viz_improved.csv')
        default_state_index = np.where(data['State'] == "California")[0][0]
        selected = st.selectbox("Select the US state:", data['State'].unique(), key='profile_state', index=int(default_state_index))

    # Place the Submit button in the second column, next to the selectbox
    is_button_pressed = st.button('Submit', key='profile_button')

# Generate profile outside of columns
if is_button_pressed:
    with st.spinner('Generating profile...'):
        display_percentile(comparison_type, selected, data)

# Create a section title and space
st.write("---")
st.write("---")
st.title("Population comparison tool")
st.write("Compare the average Big Five personality profiles of any two countries or cities.")
st.write("Note: there are almost always greater personality differences *within* a given location than *across* locations. Notice the large error bars (set score type to normalized scores), which signify significant trait diversity within each place.")

# Select comparison type: City vs. City or Country vs. Country
comparison_type = st.radio("Would you like to compare cities or countries?", ["Cities", "US States", "Countries"])
# Handle City vs. City comparison
if comparison_type == "Cities":
    st.header("City Comparison")
    
    city_scores = pd.read_csv('data/top_1000_city_data.csv')  

    city_options = city_scores['CityState'] + ", " + city_scores['Country']

    default_city1_index = np.where(city_options == "Los Angeles, California, United States")[0][0]
    default_city2_index = np.where(city_options == "Amsterdam, Netherlands")[0][0]

    col1, col2, col3 = st.columns(3)
    city1_selected = col1.selectbox("Select the first city:", city_options, index=int(default_city1_index))
    city2_selected = col2.selectbox("Select the second city:", city_options, index=int(default_city2_index))
    score_type = col3.selectbox("Score Type:", ["Percentiles", "Normalized Scores"], index=0)

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
                st.write(f"**Comparing {city1_selected} and {city2_selected}:** {comparison_paragraph}")


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
    score_type = col3.selectbox("Score Type:", ["Percentiles", "Normalized Scores"], index=0)

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
                st.write(f"**Comparing {country1_selected} and {country2_selected}:** {comparison_paragraph}")

# Handle State vs. State comparison
elif comparison_type == "US States":
    st.header("State Comparison")
    state_scores = pd.read_csv('data/us_state_viz_improved.csv')

    state_scores = state_scores[state_scores['Count'] > THRESHOLD_USERS]
    
    default_state1_index = np.where(state_scores['State'].unique() == "California")[0][0]
    default_state2_index = np.where(state_scores['State'].unique() == "Texas")[0][0]

    col1, col2, col3 = st.columns(3)
    state1_selected = col1.selectbox("Select the first state:", state_scores['State'].unique(), index=int(default_state1_index))
    state2_selected = col2.selectbox("Select the second state:", state_scores['State'].unique(), index=int(default_state2_index))
    score_type = col3.selectbox("Score Type:", ["Percentiles", "Normalized Scores"], index=0)

    state1_data = state_scores[state_scores['State'] == state1_selected].iloc[0]
    state2_data = state_scores[state_scores['State'] == state2_selected].iloc[0]
    
    percentiles1, percentiles2 = {}, {}
    if score_type == "Percentiles":
        percentiles1 = compute_percentile(state_scores, state1_data, trait_names)
        percentiles2 = compute_percentile(state_scores, state2_data, trait_names)
        state1_scores = list(percentiles1.values())
        state2_scores = list(percentiles2.values())
    else:
        state1_scores = [state1_data[trait] for trait in trait_names]
        state2_scores = [state2_data[trait] for trait in trait_names]

    state1_std = [state1_data[trait+'_std'] for trait in trait_names]
    state2_std = [state2_data[trait+'_std'] for trait in trait_names]

    state1_count = state_scores[state_scores['State'] == state1_selected]['Count'].values[0]
    state2_count = state_scores[state_scores['State'] == state2_selected]['Count'].values[0]

    if st.button('Submit', key='state_comparison_button'):
        with st.spinner('Generating comparison...'):
            plot_comparison(state1_scores, state2_scores, state1_std, state2_std, state1_selected, state2_selected, state1_count, state2_count, list(trait_names.values()), score_type, comparison_type.lower())
            if score_type == 'Percentiles':
                comparison_paragraph = generate_personality_comparison(state1_selected, state2_selected, percentiles1, percentiles2, trait_names, comparison_type)
                st.write(f"**Comparing {state1_selected} and {state2_selected}:** {comparison_paragraph}")
