import numpy as np
import pandas as pd
import streamlit as st

from geo_viz_utils import (
    trait_names,
    compute_percentile,
    compute_percentiles_for_all,
    display_top_bottom_places,
    display_percentile,
    generate_list_explanation,
    generate_personality_comparison,
    plot_comparison,
    plot_globe_trait_location,
    plot_us_trait_location,
)

THRESHOLD_USERS = 200

st.set_page_config(
    page_title="Personality Atlas · Truity",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Global CSS ----------
# Note: Streamlit's markdown parser closes <style> blocks at blank lines.
# The CSS below is intentionally written as one contiguous block.
_CSS = (
    "html, body, .stApp, .stMarkdown, p, label, h1, h2, h3, h4, h5, h6 {"
    " font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif !important; }"
    ".stApp { background: radial-gradient(1200px 600px at 10% -10%, "
    "rgba(61,90,128,0.07), transparent 60%), "
    "radial-gradient(900px 500px at 100% 0%, rgba(238,108,77,0.05), transparent 60%), "
    "#FBFBFD; }"
    ".block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 1320px; }"
    ".hero { display:flex; align-items:center; gap:22px; padding: 22px 26px; "
    "margin: 4px 0 22px; background: linear-gradient(135deg, rgba(61,90,128,0.08), "
    "rgba(238,108,77,0.07)); border: 1px solid #E5E7EB; border-radius: 18px; }"
    ".hero-wordmark { display:flex; flex-direction:column; align-items:center; "
    "justify-content:center; padding:14px 18px; border-radius:14px; background:#FFFFFF; "
    "border:1px solid #E5E7EB; box-shadow:0 1px 3px rgba(15,23,42,0.05); min-width:108px; }"
    ".hero-wordmark-text { font-size:1.55rem; font-weight:800; letter-spacing:0.02em; "
    "background: linear-gradient(135deg, #3D5A80 0%, #EE6C4D 100%); "
    "-webkit-background-clip: text; background-clip: text; "
    "-webkit-text-fill-color: transparent; color: transparent; line-height:1; }"
    ".hero-wordmark-tag { font-size:0.6rem; font-weight:600; color:#94A3B8; "
    "letter-spacing:0.18em; margin-top:4px; text-transform:uppercase; }"
    ".hero-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; "
    "background: linear-gradient(90deg, #293241 0%, #3D5A80 55%, #EE6C4D 100%); "
    "-webkit-background-clip: text; background-clip: text; "
    "-webkit-text-fill-color: transparent; color: transparent; line-height: 1.1; margin: 0; }"
    ".hero-sub { color:#475569; font-size: 1rem; margin-top: 6px; line-height: 1.5; }"
    ".hero-logo img { display:block; }"
    ".hero-badge { display:inline-block; font-size:0.72rem; font-weight:700; "
    "background:#FFFFFF; color:#3D5A80; border:1px solid #DBE3EE; "
    "padding: 4px 10px; border-radius: 999px; margin-bottom:8px; "
    "letter-spacing:0.06em; text-transform:uppercase; }"
    ".section-title { font-size: 1.5rem; font-weight: 700; color: #293241; "
    "letter-spacing: -0.01em; margin: 4px 0 4px; }"
    ".section-sub { color:#64748B; font-size: 0.98rem; line-height: 1.55; margin-bottom: 14px; }"
    ".card { background:#FFFFFF; border:1px solid #E5E7EB; border-radius: 14px; "
    "padding: 18px 20px; box-shadow: 0 1px 3px rgba(15,23,42,0.05); margin-bottom: 14px; }"
    ".stTabs [data-baseweb=\"tab-list\"] { gap: 4px; background: #F1F5F9; "
    "padding: 6px; border-radius: 12px; border: 1px solid #E5E7EB; }"
    ".stTabs [data-baseweb=\"tab\"] { height: 42px; padding: 0 20px; "
    "border-radius: 8px; color: #475569; font-weight: 600; background: transparent; }"
    ".stTabs [aria-selected=\"true\"] { background: #FFFFFF !important; "
    "color: #293241 !important; box-shadow: 0 1px 3px rgba(15,23,42,0.08); }"
    ".stTabs [data-baseweb=\"tab-highlight\"] { display:none !important; }"
    ".stTabs [data-baseweb=\"tab-border\"] { display:none !important; }"
    ".stButton > button { background: #3D5A80; color: white; border: none; "
    "border-radius: 10px; padding: 8px 22px; font-weight: 600; "
    "transition: transform 0.05s ease, background 0.15s ease; }"
    ".stButton > button:hover { background: #2E4763; color: white; "
    "transform: translateY(-1px); }"
    ".stButton > button:focus { box-shadow: 0 0 0 3px rgba(61,90,128,0.25) !important; }"
    ".stDownloadButton > button { border-radius: 10px; }"
    ".stSelectbox label, .stRadio label, .stNumberInput label { "
    "font-weight: 600 !important; color: #334155 !important; font-size: 0.9rem !important; }"
    "div[data-baseweb=\"select\"] > div { border-radius: 10px !important; "
    "border-color: #E5E7EB !important; }"
    "[data-testid=\"stExpander\"] { background:#FFFFFF; border:1px solid #E5E7EB; "
    "border-radius:12px; }"
    "[data-testid=\"stExpander\"] details summary { font-weight: 600 !important; "
    "color: #293241 !important; }"
    ".stSpinner > div > div { border-top-color: #3D5A80 !important; }"
    ".soft-divider { height:1px; background: linear-gradient(90deg, transparent, "
    "#E5E7EB, transparent); margin: 18px 0; }"
    ".footer { text-align:center; color:#94A3B8; font-size:0.88rem; margin-top:28px; "
    "padding: 16px 0; border-top: 1px solid #EEF2F7; }"
    ".footer a { color:#3D5A80; text-decoration:none; font-weight:500; }"
    ".footer a:hover { text-decoration:underline; }"
    ".rank-card { display:flex; align-items:flex-start; gap:12px; padding:10px 12px; "
    "border-radius:10px; background:#FFFFFF; border:1px solid #E5E7EB; "
    "margin-bottom:8px; box-shadow:0 1px 2px rgba(15,23,42,0.04); }"
    ".rank-chip { flex:0 0 28px; height:28px; border-radius:7px; color:white; "
    "font-weight:700; display:flex; align-items:center; justify-content:center; "
    "font-size:13px; }"
    ".rank-name { font-weight:600; color:#293241; font-size:15px; }"
    ".rank-meta { color:#64748B; font-size:13px; margin-top:2px; }"
)
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

# ---------- Hero ----------
_HERO_HTML = (
    '<div class="hero">'
    '<a href="https://www.truity.com/" target="_blank" rel="noopener" '
    'style="text-decoration:none;">'
    '<div class="hero-wordmark">'
    '<div class="hero-wordmark-text">Truity</div>'
    '<div class="hero-wordmark-tag">Personality</div>'
    '</div></a>'
    '<div style="flex:1;">'
    '<div class="hero-badge">Powered by Truity · 4M+ respondents</div>'
    '<div class="hero-title">Big Five Personality Atlas</div>'
    '<div class="hero-sub">Explore how the five core personality traits vary across countries, '
    'US states, and cities — built on one of the largest open personality datasets in the world.</div>'
    '</div></div>'
)
st.markdown(_HERO_HTML, unsafe_allow_html=True)

with st.expander("About this tool & a note on the data"):
    st.markdown(
        """
Welcome to Truity's **Big Five Personality Atlas**. This page brings together three views of how the
[Big Five](https://en.wikipedia.org/wiki/Big_Five_personality_traits) personality traits — *Openness*,
*Conscientiousness*, *Extraversion*, *Agreeableness*, and *Neuroticism* — vary around the world, built on Truity's
4M-person database.

- **Interactive maps** for the US (by state or city) and the world (by country or city).
- A **profile generator** for any location in the database.
- A **head-to-head comparison** tool for any two places.

You can take [Truity's validated Big Five assessment here](https://www.truity.com/test/big-five-personality-test).
The code and data are [publicly available](https://github.com/camberg23/global-personality).

Longstanding evidence suggests that the Big Five is a valid measure of personality cross-culturally
([Allik & McCrae, 1998](https://journals.sagepub.com/doi/10.1177/0022022198291009);
[Schmitt et al., 2007](https://journals.sagepub.com/doi/abs/10.1177/0022022106297299)).

> **A note on geography.** Locations are inferred from [user IP addresses](https://ip-api.com/), which can be
> a [noisy source](https://www.if-so.com/geo-targeting/) of fine-grained data. To mitigate this for cities, we
> cluster nearby points within a radius that exceeds typical IP geolocation error.
        """,
        unsafe_allow_html=True,
    )

tab_map, tab_profile, tab_compare = st.tabs([
    "🗺️  Interactive Map",
    "📊  Location Profile",
    "⚖️  Compare Places",
])


# ==================================================================
# TAB 1 — Interactive map
# ==================================================================
with tab_map:
    st.markdown('<div class="section-title">Interactive personality maps</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Configure the view below and hit <b>Generate map</b>. '
        'Maps are fully interactive — zoom, pan, and hover for details.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col1, col2, col3, col4, col5, col6 = st.columns([1.1, 1.1, 1.3, 1.2, 0.7, 0.9])

        with col1:
            us_or_global = st.selectbox('Region', ['Choose…', 'US only', 'Global'], key='map_region')
        with col2:
            if us_or_global == 'US only':
                scope_choice = st.selectbox('US scope', ['Choose…', 'State view', 'City view'], key='map_us_scope')
            elif us_or_global == 'Global':
                scope_choice = st.selectbox('Global scope', ['Choose…', 'Country view', 'City view'], key='map_global_scope')
            else:
                scope_choice = st.selectbox('Scope', ['Choose region first'], disabled=True, key='map_scope_dis')
        with col3:
            trait = st.selectbox(
                'Big Five trait',
                ['Choose…', 'Display all traits'] + list(trait_names.values()),
                key='map_trait',
            )
        with col4:
            score_type = st.selectbox(
                'Score type',
                ['Choose…', 'Percentiles', 'Normalized Scores'],
                key='map_score_type',
            )
        with col5:
            N = st.number_input('# hi/lo', min_value=0, max_value=50, value=5, key='map_n')
        with col6:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            submit_map = st.button('Generate map', key='map_submit', use_container_width=True)

    if submit_map:
        ready = (
            us_or_global in ('US only', 'Global')
            and scope_choice not in ('Choose…', 'Choose region first')
            and trait != 'Choose…'
            and score_type != 'Choose…'
        )
        if not ready:
            st.warning("Pick a region, scope, trait, and score type to generate a map.")
        else:
            traits_to_display = list(trait_names.values()) if trait == 'Display all traits' else [trait]
            is_percentile = score_type == "Percentiles"

            for current_trait in traits_to_display:
                st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

                if us_or_global == 'US only':
                    if scope_choice == 'State view':
                        top_N = 51
                        scores = pd.read_csv('data/us_state_viz_improved.csv')
                        if is_percentile:
                            scores = compute_percentiles_for_all(scores, trait_names)
                        places = display_top_bottom_places(scores, current_trait, 'US states', 'State', N, score_type)
                    else:  # City view
                        top_N = 60
                        scores = pd.read_csv('data/us_city_viz_improved.csv')
                        scores = scores.nlargest(top_N, 'Count')
                        if is_percentile:
                            scores = compute_percentiles_for_all(scores, trait_names)
                        places = display_top_bottom_places(scores, current_trait, 'US cities', 'City', N, score_type)

                    with st.spinner("Generating a possible explanation of this ranking…"):
                        explanation = generate_list_explanation(places, current_trait, score_type)
                    st.markdown(
                        f"<div class='card' style='margin-top:6px;'><b>Possible explanation</b><br>"
                        f"<span style='color:#334155;line-height:1.55;'>{explanation}</span></div>",
                        unsafe_allow_html=True,
                    )

                    plot_us_trait_location(scope_choice, current_trait, scores,
                                           top_N=top_N, is_percentile=is_percentile)

                else:  # Global
                    if scope_choice == 'Country view':
                        scores = pd.read_csv('data/country_data.csv')
                        scores = scores[scores['Count'] > THRESHOLD_USERS]
                        if is_percentile:
                            scores = compute_percentiles_for_all(scores, trait_names)
                        places = display_top_bottom_places(scores, current_trait, 'countries', 'Country', N, score_type)
                    else:  # City view
                        scores = pd.read_csv('data/top_1000_city_data.csv')
                        scores = scores[scores['Count'] > THRESHOLD_USERS]
                        if is_percentile:
                            scores = compute_percentiles_for_all(scores, trait_names)
                        places = display_top_bottom_places(scores, current_trait, 'cities', 'CityState', N, score_type)

                    with st.spinner("Generating a possible explanation of this ranking…"):
                        explanation = generate_list_explanation(places, current_trait, score_type)
                    st.markdown(
                        f"<div class='card' style='margin-top:6px;'><b>Possible explanation</b><br>"
                        f"<span style='color:#334155;line-height:1.55;'>{explanation}</span></div>",
                        unsafe_allow_html=True,
                    )

                    plot_globe_trait_location(current_trait, scope_choice, scores,
                                              top_N=1000, is_percentile=is_percentile)


# ==================================================================
# TAB 2 — Location profile
# ==================================================================
with tab_profile:
    st.markdown('<div class="section-title">Personality profile for any location</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Pick a place to see its average Big Five percentile profile, '
        'with an AI-written narrative.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col_a, col_b, col_c = st.columns([1, 1.4, 0.7])

        with col_a:
            profile_type = st.radio(
                "Place type",
                ["Global Cities", "US Cities", "US States", "Countries"],
                key='profile_type',
                horizontal=False,
            )

        with col_b:
            if profile_type == "Global Cities":
                data = pd.read_csv('data/top_1000_city_data.csv')
                city_options = data['CityState'] + ", " + data['Country']
                default_idx = int(np.where(city_options == "New York, New York, United States")[0][0])
                selected_display = st.selectbox("Select a city", city_options,
                                                key='profile_city', index=default_idx)
                selected_profile, _ = selected_display.rsplit(', ', 1)
            elif profile_type == "US Cities":
                data = pd.read_csv('data/us_city_viz_improved.csv')
                city_options = data['City']
                default_idx = int(np.where(city_options == "New York, New York")[0][0])
                selected_profile = st.selectbox("Select a US city", city_options,
                                                key='profile_us_city', index=default_idx)
            elif profile_type == "Countries":
                data = pd.read_csv('data/country_data.csv')
                default_idx = int(np.where(data['Country'] == "United States")[0][0])
                selected_profile = st.selectbox("Select a country", data['Country'].unique(),
                                                key='profile_country', index=default_idx)
            else:  # US States
                data = pd.read_csv('data/us_state_viz_improved.csv')
                default_idx = int(np.where(data['State'] == "California")[0][0])
                selected_profile = st.selectbox("Select a US state", data['State'].unique(),
                                                key='profile_state', index=default_idx)

        with col_c:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            profile_submit = st.button("Generate profile", key='profile_button', use_container_width=True)

    if profile_submit:
        with st.spinner('Building personality profile…'):
            display_percentile(profile_type, selected_profile, data)


# ==================================================================
# TAB 3 — Compare
# ==================================================================
with tab_compare:
    st.markdown('<div class="section-title">Head-to-head comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Compare two places side-by-side across the Big Five. '
        'Note: with normalized scores, the visible error bars often show that personality variation '
        '<i>within</i> a place is larger than the difference <i>between</i> places.</div>',
        unsafe_allow_html=True,
    )

    compare_type = st.radio(
        "Compare places of type:",
        ["Global Cities", "US Cities", "US States", "Countries"],
        key='compare_type',
        horizontal=True,
    )

    if compare_type == "Global Cities":
        city_scores = pd.read_csv('data/top_1000_city_data.csv')
        city_options = city_scores['CityState'] + ", " + city_scores['Country']
        d1 = int(np.where(city_options == "Los Angeles, California, United States")[0][0])
        d2 = int(np.where(city_options == "Amsterdam, Netherlands")[0][0])

        c1, c2, c3 = st.columns(3)
        sel1 = c1.selectbox("First city", city_options, index=d1, key='gc_1')
        sel2 = c2.selectbox("Second city", city_options, index=d2, key='gc_2')
        cmp_score_type = c3.selectbox("Score type", ["Percentiles", "Normalized Scores"], key='gc_st')

        cs1, cc1 = sel1.rsplit(', ', 1)
        cs2, cc2 = sel2.rsplit(', ', 1)
        d_a = city_scores[(city_scores['CityState'] == cs1) & (city_scores['Country'] == cc1)].iloc[0]
        d_b = city_scores[(city_scores['CityState'] == cs2) & (city_scores['Country'] == cc2)].iloc[0]

        pct1, pct2 = {}, {}
        if cmp_score_type == "Percentiles":
            pct1 = compute_percentile(city_scores, d_a, trait_names)
            pct2 = compute_percentile(city_scores, d_b, trait_names)
            scores1, scores2 = list(pct1.values()), list(pct2.values())
        else:
            scores1 = [d_a[t] for t in trait_names]
            scores2 = [d_b[t] for t in trait_names]

        std1 = [d_a[t + '_std'] for t in trait_names]
        std2 = [d_b[t + '_std'] for t in trait_names]
        n1 = int(d_a['Count'])
        n2 = int(d_b['Count'])
        label1, label2 = sel1, sel2

    elif compare_type == "Countries":
        country_scores = pd.read_csv('data/country_data.csv')
        country_scores = country_scores[country_scores['Count'] > THRESHOLD_USERS]
        countries = country_scores['Country'].unique()
        d1 = int(np.where(countries == "United States")[0][0])
        d2 = int(np.where(countries == "Russia")[0][0])

        c1, c2, c3 = st.columns(3)
        sel1 = c1.selectbox("First country", countries, index=d1, key='cc_1')
        sel2 = c2.selectbox("Second country", countries, index=d2, key='cc_2')
        cmp_score_type = c3.selectbox("Score type", ["Percentiles", "Normalized Scores"], key='cc_st')

        d_a = country_scores[country_scores['Country'] == sel1].iloc[0]
        d_b = country_scores[country_scores['Country'] == sel2].iloc[0]

        pct1, pct2 = {}, {}
        if cmp_score_type == "Percentiles":
            pct1 = compute_percentile(country_scores, d_a, trait_names)
            pct2 = compute_percentile(country_scores, d_b, trait_names)
            scores1, scores2 = list(pct1.values()), list(pct2.values())
        else:
            scores1 = [d_a[t] for t in trait_names]
            scores2 = [d_b[t] for t in trait_names]

        std1 = [d_a[t + '_std'] for t in trait_names]
        std2 = [d_b[t + '_std'] for t in trait_names]
        n1 = int(d_a['Count'])
        n2 = int(d_b['Count'])
        label1, label2 = sel1, sel2

    elif compare_type == "US States":
        state_scores = pd.read_csv('data/us_state_viz_improved.csv')
        state_scores = state_scores[state_scores['Count'] > THRESHOLD_USERS]
        states = state_scores['State'].unique()
        d1 = int(np.where(states == "California")[0][0])
        d2 = int(np.where(states == "Texas")[0][0])

        c1, c2, c3 = st.columns(3)
        sel1 = c1.selectbox("First state", states, index=d1, key='us_1')
        sel2 = c2.selectbox("Second state", states, index=d2, key='us_2')
        cmp_score_type = c3.selectbox("Score type", ["Percentiles", "Normalized Scores"], key='us_st')

        d_a = state_scores[state_scores['State'] == sel1].iloc[0]
        d_b = state_scores[state_scores['State'] == sel2].iloc[0]

        pct1, pct2 = {}, {}
        if cmp_score_type == "Percentiles":
            pct1 = compute_percentile(state_scores, d_a, trait_names)
            pct2 = compute_percentile(state_scores, d_b, trait_names)
            scores1, scores2 = list(pct1.values()), list(pct2.values())
        else:
            scores1 = [d_a[t] for t in trait_names]
            scores2 = [d_b[t] for t in trait_names]

        std1 = [d_a[t + '_std'] for t in trait_names]
        std2 = [d_b[t + '_std'] for t in trait_names]
        n1 = int(d_a['Count'])
        n2 = int(d_b['Count'])
        label1, label2 = sel1, sel2

    else:  # US Cities
        us_city_scores = pd.read_csv('data/us_city_viz_improved.csv')
        opts = us_city_scores['City']
        d1 = int(np.where(opts == "New York, New York")[0][0])
        d2 = int(np.where(opts == "Los Angeles, California")[0][0])

        c1, c2, c3 = st.columns(3)
        sel1 = c1.selectbox("First US city", opts, index=d1, key='usc_1')
        sel2 = c2.selectbox("Second US city", opts, index=d2, key='usc_2')
        cmp_score_type = c3.selectbox("Score type", ["Percentiles", "Normalized Scores"], key='usc_st')

        d_a = us_city_scores[us_city_scores['City'] == sel1].iloc[0]
        d_b = us_city_scores[us_city_scores['City'] == sel2].iloc[0]

        pct1, pct2 = {}, {}
        if cmp_score_type == "Percentiles":
            pct1 = compute_percentile(us_city_scores, d_a, trait_names)
            pct2 = compute_percentile(us_city_scores, d_b, trait_names)
            scores1, scores2 = list(pct1.values()), list(pct2.values())
        else:
            scores1 = [d_a[t] for t in trait_names]
            scores2 = [d_b[t] for t in trait_names]

        std1 = [d_a[t + '_std'] for t in trait_names]
        std2 = [d_b[t + '_std'] for t in trait_names]
        n1 = int(d_a['Count'])
        n2 = int(d_b['Count'])
        label1, label2 = sel1, sel2

    compare_submit = st.button('Compare', key='compare_button')

    if compare_submit:
        with st.spinner('Generating comparison…'):
            plot_comparison(
                scores1, scores2, std1, std2, label1, label2, n1, n2,
                list(trait_names.values()), cmp_score_type, compare_type.lower(),
            )
            if cmp_score_type == 'Percentiles':
                narrative = generate_personality_comparison(
                    label1, label2, pct1, pct2, trait_names, compare_type,
                )
                st.markdown(
                    f'<div class="card" style="margin-top:8px;">'
                    f'<div style="font-weight:600;color:#293241;margin-bottom:6px;">'
                    f'Comparing {label1} and {label2}</div>'
                    f'<div style="color:#334155;line-height:1.6;">{narrative}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ---------- Footer ----------
st.markdown(
    '<div class="footer">'
    'Built by <a href="https://www.truity.com/" target="_blank" rel="noopener">Truity</a> · '
    '<a href="https://www.truity.com/test/big-five-personality-test" target="_blank" rel="noopener">Take the Big Five test</a> · '
    '<a href="https://github.com/camberg23/global-personality" target="_blank" rel="noopener">Source on GitHub</a>'
    '</div>',
    unsafe_allow_html=True,
)
