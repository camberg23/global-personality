import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from sklearn.cluster import DBSCAN

# ---------- Constants ----------

trait_names = {
    'o': 'Openness',
    'c': 'Conscientiousness',
    'e': 'Extraversion',
    'a': 'Agreeableness',
    'n': 'Neuroticism',
}

traits = list(trait_names.keys())

TRAIT_ADJECTIVES = {
    'Openness': 'open',
    'Conscientiousness': 'conscientious',
    'Extraversion': 'extraverted',
    'Agreeableness': 'agreeable',
    'Neuroticism': 'neurotic',
}

# Refined visual palette
PRIMARY = "#3D5A80"          # deep slate blue
ACCENT = "#EE6C4D"           # warm coral
NEUTRAL_DARK = "#293241"
NEUTRAL_LIGHT = "#E0FBFC"
BAR_COLOR_A = "#3D5A80"
BAR_COLOR_B = "#EE6C4D"

CONTINUOUS_SCALE = "Viridis"

PLOTLY_FONT = dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                   size=14, color=NEUTRAL_DARK)


def _base_layout(title_text, height=640):
    return dict(
        title=dict(
            text=f"<b>{title_text}</b>",
            x=0.5, xanchor='center', y=0.97,
            font=dict(size=22, color=NEUTRAL_DARK,
                      family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        ),
        height=height,
        margin=dict(l=10, r=10, t=80, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        coloraxis_colorbar=dict(
            lenmode="fraction", len=0.7,
            yanchor="bottom", y=-0.12,
            xanchor="center", x=0.5,
            orientation="h",
            thickness=14,
            outlinewidth=0,
            tickfont=dict(size=12, color=NEUTRAL_DARK),
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=PRIMARY,
            font=dict(family=PLOTLY_FONT['family'], size=13, color=NEUTRAL_DARK),
        ),
    )


# ---------- OpenAI client ----------

@st.cache_resource(show_spinner=False)
def _openai_client():
    """Lazily instantiate the OpenAI client using Streamlit secrets."""
    org = st.secrets.get("ORG") if hasattr(st, "secrets") else None
    key = st.secrets.get("KEY") if hasattr(st, "secrets") else None
    kwargs = {}
    if key:
        kwargs["api_key"] = key
    if org:
        kwargs["organization"] = org
    return OpenAI(**kwargs)


LLM_MODEL = "gpt-5-nano"


def _chat(messages, max_tokens=500):
    """Wrap chat completion with graceful fallback so the UI never crashes."""
    try:
        client = _openai_client()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"_AI narrative unavailable right now ({type(e).__name__})._"


# ---------- Maps ----------

def plot_globe_trait_location(trait, level, scores, top_N=1000, is_percentile=False):
    data = scores.copy()
    inv_trait_names = {v: k for k, v in trait_names.items()}
    trait_abbrev = inv_trait_names[trait]
    data[trait] = data[trait_abbrev]
    full_trait_name = trait

    if level == "Country view":
        if is_percentile:
            hover_template = (
                f"<b>%{{hovertext}}</b><br>"
                f"Percentile {trait}: %{{customdata[1]:.1f}}<br>"
                "Sample size: %{customdata[0]:,}"
            )
            hover_data = {"Count": True, trait: f"Percentile {trait}"}
        else:
            hover_template = (
                f"<b>%{{hovertext}}</b><br>"
                f"Mean {trait}: %{{customdata[1]:.3f}}<br>"
                "Std dev: %{customdata[2]:.3f}<br>"
                "Sample size: %{customdata[0]:,}"
            )
            hover_data = {
                "Count": True,
                trait: f"Mean {trait}",
                f"{trait_abbrev}_std": f"Std Dev {trait}",
            }

        fig = px.choropleth(
            data,
            locations="Country",
            locationmode="country names",
            color=full_trait_name,
            hover_name="Country",
            hover_data=hover_data,
            color_continuous_scale=CONTINUOUS_SCALE,
        )
        fig.update_traces(hovertemplate=hover_template, marker_line_color="white",
                          marker_line_width=0.4)
        fig.update_geos(
            showcoastlines=True, coastlinecolor="#CBD5E1",
            showland=True, landcolor="#F8FAFC",
            showocean=True, oceancolor="#EFF6FF",
            showframe=False,
            projection_type="natural earth",
        )
        fig.update_layout(**_base_layout(f"{full_trait_name} by Country", height=640))
        st.plotly_chart(fig, use_container_width=True)
        return

    # ---- City view ----
    city_counts = data.groupby('CityState').agg({'Count': 'sum'}).reset_index()
    top_cities = city_counts.nlargest(top_N, 'Count')['CityState']
    data = data[data['CityState'].isin(top_cities)]

    kms_per_radian = 6371.0088
    epsilon = 50 / kms_per_radian
    coords_global = data[['Latitude', 'Longitude']].values
    db_global = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree',
                       metric='haversine').fit(np.radians(coords_global))
    data['Cluster'] = db_global.labels_

    clustered_data = data[data['Cluster'] != -1]
    cluster_agg = clustered_data.groupby('Cluster').agg({
        'Latitude': 'mean',
        'Longitude': 'mean',
        'Count': 'sum',
        'CityState': lambda x: x.value_counts().idxmax(),
        'Country': lambda x: x.value_counts().idxmax(),
    }).reset_index()

    city_scores_global = (
        data.groupby(['CityState', 'Country'])
        .mean(numeric_only=True)
        .reset_index()
    )
    clustered = cluster_agg.merge(city_scores_global, on=['CityState', 'Country'], how='left')
    clustered.rename(columns={"Count_x": "Count", "Latitude_x": "Latitude",
                              "Longitude_x": "Longitude"}, inplace=True)

    if is_percentile:
        hover_template = (
            "<b>%{hovertext}, %{customdata[2]}</b><br>"
            f"Percentile {trait}: " + "%{customdata[0]:.1f}<br>"
            "Sample size: %{customdata[1]:,}"
        )
    else:
        hover_template = (
            "<b>%{hovertext}, %{customdata[2]}</b><br>"
            f"Avg {trait}: " + "%{customdata[0]:.3f}<br>"
            f"Std dev: " + "%{customdata[3]:.3f}<br>"
            "Sample size: %{customdata[1]:,}"
        )

    fig = px.scatter_geo(
        clustered,
        lat='Latitude',
        lon='Longitude',
        color=full_trait_name,
        hover_name='CityState',
        hover_data={trait: True, 'Count': True, 'Country': True,
                    f"{trait_abbrev}_std": f"Std Dev {full_trait_name}"},
        color_continuous_scale=CONTINUOUS_SCALE,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.6, color="white")),
                      hovertemplate=hover_template)
    fig.update_geos(
        showcountries=True, countrywidth=0.4, countrycolor="#94A3B8",
        showcoastlines=True, coastlinecolor="#CBD5E1",
        showland=True, landcolor="#F8FAFC",
        showocean=True, oceancolor="#EFF6FF",
        showframe=False,
        projection_type="natural earth",
    )
    fig.update_layout(**_base_layout(f"{full_trait_name} across major cities", height=640))
    st.plotly_chart(fig, use_container_width=True)


def plot_us_trait_location(state_or_city, trait, scores, top_N=100, is_percentile=False):
    inv_trait_names = {v: k for k, v in trait_names.items()}
    trait_abbrev = inv_trait_names[trait]

    if state_or_city == 'State view':
        data = scores.copy()
        full_trait_name = trait
        data[full_trait_name] = data[trait_abbrev]

        fig = px.choropleth(
            data,
            locations="State_Abbrev",
            locationmode="USA-states",
            color=full_trait_name,
            hover_name="State",
            hover_data=[trait_abbrev, 'Count', trait_abbrev + "_std"],
            color_continuous_scale=CONTINUOUS_SCALE,
            scope="usa",
        )
        if is_percentile:
            hovertemplate = (
                "<b>%{hovertext}</b><br>"
                f"Percentile {full_trait_name}: " + "%{customdata[0]:.1f}<br>"
                "Sample size: %{customdata[1]:,}"
            )
        else:
            hovertemplate = (
                "<b>%{hovertext}</b><br>"
                f"Mean {full_trait_name}: " + "%{customdata[0]:.3f}<br>"
                "Std dev: %{customdata[2]:.3f}<br>"
                "Sample size: %{customdata[1]:,}"
            )
        fig.update_traces(hovertemplate=hovertemplate, marker_line_color="white",
                          marker_line_width=0.5)
        fig.update_geos(showlakes=True, lakecolor="#EFF6FF",
                        landcolor="#F8FAFC", bgcolor="rgba(0,0,0,0)")
        fig.update_layout(**_base_layout(f"{full_trait_name} by US State", height=640))
        st.plotly_chart(fig, use_container_width=True)
        return

    # ---- US City view ----
    data = scores.copy()
    data[trait] = data[trait_abbrev]
    data = data.nlargest(top_N, 'Count')

    fig = px.scatter_geo(
        data,
        locationmode='USA-states',
        scope='usa',
        lat='Latitude',
        lon='Longitude',
        size='Count',
        color=trait,
        hover_name='City',
        custom_data=[trait, 'Count', f"{trait_abbrev}_std"],
        color_continuous_scale=CONTINUOUS_SCALE,
        size_max=55,
    )

    if is_percentile:
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            f"Percentile {trait}: " + "%{customdata[0]:.1f}<br>"
            "Sample size: %{customdata[1]:,}"
        )
    else:
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            f"Mean {trait}: " + "%{customdata[0]:.3f}<br>"
            "Std dev: %{customdata[2]:.3f}<br>"
            "Sample size: %{customdata[1]:,}"
        )

    fig.update_traces(hovertemplate=hovertemplate,
                      marker=dict(line=dict(width=0.6, color="white")))
    fig.update_geos(center=dict(lat=38.0902, lon=-95.7129),
                    showlakes=True, lakecolor="#EFF6FF",
                    landcolor="#F8FAFC", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(**_base_layout(f"{trait} across major US cities", height=640))
    st.plotly_chart(fig, use_container_width=True)


# ---------- Top/bottom rankings ----------

def _ranking_row_html(idx, place_name, value_line, accent):
    # Single-line HTML so Streamlit's markdown parser does not break the block.
    return (
        f'<div class="rank-card">'
        f'<div class="rank-chip" style="background:{accent};">{idx}</div>'
        f'<div style="flex:1;">'
        f'<div class="rank-name">{place_name}</div>'
        f'<div class="rank-meta">{value_line}</div>'
        f'</div></div>'
    )


def display_top_bottom_places(data, trait, scope, place_column, N=5, score_type="Normalized Scores"):
    inv_trait_names = {v: k for k, v in trait_names.items()}
    full_name = trait
    trait_key = inv_trait_names[trait]
    description = TRAIT_ADJECTIVES[full_name]

    if place_column == "CityState":
        text_scope = "largest 1000 cities in the world"
    elif place_column == "City":
        text_scope = "largest 60 cities in the US"
    else:
        text_scope = scope

    top_places = data.sort_values(by=trait_key, ascending=False).head(N)
    bottom_places = data.sort_values(by=trait_key, ascending=True).head(N)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<div style='font-size:1.05em;font-weight:600;color:{NEUTRAL_DARK};"
            f"margin:4px 0 10px;'>↑ Highest {N} {scope} in {full_name}</div>",
            unsafe_allow_html=True,
        )
        for idx, (_, row) in enumerate(top_places.iterrows()):
            place_name = row[place_column]
            if 'Country' in data.columns and scope == 'cities':
                country_name = 'US' if row['Country'] == 'United States' else row['Country']
                place_name += f", {country_name}"
            if score_type == "Percentiles":
                value_line = (f"More {description} than {row[trait_key]:.1f}% "
                              f"of {text_scope} · n={int(row['Count']):,}")
            else:
                value_line = (f"{row[trait_key]:.2f} ± {row[trait_key + '_std']:.2f} "
                              f"· n={int(row['Count']):,}")
            st.markdown(_ranking_row_html(idx + 1, place_name, value_line, PRIMARY),
                        unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"<div style='font-size:1.05em;font-weight:600;color:{NEUTRAL_DARK};"
            f"margin:4px 0 10px;'>↓ Lowest {N} {scope} in {full_name}</div>",
            unsafe_allow_html=True,
        )
        for idx, (_, row) in enumerate(bottom_places.iterrows()):
            place_name = row[place_column]
            if 'Country' in data.columns and scope == 'cities':
                country_name = 'US' if row['Country'] == 'United States' else row['Country']
                place_name += f", {country_name}"
            if score_type == "Percentiles":
                value_line = (f"Less {description} than {100 - row[trait_key]:.1f}% "
                              f"of {text_scope} · n={int(row['Count']):,}")
            else:
                value_line = (f"{row[trait_key]:.2f} ± {row[trait_key + '_std']:.2f} "
                              f"· n={int(row['Count']):,}")
            st.markdown(_ranking_row_html(idx + 1, place_name, value_line, ACCENT),
                        unsafe_allow_html=True)

    return {
        'top': [row[place_column] for _, row in top_places.iterrows()],
        'bottom': [row[place_column] for _, row in bottom_places.iterrows()],
    }


# ---------- Comparison plot ----------

def plot_comparison(scores1, scores2, std1, std2, label1, label2, count1, count2,
                    traits_full, score_type, comparison_type):
    y_values_plot_1 = [max(s, 1) for s in scores1]
    y_values_plot_2 = [max(s, 1) for s in scores2]

    error_visible = score_type == "Normalized Scores"

    if score_type == "Percentiles":
        ht1 = f"<b>{label1}</b><br>%{{x}}: %{{customdata:.1f}} percentile<extra></extra>"
        ht2 = f"<b>{label2}</b><br>%{{x}}: %{{customdata:.1f}} percentile<extra></extra>"
        yaxis_title = "Percentile"
    else:
        ht1 = f"<b>{label1}</b><br>%{{x}}: %{{customdata:.3f}}<extra></extra>"
        ht2 = f"<b>{label2}</b><br>%{{x}}: %{{customdata:.3f}}<extra></extra>"
        yaxis_title = "Normalized score"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=traits_full, y=y_values_plot_1, customdata=scores1,
        name=f"{label1} (n={count1:,})",
        error_y=dict(type='data', array=std1, visible=error_visible,
                     color="rgba(61,90,128,0.5)", thickness=1.5),
        marker=dict(color=BAR_COLOR_A, line=dict(width=0)),
        hovertemplate=ht1,
    ))
    fig.add_trace(go.Bar(
        x=traits_full, y=y_values_plot_2, customdata=scores2,
        name=f"{label2} (n={count2:,})",
        error_y=dict(type='data', array=std2, visible=error_visible,
                     color="rgba(238,108,77,0.5)", thickness=1.5),
        marker=dict(color=BAR_COLOR_B, line=dict(width=0)),
        hovertemplate=ht2,
    ))

    fig.update_layout(
        title=dict(
            text=(f"<b>{label1}</b>  <span style='color:#94A3B8'>vs</span>  "
                  f"<b style='color:{ACCENT}'>{label2}</b>"),
            x=0.5, xanchor='center', y=0.95,
            font=dict(size=22, family=PLOTLY_FONT['family'], color=NEUTRAL_DARK),
        ),
        xaxis=dict(title="", showgrid=False, tickfont=dict(size=14)),
        yaxis=dict(title=yaxis_title, gridcolor="#EEF2F7", zerolinecolor="#E5E7EB"),
        barmode='group',
        bargap=0.25,
        bargroupgap=0.08,
        legend=dict(yanchor="top", y=1.18, xanchor="center", x=0.5,
                    orientation="h", bgcolor="rgba(0,0,0,0)"),
        font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=120, b=40, l=20, r=20),
        height=520,
        hoverlabel=dict(bgcolor="white", bordercolor=PRIMARY,
                        font=dict(family=PLOTLY_FONT['family'], size=13, color=NEUTRAL_DARK)),
    )
    # color the first label in title
    fig.update_layout(title_text=(
        f"<b style='color:{PRIMARY}'>{label1}</b>  "
        f"<span style='color:#94A3B8'>vs</span>  "
        f"<b style='color:{ACCENT}'>{label2}</b>"
    ))

    st.plotly_chart(fig, use_container_width=True)


# ---------- LLM narratives ----------

def generate_personality_description(selected, percentiles, trait_names_map):
    system_message = (
        "You are a thoughtful, courteous assistant that summarizes a location's "
        "overall personality blend based on Big Five percentiles. Where useful, "
        "blend in concrete things you know about the place to make a holistic, "
        "harmonious profile. Always use relative language (the data is "
        "percentile-based). Be sensitive: there are no 'good' or 'bad' personalities, "
        "though high openness/conscientiousness/agreeableness/extraversion and low "
        "neuroticism are typically framed as desirable. "
        "Limit your reply to ONE strong paragraph. "
        "Then, on a new line, write exactly one interesting fact about the location, "
        "formatted as: 'Bonus information about [location]:'"
    )
    user_messages = [
        f"{selected} is in the {percentiles[trait]} percentile in the world for {trait_full}."
        for trait, trait_full in trait_names_map.items()
    ]
    messages = [{"role": "system", "content": system_message}] + \
               [{"role": "user", "content": m} for m in user_messages]
    return _chat(messages, max_tokens=600)


def generate_list_explanation(places, trait, score_type):
    system_message = (
        "You are a thoughtful, courteous assistant that offers an educated guess for "
        "why particular places rank highest and lowest on a Big Five trait. Where useful, "
        "blend in concrete things you know about the place. Use relative language "
        "(this is percentile-based). Be sensitive: there are no 'good' or 'bad' "
        "personalities. Limit your reply to ONE strong paragraph."
    )
    top_places = ', '.join(places['top'])
    bottom_places = ', '.join(places['bottom'])
    user_message = (
        f"The highest-ranking places for {trait} are {top_places}. "
        f"The lowest-ranking places for {trait} are {bottom_places}. "
        "Offer a brief, thoughtful explanation."
    )
    return _chat([{"role": "system", "content": system_message},
                  {"role": "user", "content": user_message}], max_tokens=500)


def generate_personality_comparison(selected1, selected2, percentiles1, percentiles2,
                                    trait_names_map, comparison_type):
    system_message = (
        "You are a thoughtful, courteous assistant that compares two places' Big Five "
        "percentile profiles. Synthesize the biggest differences intuitively rather "
        "than restating numbers. Blend in concrete things you know about the places. "
        "Use relative language (this is percentile-based). Be sensitive: there are no "
        "'good' or 'bad' personalities. Limit your reply to ONE or TWO paragraphs."
    )
    comparison_details = [
        f"{selected1} is in the {percentiles1[trait]} percentile and {selected2} "
        f"is in the {percentiles2[trait]} percentile for {trait_full.lower()}."
        for trait, trait_full in trait_names_map.items()
    ]
    return _chat([
        {"role": "system", "content": system_message},
        {"role": "user", "content": " ".join(comparison_details)},
    ], max_tokens=700)


# ---------- Single-location percentile plot ----------

def plot_percentile(percentiles, trait_names_values, selected, comparison_type):
    comparison_type = "US states" if comparison_type == "US States" else comparison_type.lower()

    y_labels = list(trait_names_values.values())
    original_values = list(percentiles.values())
    x_values = [max(p, 1) for p in original_values]

    # Color each bar by its percentile value using viridis
    colors = px.colors.sample_colorscale(CONTINUOUS_SCALE, [v / 100 for v in original_values])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y_labels,
        x=x_values,
        orientation='h',
        text=[f"{v:.0f}" for v in original_values],
        textposition='outside',
        textfont=dict(size=14, color=NEUTRAL_DARK),
        customdata=[[v, TRAIT_ADJECTIVES.get(lbl, 'unknown')] for v, lbl in zip(original_values, y_labels)],
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate=(
            f"<b>{selected}</b><br>"
            "More %{customdata[1]} than %{customdata[0]:.1f}% of "
            f"{comparison_type}<extra></extra>"
        ),
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Personality profile of {selected}</b>",
            x=0.5, xanchor='center', y=0.95,
            font=dict(size=22, family=PLOTLY_FONT['family'], color=NEUTRAL_DARK),
        ),
        xaxis=dict(title="Percentile", range=[0, 108],
                   gridcolor="#EEF2F7", zerolinecolor="#E5E7EB"),
        yaxis=dict(title="", autorange="reversed",
                   tickfont=dict(size=14, color=NEUTRAL_DARK)),
        font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=80, b=40, l=20, r=20),
        height=440,
        hoverlabel=dict(bgcolor="white", bordercolor=PRIMARY,
                        font=dict(family=PLOTLY_FONT['family'], size=13, color=NEUTRAL_DARK)),
    )
    return fig


def display_percentile(comparison_type, selected, data):
    if comparison_type == "Global Cities":
        selected_data = data[data['CityState'] == selected]
    elif comparison_type == "US Cities":
        selected_data = data[data['City'] == selected]
    elif comparison_type == "Countries":
        selected_data = data[data['Country'] == selected]
    elif comparison_type == "US States":
        selected_data = data[data['State'] == selected]

    percentiles = compute_percentile(data, selected_data, trait_names)
    fig = plot_percentile(percentiles, trait_names, selected, comparison_type)
    st.plotly_chart(fig, use_container_width=True)

    description = generate_personality_description(selected, percentiles, trait_names)
    st.markdown(
        f'<div class="card" style="margin-top:6px;">'
        f'<div style="font-weight:600;font-size:1.05em;color:{NEUTRAL_DARK};margin-bottom:8px;">'
        f'Profile narrative — {selected}</div>'
        f'<div style="color:#334155;line-height:1.6;font-size:0.98em;">{description}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------- Percentile math ----------

def compute_percentile(data, selected_data, trait_names_map):
    percentile_scores = {}
    if isinstance(selected_data, pd.DataFrame):
        if selected_data.empty:
            raise ValueError("No data found for the selected location.")
        selected_data = selected_data.iloc[0]

    for trait in trait_names_map:
        scores = data[trait].values
        selected_score = selected_data[trait]
        less_than = np.sum(scores < selected_score)
        equal_to = np.sum(scores == selected_score)
        percentile = (less_than + 0.5 * equal_to) / len(scores) * 100
        percentile_scores[trait] = round(percentile, 2)
    return percentile_scores


def compute_percentiles_for_all(data, trait_names_map):
    new_data = data.copy()
    for i, row in new_data.iterrows():
        percentiles = compute_percentile(data.drop(index=i), row, trait_names_map)
        for trait, percentile in percentiles.items():
            new_data.at[i, trait] = percentile
    return new_data
