import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Cross-Country Solar Analysis Dashboard", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .country-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load cleaned data (assuming CSV files are in a 'data' folder)
@st.cache_data
def load_data():
    file_paths = {
        'Benin': 'src/notebooks/data1/benin.csv',
        'Sierra Leone': 'src/notebooks/data1/sierraleone.csv',
        'Togo': 'src/notebooks/data1/togo.csv'
    }
    countries = {}
    for country, path in file_paths.items():
        try:
            df = pd.read_csv(path)
            df['Country'] = country
            # Parse timestamp
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            countries[country] = df
            st.info(f"✓ Loaded {country}: {df.shape[0]} records")
        except Exception as e:
            st.error(f"✗ Error loading {country}: {e}")
    
    if countries:
        comparison_df = pd.concat(countries.values(), ignore_index=True)
        comparison_df['Timestamp'] = pd.to_datetime(comparison_df['Timestamp'])
        return comparison_df, countries
    return None, {}

# Fixed CrossCountryAnalyzer class (with error fix)
class CrossCountryAnalyzer:
    def __init__(self, df):
        self.df = df
    
    def metric_comparison_boxplots(self):
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('GHI Distribution', 'DNI Distribution', 'DHI Distribution'),
            specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}]]
        )
        
        metrics = ['GHI', 'DNI', 'DHI']
        countries = self.df['Country'].unique()
        colors = px.colors.qualitative.Set2  # Use a palette for countries
        
        for col_idx, metric in enumerate(metrics, 1):
            for idx, country in enumerate(countries):
                country_data = self.df[self.df['Country'] == country][metric].dropna()
                fig.add_trace(
                    go.Box(
                        y=country_data.values, 
                        name=country, 
                        marker_color=colors[idx % len(colors)]
                    ),
                    row=1, col=col_idx
                )
        
        fig.update_layout(height=500, showlegend=True, title_text="Solar Irradiance Comparison Across Countries")
        return fig
    
    def summary_statistics_table(self):
        metrics = ['GHI', 'DNI', 'DHI']
        summary_data = []
        
        for country in self.df['Country'].unique():
            country_df = self.df[self.df['Country'] == country]
            for metric in metrics:
                row = {
                    'Country': country,
                    'Metric': metric,
                    'Mean': country_df[metric].mean(),
                    'Median': country_df[metric].median(),
                    'Std Dev': country_df[metric].std(),
                    'Min': country_df[metric].min(),
                    'Max': country_df[metric].max()
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        # Pivot for better display
        pivot_df = summary_df.pivot(index='Country', columns='Metric', values=['Mean', 'Median', 'Std Dev', 'Min', 'Max'])
        return pivot_df
    
    def generate_key_observations(self):
        # Fixed version: Get the index (country name) properly
        temp_means = self.df.groupby('Country')['Tamb'].mean()
        closest_idx = (temp_means - 25).abs().argsort()[0]
        optimal_temp_country = temp_means.index[closest_idx]
        optimal_temp_value = temp_means.iloc[closest_idx]
        
        ghi_means = self.df.groupby('Country')['GHI'].mean()
        dni_means = self.df.groupby('Country')['DNI'].mean()
        dhi_means = self.df.groupby('Country')['DHI'].mean()
        
        observations = f"""
        📊 SOLAR POTENTIAL RANKING:
        1. Benin: Avg GHI {ghi_means.get('Benin', 0):.2f} W/m²
        2. Togo: Avg GHI {ghi_means.get('Togo', 0):.2f} W/m²
        3. Sierra Leone: Avg GHI {ghi_means.get('Sierra Leone', 0):.2f} W/m²
        
        🎯 TOP INSIGHTS:
        1. Highest Solar Potential: Benin ({ghi_means.get('Benin', 0):.2f} W/m² GHI)
        2. Best DNI: Benin ({dni_means.get('Benin', 0):.2f} W/m²)
        3. Optimal Temperature: {optimal_temp_country} ({optimal_temp_value:.2f}°C)
        """
        return observations

# Individual country insights (simplified from notebooks)
def generate_country_insights(df, country):
    if df is None or df.empty:
        return "No data available."
    
    # Extract hour from timestamp for peak
    df['Hour'] = df['Timestamp'].dt.hour
    peak_hour = df.groupby('Hour')['GHI'].mean().idxmax()
    
    insights = f"""
    ### {country} Key Insights
    - **Avg GHI**: {df['GHI'].mean():.2f} W/m²
    - **Avg DNI**: {df['DNI'].mean():.2f} W/m²
    - **Avg DHI**: {df['DHI'].mean():.2f} W/m²
    - **Avg Temp**: {df['Tamb'].mean():.2f} °C
    - **Peak Hour**: {peak_hour}:00
    """
    return insights

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🌞 Cross-Country Solar Farm Analysis</h1>', unsafe_allow_html=True)
    
    # Load data
    comparison_df, countries = load_data()
    
    if comparison_df is None:
        st.error("No data loaded. Please check file paths.")
        return
    
    comparator = CrossCountryAnalyzer(comparison_df)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Overview", "Country Comparison", "Individual Countries", "Key Observations"])
    
    if page == "Overview":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(comparison_df))
        with col2:
            st.metric("Countries Analyzed", len(countries))
        with col3:
            avg_ghi = comparison_df['GHI'].mean()
            st.metric("Overall Avg GHI", f"{avg_ghi:.2f} W/m²")
        
        # Bubble chart (simplified - assuming hourly data; adjust index if needed)
        comparison_df['Hour'] = comparison_df['Timestamp'].dt.hour
        fig_bubble = px.scatter(comparison_df.sample(5000), x='Hour', y='GHI', size='DNI', color='Country',  # Sample for performance
                                title="Solar Irradiance Bubble Chart (Size = DNI)")
        st.plotly_chart(fig_bubble, use_container_width=True)
    
    elif page == "Country Comparison":
        st.subheader("Box Plots Comparison")
        fig_box = comparator.metric_comparison_boxplots()
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.subheader("Summary Statistics")
        summary_table = comparator.summary_statistics_table()
        st.dataframe(summary_table, use_container_width=True)
    
    elif page == "Individual Countries":
        country = st.selectbox("Select Country", list(countries.keys()))
        df_country = countries[country].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="country-card">{generate_country_insights(df_country, country)}</div>', unsafe_allow_html=True)
        with col2:
            # Simple line plot for GHI over time (sample for performance)
            sample_df = df_country.head(1000).copy()
            fig_line = px.line(sample_df, x='Timestamp', y='GHI', title=f"{country} GHI Trend (First 1000 records)")
            st.plotly_chart(fig_line, use_container_width=True)
    
    elif page == "Key Observations":
        st.markdown(comparator.generate_key_observations())

if __name__ == "__main__":
    main()
