#Importing Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from dateutil import parser


st.set_page_config(page_title="Global Climate Analyzer", layout="wide")
st.title("🌍 Global Climate Dataset Analyzer")
st.markdown("Upload a climate dataset to explore EDA, visualize temperature trends, and predict future values.")

st.markdown(
    """
    <style>
    /* App background and general text */
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                          url('https://images.pexels.com/photos/209831/pexels-photo-209831.jpeg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
        font-size: 18px;
    }

    html, body, [class*="css"] {
        color: white !important;
    }

    /* Sidebar appearance */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .st-bb {
        color: black !important;
        font-weight: 600;
    }
    


    </style>
    
    """,
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("📁 Upload your climate CSV file", type=["csv"])

def convert_coord(coord):
    try:
        if isinstance(coord, float) or isinstance(coord, int):
            return float(coord)
        coord = coord.strip()
        if coord[-1] in ['N', 'E']:
            return float(coord[:-1])
        elif coord[-1] in ['S', 'W']:
            return -float(coord[:-1])
        return float(coord)
    except:
        return None

def parsing_climate_data_date(date):
    try:
        return parser.parse(date, dayfirst=False) 
    except:
        return pd.NaT
    
#Cleaning and Preprocessing dataset
def preprocessing_data(climate_data, year_col, temp_col, lat_col=None, lon_col=None):

    #Extracting year, month, day
    if 'dt' in climate_data.columns:
        try:
            climate_data['dt'] = pd.to_datetime(climate_data['dt'], errors='coerce')

            # Check if parsing was successful
            if climate_data['dt'].isnull().all():
                st.warning("⚠️ Warning: 'dt' column could not be parsed to dates.")
                climate_data.drop(columns=['dt'], inplace=True)
            else:
                # Only extract Year from valid 'dt'
                if 'Year' not in climate_data.columns:
                    climate_data['Year'] = climate_data['dt'].dt.year
                    year_col = 'Year'

        except Exception as e:
            st.error(f"❌ Error: Cannot parse date column: {e}")



    #Detecting outliers from average temperature
    if 'AverageTemperature' in climate_data.columns or temp_col in climate_data.columns:
        target_temp_col = 'AverageTemperature' if 'AverageTemperature' in climate_data.columns else temp_col
        climate_data[target_temp_col] = pd.to_numeric(climate_data[target_temp_col], errors='coerce')

        Q1 = climate_data[target_temp_col].quantile(0.25)
        Q3 = climate_data[target_temp_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        climate_data = climate_data[(climate_data[target_temp_col] >= lower_bound) & (climate_data[target_temp_col] <= upper_bound)]
    else:
        st.error("❌ 'AverageTemperature' column not found.")
        return pd.DataFrame()

    if year_col not in climate_data.columns or temp_col not in climate_data.columns:
        st.warning("⚠️ Selected Year or Temperature column not found. Attempting automatic detection...")
        numeric_cols = climate_data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) >= 2:
            year_col, temp_col = numeric_cols[0], numeric_cols[1]
            st.info(f"Auto-selected '{year_col}' as Year and '{temp_col}' as Temperature.")
        else:
            st.error("❌ Could not auto-detect appropriate columns.")
            return pd.DataFrame()

    climate_data[year_col] = pd.to_numeric(climate_data[year_col], errors='coerce')
    climate_data[temp_col] = pd.to_numeric(climate_data[temp_col], errors='coerce')
    climate_data = climate_data.dropna(subset=[year_col, temp_col])
    climate_data = climate_data.rename(columns={year_col: "Year", temp_col: "AverageTemperature"})

    climate_data['AverageTemperature'] = climate_data['AverageTemperature'].interpolate()

    #Converting coordinate in latitude and longitude
    if lat_col and lat_col in climate_data.columns:
        climate_data['Latitude'] = climate_data[lat_col].apply(convert_coord)
    elif lat_col:
        st.warning(f"⚠️ Latitude column '{lat_col}' not found.")

    if lon_col and lon_col in climate_data.columns:
        climate_data['Longitude'] = climate_data[lon_col].apply(convert_coord)
    elif lon_col:
        st.warning(f"⚠️ Longitude column '{lon_col}' not found.")

    return climate_data

#Exploratory Data analysis
def explore_data(climate_data):
    st.subheader("📊 Dataset Overview")
    st.write("Shape:", climate_data.shape)
    st.dataframe(climate_data.head(50))

    with st.expander("🔎 Data Types & Missing Values"):
        st.write(climate_data.dtypes)
        st.write(climate_data.isnull().sum())

    with st.expander("📌 Summary Statistics"):
        st.write(climate_data.describe())


#Visualizing dataset
def visualize_dataset(climate_data):
    st.subheader("📊 Visualize Dataset")
    available_columns = climate_data.columns.tolist()
    
    visualization_type = st.selectbox("Choose Visualization Type", 
                                     ["Temperature Distribution", "Temperature Trend Over Years", 
                                     
                                      "Top 10 Hottest Cities", "Latitude vs Temperature", 
                                      
                                      "Temperature vs Longitude", 
                                    ])
    

    if visualization_type == "Temperature Distribution":
        if 'AverageTemperature' in available_columns:
            st.subheader("Temperature Distribution")
            plt.figure(figsize=(10,5))
            sns.histplot(climate_data['AverageTemperature'], bins=30, kde=True, color='royalblue')
            plt.title("Distribution of Average Temperatures")
            plt.xlabel("Average Temperature (°C)")
            plt.ylabel("Frequency")
            st.pyplot(plt.gcf())
        else:
            st.warning("Column 'AverageTemperature' not found in dataset!")

    elif visualization_type == "Temperature Trend Over Years":
        if 'Year' in available_columns and 'AverageTemperature' in available_columns:
            st.subheader("Temperature Trend Over Years")
            plt.figure(figsize=(12,6))
            sns.lineplot(x=climate_data['Year'], y=climate_data['AverageTemperature'], marker="o", color="red")
            plt.title("Temperature Change Over the Years")
            plt.xlabel("Year")
            plt.ylabel("Average Temperature (°C)")
            plt.grid()
            st.pyplot(plt.gcf())
        else:
            st.warning("Columns 'Year' or 'AverageTemperature' not found in dataset!")

    elif visualization_type == "Top 10 Hottest Cities":
        if 'City' in available_columns and 'AverageTemperature' in available_columns:
            st.subheader("Top 10 Hottest Cities")
            top_cities = climate_data.groupby('City')['AverageTemperature'].mean().nlargest(10)
            plt.figure(figsize=(12,6))
            sns.barplot(x=top_cities.index, y=top_cities.values, palette="Reds")
            plt.xticks(rotation=90)
            plt.title("Top 10 Hottest Cities by Average Temperature")
            plt.ylabel("Average Temperature (°C)")
            st.pyplot(plt.gcf())
        else:
            st.warning("Columns 'City' or 'AverageTemperature' not found in dataset!")

    elif visualization_type == "Latitude vs Temperature":
        if 'Latitude' in available_columns and 'AverageTemperature' in available_columns:
            st.subheader("Latitude vs Temperature")
            plt.figure(figsize=(10,5))
            sns.scatterplot(x='Latitude', y='AverageTemperature', data=climate_data, alpha=0.5, color='purple')
            plt.title("Latitude vs. Average Temperature")
            plt.xlabel("Latitude")
            plt.ylabel("Average Temperature (°C)")
            st.pyplot(plt.gcf())
        else:
            st.warning("Columns 'Latitude' or 'AverageTemperature' not found in dataset!")

    elif visualization_type == "Temperature vs Longitude":
        if 'Longitude' in available_columns and 'AverageTemperature' in available_columns:
            st.subheader("Temperature vs Longitude")
            plt.figure(figsize=(10,5))
            sns.scatterplot(x='Longitude', y='AverageTemperature', data=climate_data, alpha=0.5, color='orange')
            plt.title("Longitude vs. Average Temperature")
            plt.xlabel("Longitude")
            plt.ylabel("Average Temperature (°C)")
            st.pyplot(plt.gcf())
        else:
            st.warning("Columns 'Longitude' or 'AverageTemperature' not found in dataset!")



#Visualizing trend in the temperature
def show_trend(climate_data):
    if 'Year' in climate_data.columns and 'AverageTemperature' in climate_data.columns:
        yearly_avg = climate_data.groupby('Year')['AverageTemperature'].mean().reset_index()

        trace = go.Scatter(
            x=yearly_avg['Year'],
            y=yearly_avg['AverageTemperature'],
            mode='lines+markers',
            name='Yearly Average Temperature',
            line=dict(color='blue'),
            marker=dict(symbol='circle', size=8)
        )

        layout = go.Layout(
            title="Average Temperature Over the Years",
            xaxis=dict(title='Year'),
            yaxis=dict(title='Average Temperature (°C)'),
            showlegend=True,
            hovermode='closest',
            template='plotly_dark'
        )

        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig)

    else:
        st.error("❌ 'Year' or 'AverageTemperature' column cannot be found ")

#Using Linear regression to predict temperature
def predict_temperature(climate_data):
    st.subheader("Predict Future Temperature")

    if 'Year' not in climate_data.columns or 'AverageTemperature' not in climate_data.columns:
        st.error("❌ Required columns ('Year' and 'AverageTemperature') missing.")
        return

    climate_data = climate_data.dropna(subset=['Year', 'AverageTemperature'])
    X = climate_data[['Year']]
    y = climate_data['AverageTemperature']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    st.markdown(
        """
        <style>
        .slider-label {
            font-size: 18px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<div class="slider-label">Select year to predict temperature</div>', unsafe_allow_html=True)

    year_to_predict = st.slider(
        ".", 
        int(climate_data['Year'].min()), 
        int(climate_data['Year'].max()) + 10,
        key="year_slider"
    )

    year_scaled = scaler.transform([[year_to_predict]])
    predicted_temp = model.predict(year_scaled)[0]
    predicted_temp = np.clip(predicted_temp, -50, 60)
    st.markdown(
        f"""
        <div class="stAlert-success" style="text-align:center; font-size: 20px;">
            Estimated Avg Temp for {year_to_predict}: <strong>{predicted_temp:.2f}°C</strong>
        </div>
        """, unsafe_allow_html=True
    )
    trace1 = go.Scatter(
        x=climate_data['Year'],
        y=climate_data['AverageTemperature'],
        mode='markers',
        name='Historical Data',
        marker=dict(color='royalblue', size=6, opacity=0.6)
    )

    trace2 = go.Scatter(
        x=[year_to_predict, year_to_predict],
        y=[climate_data['AverageTemperature'].min(), climate_data['AverageTemperature'].max()],
        mode='lines',
        name=f'Predicted Year ({year_to_predict})',
        line=dict(color='green', dash='dash')
    )

    future_years = np.arange(2025, 2031)
    future_years_scaled = scaler.transform(future_years.reshape(-1, 1))
    future_temps = model.predict(future_years_scaled)

    trace3 = go.Scatter(
        x=future_years,
        y=future_temps,
        mode='lines+markers',
        name='Predicted Future Temps',
        line=dict(color='orange', dash='dot')
    )

    layout = go.Layout(
        title="Temperature Prediction",
        xaxis=dict(title='Year'),
        yaxis=dict(title='Average Temperature (°C)'),
        hovermode='closest',
        showlegend=True
    )

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    st.plotly_chart(fig)

def location_based_analysis(climate_data):
    st.subheader("📍 Location-Based Temperature Analysis")

    # Check if there are potential location columns
    location_columns = [col for col in climate_data.columns if any(keyword in col.lower() for keyword in ['city', 'state', 'region', 'country'])]

    if not location_columns:
        st.warning("⚠️ No City, State, Region, or Country column found in the dataset!")
        return

    selected_location = st.selectbox("Select Location Column to Analyze:", location_columns)

    if selected_location:
        temp_summary = climate_data.groupby(selected_location)['AverageTemperature'].mean().sort_values(ascending=False).reset_index()

        # Plotting graph 
        plt.figure(figsize=(14, 6))
        sns.barplot(x=selected_location, y='AverageTemperature', data=temp_summary, palette='coolwarm')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Average Temperature by {selected_location}")
        plt.ylabel("Average Temperature (°C)")
        plt.xlabel(selected_location)
        st.pyplot(plt.gcf())

        with st.expander(f"📋 Full Data: {selected_location} Temperature Averages"):
            st.dataframe(temp_summary)




if uploaded_file is not None:
    try:
        climate_data_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded!")

        st.sidebar.subheader("Column Mapping")
        default_year_col = 'dt' if 'dt' in climate_data_raw.columns else climate_data_raw.columns[0]
        default_temp_col = 'AverageTemperature' if 'AverageTemperature' in climate_data_raw.columns else climate_data_raw.columns[1]


       # Year, Temperature, Latitude, Longitude
        year_col = st.sidebar.selectbox("Select Year Column", climate_data_raw.columns, index=climate_data_raw.columns.get_loc(default_year_col))
        temp_col = st.sidebar.selectbox("Select Temperature Column", climate_data_raw.columns, index=climate_data_raw.columns.get_loc(default_temp_col))
        lat_col = st.sidebar.selectbox("Select Latitude Column (optional)", ["None"] + list(climate_data_raw.columns))
        lon_col = st.sidebar.selectbox("Select Longitude Column (optional)", ["None"] + list(climate_data_raw.columns))

        # Location Mapping (City / State / Region / Country)
        location_col = st.sidebar.selectbox(
            "Select Location Column (optional - City/State/Region/Country)", 
             ["None"] + list(climate_data_raw.columns)
            )

        lat_col = None if lat_col == "None" else lat_col
        lon_col = None if lon_col == "None" else lon_col
        location_col = None if location_col == "None" else location_col


        climate_data = preprocessing_data(climate_data_raw.copy(), year_col, temp_col, lat_col, lon_col)

        st.sidebar.subheader("📌 Select View")
        view_option = st.sidebar.radio(
    "What would you like to do?",
    ("Dataset Insight", "Visualize Dataset", "Trend Visualization", "Temperature Prediction", "City/State/Region Analysis")
)

        if view_option == "Dataset Insight":
            explore_data(climate_data)
        elif view_option == "Visualize Dataset":
            visualize_dataset(climate_data)
        elif view_option == "Trend Visualization":
            show_trend(climate_data)
        elif view_option == "Temperature Prediction":
            predict_temperature(climate_data)
        elif view_option == "City/State/Region Analysis":
            location_based_analysis(climate_data)


    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    
    st.markdown(
    f"""
    <div class="stAlert-success" style="text-align:left" font-size: 18px;>
        📤 Upload a CSV file to begin.
    </div>
    """, unsafe_allow_html=True
    )
