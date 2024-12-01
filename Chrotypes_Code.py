import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import plotly.express as px
import plotly.graph_objects as go

# Title and description
st.title("Chronotypes Analysis: Night Owls vs. Early Birds")
st.write(
    """
    This application analyzes wearable data to explore chronotypes, sleep quality, and stress levels. It includes advanced visualizations and machine learning predictions.
### Chronotype Categories:
- **Early Bird**: Early sleepers who maintain a consistent bedtime and get 7+ hours of sleep.
- **Night Owl**: Late sleepers with inconsistent bedtimes and shorter sleep durations.
- **Intermediate**: Those who fall between the two extremes.

### How We Measure Chronotypes:
1. **Bedtime Consistency**: Measures how consistent your bedtime is (scale: 0 to 1). A higher score means you go to bed around the same time daily.
2. **Sleep Duration**: Total hours of sleep per night.
   - **Early Bird**: Bedtime Consistency > 0.7 and Sleep Duration ≥ 7 hours.
   - **Night Owl**: Bedtime Consistency < 0.4 and Sleep Duration < 7 hours.
   - **Intermediate**: All others.
    """
)

# Load dataset
data_url = "https://raw.githubusercontent.com/C0D3Dr4G0N/SCSU-CSC398/refs/heads/main/wearable_tech_sleep_quality_1%201.csv"
try:
    data = pd.read_csv(data_url)
    st.subheader("Dataset Overview")
    st.write("This dataset contains information about heart rate variability, stress levels, sleep duration, and consistency, which we analyze to classify chronotypes.")
    st.dataframe(data.head())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Define chronotypes
def classify_chronotype(row):
    if row['Bedtime_Consistency'] > 0.7 and row['Sleep_Duration_Hours'] >= 7:
        return 'Early Bird'
    elif row['Bedtime_Consistency'] < 0.4 and row['Sleep_Duration_Hours'] < 7:
        return 'Night Owl'
    else:
        return 'Intermediate'

# Add chronotype column
try:
    data['Chronotype'] = data.apply(classify_chronotype, axis=1)
except Exception as e:
    st.error(f"Error classifying chronotypes: {e}")
    st.stop()

# Visualizations
st.subheader("Chronotype Distribution")
st.write("Explore the distribution of chronotypes in the dataset.")
chronotype_counts = data['Chronotype'].value_counts()
fig = go.Figure(go.Pie(labels=chronotype_counts.index, values=chronotype_counts.values, hole=0.3))
fig.update_layout(title_text="Chronotype Distribution")
st.plotly_chart(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
st.write("This heatmap shows relationships between numerical features in the dataset.")
try:
    corr = data.select_dtypes(include=['float64', 'int64']).corr()
    heatmap_fig = px.imshow(corr, text_auto=True, color_continuous_scale="viridis")
    heatmap_fig.update_layout(title="Feature Correlation Heatmap")
    st.plotly_chart(heatmap_fig)
except Exception as e:
    st.error(f"Error generating correlation heatmap: {e}")


# Sleep Quality and HRV Comparison
st.subheader("Sleep Quality and HRV Comparison")
st.write("Comparing sleep quality and HRV across chronotypes.")
if 'Sleep_Quality_Score' in data.columns and 'Heart_Rate_Variability' in data.columns:
  data['Sleep_Quality_Score'] = data['Sleep_Quality_Score'].astype(float)
  data['Heart_Rate_Variability'] = data['Heart_Rate_Variability'].astype(float)
  st.write("Sleep Quality vs HRV Comparison")
  st.write("This scatter plot shows the relationship between sleep quality and HRV.")
  fig, ax = plt.subplots(figsize=(8, 6))
  sns.scatterplot(x='Heart_Rate_Variability', y='Sleep_Quality_Score', data=data, ax=ax)
  ax.set_title("Sleep Quality vs HRV Comparison")
  ax.set_xlabel("Heart Rate Variability")
  ax.set_ylabel("Sleep Quality Score")
  st.pyplot(fig)
else:
  st.write("Sleep Quality Score or Heart Rate Variability not found in the dataset.")

#Sleep Cycle Visualization
st.subheader("Sleep Cycle Visualization")
st.write("Explore how your sleep is distributed across different stages throughout the night.")

if 'Sleep_Duration_Hours' in data.columns:
    try:
        # Convert sleep duration hours to minutes
        data['Sleep_Duration_Minutes'] = data['Sleep_Duration_Hours'] * 60

        # Estimate time spent in different sleep stages
        data['Light_Sleep'] = data['Sleep_Duration_Minutes'] * 0.55
        data['Deep_Sleep'] = data['Sleep_Duration_Minutes'] * 0.2
        data['REM_Sleep'] = data['Sleep_Duration_Minutes'] * 0.25

        # Calculate average time spent in each stage
        sleep_stage_totals = data[['Light_Sleep', 'Deep_Sleep', 'REM_Sleep']].mean().reset_index()
        sleep_stage_totals.columns = ['Sleep Stage', 'Average Time Spent (minutes)']

        # Create bar chart using Plotly
        st.subheader("Average Time Spent in Sleep Stages")
        st.write("This bar chart shows the average time spent in different sleep stages.")
        fig = px.bar(
            sleep_stage_totals,
            x="Sleep Stage",
            y="Average Time Spent (minutes)",
            color="Sleep Stage",
            title="Average Time Spent in Sleep Stages",
            labels={"Average Time Spent (minutes)": "Time Spent (minutes)", "Sleep Stage": "Sleep Stage"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            xaxis_title="Sleep Stage",
            yaxis_title="Average Time Spent (minutes)",
            legend_title="Sleep Stage",
            template="plotly_white",
            yaxis=dict(tickformat=".0f")  # Ensure no 'k' notation
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while processing the sleep cycle visualization: {e}")
else:
    st.warning("The 'Sleep_Duration_Hours' column is missing in the dataset. Please ensure it is included.")


#Sleep Quality vs. Light Exposure (Scatter Plot with Regression Line)
st.subheader("Sleep Quality vs. Light Exposure")
st.write("Analyze the relationship between light exposure and sleep quality.")

if 'Sleep_Quality_Score' in data.columns and 'Light_Exposure_hours' in data.columns:
    try:
        # Scatter plot with regression line
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Light_Exposure_hours',
            y='Sleep_Quality_Score',
            data=data,
            alpha=0.6,
            ax=ax
        )
        sns.regplot(
            x='Light_Exposure_hours',
            y='Sleep_Quality_Score',
            data=data,
            scatter=False,
            color='red',
            ax=ax
        )
        ax.set_title("Sleep Quality vs Light Exposure", fontsize=16)
        ax.set_xlabel("Light Exposure (hours)", fontsize=12)
        ax.set_ylabel("Sleep Quality Score", fontsize=12)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred while generating the Sleep Quality vs Light Exposure plot: {e}")
else:
    st.warning("Required columns ('Sleep_Quality_Score' or 'Light_Exposure_hours') are missing from the dataset.")


# Clustering analysis
st.subheader("Clustering Analysis")
st.write("Grouping individuals based on sleep and stress patterns.")
kmeans_features = ['Heart_Rate_Variability', 'Stress_Level', 'Sleep_Duration_Hours']
try:
    data_cleaned = data.dropna(subset=kmeans_features)
    scaler = StandardScaler()
    kmeans_scaled = scaler.fit_transform(data_cleaned[kmeans_features])
    kmeans = KMeans(n_clusters=3, random_state=42)
    data_cleaned['Cluster'] = kmeans.fit_predict(kmeans_scaled)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(kmeans_scaled)
    data_cleaned['PCA1'] = pca_results[:, 0]
    data_cleaned['PCA2'] = pca_results[:, 1]
    cluster_fig = px.scatter(
        data_cleaned, x='PCA1', y='PCA2', color='Cluster',
        title="Clustering Results (PCA-Reduced Features)",
        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'}
    )
    st.plotly_chart(cluster_fig)
except Exception as e:
    st.error(f"Error during clustering analysis: {e}")



# Chronotype prediction
st.subheader("Predicting Chronotypes")
st.write("Building a Random Forest model to predict chronotypes based on sleep and stress metrics.")
chronotype_features = ['Heart_Rate_Variability', 'Stress_Level', 'Sleep_Duration_Hours', 'Bedtime_Consistency']
try:
    X = data[chronotype_features].dropna()
    y = LabelEncoder().fit_transform(data['Chronotype'].dropna())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    st.text(classification_report(y_test, y_pred, target_names=['Early Bird', 'Intermediate', 'Night Owl']))
except Exception as e:
    st.error(f"Error during chronotype prediction: {e}")

#Advanced Visualizations 
st.subheader("Advanced Visualizations")
st.write("Analyze how sleep duration varies across different chronotypes over time.")

try:
    # Ensure the data includes the 'Chronotype' column
    if 'Chronotype' in data.columns:
        # Create a line plot to visualize sleep trends by chronotype
        advanced_fig = px.line(
            data,
            x=np.arange(len(data)),  # Assuming chronological order of entries
            y='Sleep_Duration_Hours',
            color='Chronotype',
            title="Sleep Duration Trends by Chronotype",
            labels={
                'x': 'Entry Index', 
                'Sleep_Duration_Hours': 'Sleep Duration (Hours)', 
                'Chronotype': 'Chronotype'
            }
        )
        # Update layout for better visualization
        #using record index since we don't have dates or times of sleep logs in out dataset
        advanced_fig.update_layout(
            xaxis_title="Record Index (Chronological Order)",
            yaxis_title="Sleep Duration (Hours)",
            legend_title="Chronotype",
            template="plotly_white"
        )
        # Display the chart
        st.plotly_chart(advanced_fig)
    else:
        st.warning("The 'Chronotype' column is missing in the dataset. Please ensure chronotypes are defined.")
except Exception as e:
    st.error(f"An error occurred while generating the visualization: {e}")



# Visualize the user's position on the clustering plot
try:
    st.subheader("Input Your Data for Cluster Visualization")
    # Get user inputs
    heart_rate_variability = st.number_input("Heart Rate Variability (ms):", min_value=0.0, max_value=200.0, step=1.0, value=50.0)
    stress_level = st.slider("Stress Level (1-10):", min_value=1, max_value=10, step=1, value=5)
    sleep_duration_hours = st.number_input("Sleep Duration (hours):", min_value=0.0, max_value=24.0, step=0.1, value=6.5)
    
    # Create a DataFrame from the input values
    quiz_data = pd.DataFrame({
        'Heart_Rate_Variability': [heart_rate_variability],
        'Stress_Level': [stress_level],
        'Sleep_Duration_Hours': [sleep_duration_hours]
    })
    
    # Scale the quiz data
    quiz_data_scaled = scaler.transform(quiz_data)
    # Predict the cluster for the quiz data
    predicted_cluster = kmeans.predict(quiz_data_scaled)[0]
    # Transform quiz data using PCA
    quiz_pca = pca.transform(quiz_data_scaled)
    # Prepare data for visualization
    quiz_point = pd.DataFrame({
        'PCA1': [quiz_pca[0, 0]],
        'PCA2': [quiz_pca[0, 1]],
        'Cluster': [predicted_cluster]
    })
    # Add the quiz data point to the clustering figure
    cluster_fig.add_trace(
        go.Scatter(
            x=quiz_point['PCA1'], y=quiz_point['PCA2'],
            mode='markers+text',
            text=["You"],
            textposition="top center",
            marker=dict(color='red', size=12, symbol='star')
        )
    )
    # Display the updated clustering plot
    st.plotly_chart(cluster_fig)
except Exception as e:
    st.error(f"Error visualizing your cluster position: {e}")



# Add a habit tracker section
st.subheader("Habit Tracker and Optimization Suggestions")
st.write("Log your habits and receive personalized suggestions for better sleep and lower stress.")

bedtime = st.text_input("What time do you usually go to bed? (e.g., 10:30 PM)")
wake_time = st.text_input("What time do you usually wake up? (e.g., 6:30 AM)")
consistency_rating = st.slider("Rate your bedtime consistency (1-10):", 1, 10, 7)
stress_rating = st.slider("Rate your stress levels before bed (1-10):", 1, 10, 5)

if st.button("Submit Habit Log"):
    st.write("### Habit Log Submitted")
    st.write(f"Bedtime: {bedtime}")
    st.write(f"Wake Time: {wake_time}")
    st.write(f"Consistency Rating: {consistency_rating}")
    st.write(f"Stress Rating: {stress_rating}")

    # Provide personalized suggestions
    st.write("### Personalized Suggestions Based on Your Log")
    if consistency_rating < 5:
        st.write("- Improve bedtime consistency by setting an alarm for bedtime.")
        st.write("- Avoid stimulating activities close to bedtime.")
    if stress_rating > 7:
        st.write("- Practice relaxation techniques like deep breathing or meditation.")
        st.write("- Consider reducing screen time an hour before bed.")
    st.write("- Maintain a regular wake-up time to improve overall sleep quality.")


# Train Machine Learning on the data, then using user input, spit out a
# predicted sleep score, user uses sliders to see how the sleep score changes

### Using these data because these are values that the "patient" can control
st.subheader("Predict Your Sleep Quality Score")
st.write("Enter the following values to predict your sleep quality score:")

X = data[['Body_Temperature', 'Sleep_Duration_Hours', 'Caffeine_Intake_mg',
          'Stress_Level', 'Bedtime_Consistency', 'Light_Exposure_hours']]
y = data['Sleep_Quality_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNeighborsRegressor model
knn = KNeighborsRegressor(n_neighbors=10, weights='uniform')
knn.fit(X_train, y_train)

# R-squared score for evaluation (optional, for display in the app)
st.write(f"Model R-squared Score: {knn.score(X_test, y_test):.2f}")

# User input for prediction
st.subheader("Enter Your Data")
body_temp = st.number_input("Body Temperature (°C):", min_value=35.0, max_value=40.0, value=36.5)
sleep_duration = st.number_input("Sleep Duration (hours):", min_value=0.0, max_value=12.0, value=8.0)
caffeine_intake = st.number_input("Caffeine Intake (mg):", min_value=0.0, max_value=500.0, value=50.0)
stress_level = st.number_input("Stress Level (1-10):", min_value=1.0, max_value=10.0, value=5.0)
bedtime_consistency = st.number_input("Bedtime Consistency (1-10):", min_value=1.0, max_value=10.0, value=7.0)
light_exposure = st.number_input("Light Exposure (hours):", min_value=0.0, max_value=24.0, value=2.0)

# Predicting based on user input
if st.button("Predict"):
    user_input = np.array([[body_temp, sleep_duration, caffeine_intake,
                            stress_level, bedtime_consistency, light_exposure]])
    prediction = knn.predict(user_input)
    st.success(f"Your predicted sleep quality score is: {float(prediction[0]):.2f}")
   
