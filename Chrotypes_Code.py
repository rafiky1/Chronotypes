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

# Title and description (Youssef)
st.title("Chronotypes and Sleep Quality Scores")
st.write(
    """
    This application analyzes wearable data to explore chronotypes, sleep quality, and stress levels. It includes advanced visualizations and machine learning predictions.
###
    Exploring how wearable technology data can help us understand and improve sleep patterns and stress management.

    """
)

# Load dataset (Youssef)
data_url = "https://raw.githubusercontent.com/C0D3Dr4G0N/SCSU-CSC398/refs/heads/main/wearable_tech_sleep_quality_1%201.csv"
try:
    data = pd.read_csv(data_url)
    st.subheader("Dataset Overview")
    st.write("This dataset contains information about heart rate variability, stress levels, sleep duration, and consistency, which we analyze to classify chronotypes.")
    st.dataframe(data.head())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Define chronotypes (Youssef)
def classify_chronotype(row):
    if row['Bedtime_Consistency'] > 0.7 and row['Sleep_Duration_Hours'] >= 7:
        return 'Early Bird'
    elif row['Bedtime_Consistency'] < 0.4 and row['Sleep_Duration_Hours'] < 7:
        return 'Night Owl'
    else:
        return 'Intermediate'

# Add chronotype column (Youssef)
try:
    data['Chronotype'] = data.apply(classify_chronotype, axis=1)
except Exception as e:
    st.error(f"Error classifying chronotypes: {e}")
    st.stop()

# Chronotype Distribution (Youssef)
st.subheader("Chronotype Distribution")
st.write("Explore the distribution of chronotypes in the dataset.")
chronotype_counts = data['Chronotype'].value_counts()
fig = go.Figure(go.Pie(labels=chronotype_counts.index, values=chronotype_counts.values, hole=0.3))
fig.update_layout(title_text="Chronotype Distribution")
st.plotly_chart(fig)

# Sleep Quality, Caffeine Intake and Sleep Duration Comparison (Sydney)
st.subheader("Sleep Quality, Caffeine Intake and Sleep Duration Comparison")
st.write("Comparing sleep quality, caffeine intake, and sleep duration across chronotypes.")
if 'Sleep_Quality_Score' in data.columns and 'Caffeine_Intake_mg' in data.columns and 'Sleep_Duration_Hours' in data.columns:
    data['Sleep_Quality_Score'] = data['Sleep_Quality_Score'].astype(float)
    data['Caffeine_Intake_mg'] = data['Caffeine_Intake_mg'].astype(float)
    data['Sleep_Duration_Hours'] = data['Sleep_Duration_Hours'].astype(float)
    st.subheader("Sleep Quality vs Caffeine Intake vs Sleep Duration")
    st.write("This bubble chart shows the relationship between sleep quality, caffeine intake, and sleep duration.")
    fig, ax = plt.subplots(figsize=(10, 6))
    bubble_chart = sns.scatterplot(
        x='Caffeine_Intake_mg', 
        y='Sleep_Quality_Score', 
        size='Sleep_Duration_Hours', 
        hue='Sleep_Duration_Hours',
        sizes=(20, 200),
        alpha=0.5,
        data=data, 
        ax=ax
    )
    ax.set_title("Sleep Quality vs Caffeine Intake vs Sleep Duration")
    ax.set_xlabel("Caffeine Intake (mg)")
    ax.set_ylabel("Sleep Quality Score")
    ax.legend(title="Sleep Duration (Hours)")
    st.pyplot(fig)
else:
    st.write("Sleep Quality Score, Caffeine Intake, or Sleep Duration not found in the dataset.")


# Correlation heatmap (Youssef)
st.subheader("Correlation Heatmap")
st.write("This heatmap shows relationships between numerical features in the dataset.")
try:
    corr = data.select_dtypes(include=['float64', 'int64']).corr()
    heatmap_fig = px.imshow(corr, text_auto=True, color_continuous_scale="viridis")
    heatmap_fig.update_layout(title="Feature Correlation Heatmap")
    st.plotly_chart(heatmap_fig)
except Exception as e:
    st.error(f"Error generating correlation heatmap: {e}")



#Sleep Cycle Visualization (Sydney)
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



# Clustering analysis (Youssef)
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

# Visualize the user's position on the clustering plot (Youssef)
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


# Chronotype prediction (Youssef)
st.subheader("Predict Your Chronotype")
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

# Adding User Input for Chronotype Prediction (Youssef)

st.write("Input your sleep and stress metrics below to discover your chronotype!")

# User input section
try:
    hrv = st.number_input("Heart Rate Variability (ms)", min_value=30.0, max_value=150.0, value=60.0, step=1.0)
    stress = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    bedtime_consistency = st.slider("Bedtime Consistency (1-10)", min_value=1, max_value=10, value=7)

    # Prepare user input for prediction
    user_data = pd.DataFrame([[hrv, stress, sleep_duration, bedtime_consistency]],
                             columns=chronotype_features)

    if st.button("Predict My Chronotype"):
        predicted_class = classifier.predict(user_data)[0]
        confidence_scores = classifier.predict_proba(user_data)[0]

        # Decode prediction
        chronotype_labels = ['Early Bird', 'Intermediate', 'Night Owl']
        predicted_label = chronotype_labels[predicted_class]

        # Display prediction
        st.subheader(f"Your Predicted Chronotype: **{predicted_label}**")

        # Display confidence scores
        st.write("**Confidence Scores:**")
        for idx, chronotype in enumerate(chronotype_labels):
            st.write(f"- {chronotype}: {confidence_scores[idx]*100:.2f}%")

    
except Exception as e:
    st.error(f"An error occurred during user prediction: {e}")

# Train Machine Learning on the data, then using user input, spit out a
# predicted sleep score, user uses sliders to see how the sleep score changes (Justin)
 
### Using these data because these are values that the "patient" can control
st.subheader("Predict Your Sleep Quality Score")
st.write("Enter the following values to predict your sleep quality score:")
 
X = data[['Sleep_Duration_Hours', 'Caffeine_Intake_mg', 'Stress_Level', 'Bedtime_Consistency', 'Light_Exposure_hours']]
y = data['Sleep_Quality_Score']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Train the KNeighborsRegressor model
knn = KNeighborsRegressor(n_neighbors=10, weights='uniform')
knn.fit(X_train, y_train)
 
# R-squared score for evaluation (optional, for display in the app)
st.write(f"Model R-squared Score: {knn.score(X_test, y_test):.2f}")
 
# User input for prediction
st.subheader("Enter Your Data:")
sleep_duration = st.number_input("Sleep Duration (hours):", min_value=0.0, max_value=24.0)
caffeine_intake = st.number_input("Caffeine Intake (mg):", min_value=0.0, max_value=500.0)
stress_level = st.number_input("Stress Level (0-10):", min_value=0.0, max_value=10.0)
bedtime_consistency = st.number_input("Bedtime Consistency (0.0-1.0):", min_value=0.0, max_value=1.0)
light_exposure = st.number_input("Light Exposure (hours):", min_value=0.0, max_value=24.0)
 
# Predicting based on user input (Justin)
if st.button("Predict"):
    user_input = np.array([[sleep_duration, caffeine_intake,
                            stress_level, bedtime_consistency, light_exposure]])
    prediction = knn.predict(user_input)
    st.success(f"Your predicted sleep quality score is: {float(prediction[0]):.2f}")
# Grouped Bar Chart for Avg Sleep Duration, Caffeine Intake, Stress Level, Bedtime Consistency, and Light Exposure, Compared to User Entry
st.subheader("Avg Prediction Metric Comparison (Grouped Bar Charts)")
st.write("This chart compares averages of sleep durations and light exposure times compared to user.")
 
# Averages for Grouped Bar Chart Comparison
 
avg_sleepdur = X.loc[:, 'Sleep_Duration_Hours'].mean().round(2)
avg_caffintake = X.loc[:, 'Caffeine_Intake_mg'].mean().round(2)
avg_stresslvl = X.loc[:, 'Stress_Level'].mean().round(2)
avg_bedconsis = X.loc[:, 'Bedtime_Consistency'].mean().round(2)
avg_lightexpose = X.loc[:, 'Light_Exposure_hours'].mean().round(2)
 
# Sleep Duration and Light Exposure
 
groups = ['Sleep Duration', 'Light Exposure']
values1 = [avg_sleepdur, avg_lightexpose]
values2 = [sleep_duration, light_exposure]
 
# Create positions for the bars
x = np.arange(len(groups))
width = 0.35
 
# Create the figure and axes
fig, ax = plt.subplots()
 
# Plot the bars
rects1 = ax.bar(x - width/2, values1, width, label='Average')
rects2 = ax.bar(x + width/2, values2, width, label='User')
 
# Add labels, title, and legend
ax.set_ylabel('Values')
ax.set_title('Sleep Duration and Light Exposure Grouped Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()
 
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
 
st.pyplot(fig)
 
# Caffeine Intake
 
groups = ['Caffeine Intake']
values1 = [avg_caffintake]
values2 = [caffeine_intake]
 
# Create positions for the bars
x = np.arange(len(groups))
width = 0.35
 
# Create the figure and axes
fig, ax = plt.subplots()
 
# Plot the bars
rects1 = ax.bar(x - width/2, values1, width, label='Average')
rects2 = ax.bar(x + width/2, values2, width, label='User')
 
# Add labels, title, and legend
ax.set_ylabel('Values')
ax.set_title('Caffeine Intake Grouped Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()
 
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
 
st.pyplot(fig)
 
# Stress Level
 
groups = ['Stress Level']
values1 = [avg_stresslvl]
values2 = [stress_level]
 
# Create positions for the bars
x = np.arange(len(groups))
width = 0.35
 
# Create the figure and axes
fig, ax = plt.subplots()
 
# Plot the bars
rects1 = ax.bar(x - width/2, values1, width, label='Average')
rects2 = ax.bar(x + width/2, values2, width, label='User')
 
# Add labels, title, and legend
ax.set_ylabel('Values')
ax.set_title('Stress Level Grouped Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()
 
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
 
st.pyplot(fig)
 
# Bedtime Consistency
 
groups = ['Bedtime Consistency']
values1 = [avg_bedconsis]
values2 = [bedtime_consistency]
 
# Create positions for the bars
x = np.arange(len(groups))
width = 0.35
 
# Create the figure and axes
fig, ax = plt.subplots()
 
# Plot the bars
rects1 = ax.bar(x - width/2, values1, width, label='Average')
rects2 = ax.bar(x + width/2, values2, width, label='User')
 
# Add labels, title, and legend
ax.set_ylabel('Values')
ax.set_title('Bedtime Consistency Grouped Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()
 
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
 
st.pyplot(fig)

#Advanced Visualizations (Youssef)
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



# Add a habit tracker section (Youssef)
st.subheader("Habit Tracker and Optimization Suggestions")
st.write("Log your habits and receive quick suggestions for better sleep and lower stress.")

# User Inputs
bedtime = st.text_input("What time do you go to bed? (e.g., 10:30 PM)")
wake_time = st.text_input("What time do you wake up? (e.g., 6:30 AM)")
consistency = st.slider("Rate your bedtime consistency (1-10):", 1, 10, 7)
stress = st.slider("Rate your stress levels before bed (1-10):", 1, 10, 5)

# Calculate Sleep Duration
def calculate_sleep(bed, wake):
    try:
        from datetime import datetime, timedelta
        bed_time = datetime.strptime(bed, "%I:%M %p")
        wake_time = datetime.strptime(wake, "%I:%M %p")
        if wake_time < bed_time:  # Handle overnight sleep
            wake_time += timedelta(days=1)
        return (wake_time - bed_time).total_seconds() / 3600  # Convert to hours
    except:
        return None

if st.button("Submit Habit Log"):
    sleep_duration = calculate_sleep(bedtime, wake_time)

    # Display Results
    st.write("### Your Habit Log")
    st.write(f"Bedtime: {bedtime}")
    st.write(f"Wake Time: {wake_time}")
    st.write(f"Consistency Rating: {consistency}")
    st.write(f"Stress Rating: {stress}")
    
    if sleep_duration is not None:
        st.write(f"Estimated Sleep Duration: **{sleep_duration:.1f} hours**")
    else:
        st.error("Invalid time format. Use HH:MM AM/PM.")

    # Suggestions
    st.write("### Personalized Suggestions")
    if sleep_duration is not None and sleep_duration < 6:
        st.write("- Aim for 7-9 hours of sleep. Avoid caffeine in the evening.")
    elif sleep_duration and sleep_duration >= 6:
        st.write("- Good sleep duration! Focus on maintaining it.")
    if consistency < 5:
        st.write("- Improve bedtime consistency with a fixed schedule.")
    if stress > 7:
        st.write("- Reduce stress before bed with relaxation techniques or reading.")

    st.write("- Maintain a cool, dark, and quiet sleep environment.")


