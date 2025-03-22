import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated dataset (Replace with real keyword data)
data = {
    'Keyword': ['marketing', 'digital marketing', 'SEO', 'content marketing', 'PPC'],
    'Search Volume': [1000000, 750000, 500000, 300000, 250000],
    'CPC': [1.96, 2.5, 1.2, 0.9, 2.0],
    'Keyword Difficulty': [99, 85, 70, 60, 75]
}
df = pd.DataFrame(data)

# Prepare data for ML model
X = df[['Search Volume', 'CPC']]
y = df['Keyword Difficulty']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Machine Learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Keyword Magic Tool")
st.write("Find millions of keyword suggestions for your SEO.")

# User input
keyword_input = st.text_input("Enter Keyword:")

if keyword_input:
    # Simulate predictions (Replace with real ML inference)
    pred_data = np.array([[np.random.randint(100000, 1000000), np.random.uniform(0.5, 3.0)]])
    pred_data_scaled = scaler.transform(pred_data)
    pred_kd = model.predict(pred_data_scaled)
    
    st.subheader("Keyword Data:")
    st.write(f"**Search Volume:** {int(pred_data[0, 0])}")
    st.write(f"**CPC:** ${round(pred_data[0, 1], 2)}")
    st.write(f"**Keyword Difficulty:** {int(pred_kd[0])}")
    
    # Simulated keyword suggestions
    suggestions = pd.DataFrame({
        'Keyword': [f"{keyword_input} strategy", f"{keyword_input} tools", f"best {keyword_input}"],
        'Search Volume': np.random.randint(50000, 500000, size=3),
        'CPC': np.round(np.random.uniform(0.5, 3.0, size=3), 2),
        'Keyword Difficulty': np.random.randint(30, 90, size=3)
    })
    st.write("### Suggested Keywords:")
    st.dataframe(suggestions)

In which online platform it will execute 




