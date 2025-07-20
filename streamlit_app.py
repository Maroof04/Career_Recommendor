
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neighbors import NearestNeighbors

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_career_dataset.csv")
    return df

# Preprocess data and fit recommender
def prepare_recommender(df):
    df["num_skills"] = df["skills"].apply(lambda x: len(x.split(", ")))

    categorical_cols = ["education_level", "previous_job_title", "location", "employment_type"]
    ohe = OneHotEncoder(sparse_output=False)
    ohe_encoded = ohe.fit_transform(df[categorical_cols])
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(categorical_cols))

    X = pd.concat([df[["num_skills", "years_experience"]].reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    model = NearestNeighbors(n_neighbors=4, metric='euclidean')
    model.fit(X)
    return model, X

# Recommend similar jobs
def recommend_jobs(index, df, model, X):
    distances, indices = model.kneighbors([X.iloc[index]])
    recommendations = []
    for i in indices[0][1:]:  # Skip the first (itself)
        job = {
            "Suggested Job": df.iloc[i]["current_job_title"],
            "Skills": df.iloc[i]["skills"],
            "Education": df.iloc[i]["education_level"],
            "Experience (yrs)": df.iloc[i]["years_experience"]
        }
        recommendations.append(job)
    return recommendations

# Streamlit App UI
st.title("🔍 Career Recommender System")
st.write("Upload your career dataset or use the default synthetic one to get job recommendations.")

df = load_data()
st.success("Dataset loaded successfully!")

st.subheader("🎯 Choose a user index to get job recommendations")
selected_index = st.number_input("Enter a user index (0 to {}):".format(len(df)-1), min_value=0, max_value=len(df)-1, step=1)

model, features = prepare_recommender(df)
recommendations = recommend_jobs(selected_index, df, model, features)

st.subheader("🔁 Recommended Career Paths")
for rec in recommendations:
    st.markdown(f"**{rec['Suggested Job']}**")
    st.write(f"📌 Skills: {rec['Skills']}")
    st.write(f"🎓 Education: {rec['Education']}")
    st.write(f"💼 Experience: {rec['Experience (yrs)']} years")
    st.markdown("---")
