# 💼 Career Path Predictor & Recommender System

This machine learning project predicts a user's most suitable job role and recommends similar career paths based on their profile. It combines classification and recommendation using models like Random Forest and K-Nearest Neighbors (KNN).

---

## 🚀 Live Demo

👉 Try it out here: [🌐 Streamlit App](https://careerrecommendor-ir8g4dyhlugtfqymf395kx.streamlit.app/)
---

## 📊 Project Overview

The project includes:

- ✅ **Job Role Prediction**: Predicts most suitable job using Random Forest and KNN
- 🔁 **Recommender System**: Suggests similar job profiles using KNN similarity
- 📊 **Custom synthetic dataset**: Simulates realistic career paths

---

## 🧠 Technologies Used

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`
  - `scikit-learn` (`RandomForestClassifier`, `KNN`, `OneHotEncoder`, `LabelEncoder`)
- **Frontend**: Streamlit for interactive deployment
- **IDE**: Jupyter Notebook

---

## 📁 Files

| File                            | Description                                       |
|---------------------------------|---------------------------------------------------|
| `Career_Path_Predictor.ipynb`   | Jupyter Notebook with full ML pipeline           |
| `synthetic_career_dataset.csv`  | Synthetic dataset used for training              |
| `streamlit_app.py`              | Main deployment file for Streamlit UI            |
| `README.md`                     | Project overview and documentation (this file)   |

---

## 📌 Dataset Details

The dataset includes:

- `education_level`
- `previous_job_title`
- `skills`
- `years_experience`
- `location`
- `employment_type`
- `current_job_title` (target)
- `expected_salary` (not used in current version)

Derived feature:
- `num_skills` = number of comma-separated skills listed

---

## 🔁 Recommender System

The recommender suggests **3 similar career profiles** based on a user's input using KNN on the feature space.

```python
recommend_similar_jobs(index=10)
