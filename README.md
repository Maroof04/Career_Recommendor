# 💼 Career Path Predictor & Recommender System

This machine learning project predicts a user's most suitable job role and expected salary, and recommends similar career paths based on their profile. It combines classification, regression, and recommendation using models like Random Forest and K-Nearest Neighbors (KNN).

---

## 🚀 Project Overview

The project includes:

- ✅ **Classification**: Predict current job title (career path)
- 💰 **Regression**: Estimate expected salary
- 🔁 **Recommendation System**: Suggest similar job profiles using KNN
- 📊 **Custom synthetic dataset**: Created to simulate realistic career paths

---

## 🧠 Technologies Used

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`
  - `scikit-learn` (`RandomForestClassifier`, `KNN`, `OneHotEncoder`, `LabelEncoder`)
- **IDE**: Jupyter Notebook

---

## 📁 Files

| File                            | Description                                       |
|---------------------------------|---------------------------------------------------|
| `Career_Path_Predictor.ipynb`   | Jupyter Notebook with full ML pipeline           |
| `synthetic_career_dataset.csv`  | Synthetic career data used for modeling          |
| `README.md`                     | Project overview (this file)                     |

---

## 📌 Dataset Details

The dataset is **synthetically generated** and contains:

- `education_level`
- `previous_job_title`
- `skills`
- `years_experience`
- `location`
- `employment_type`
- `current_job_title` (Target for classification)
- `expected_salary` (Target for regression)

A new feature `num_skills` is also derived from the `skills` column.

---
