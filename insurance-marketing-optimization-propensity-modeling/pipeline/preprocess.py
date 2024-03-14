import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib


def preprocess_pipeline(df):
    # Drop columns if 'profit' is present
    if "profit" in df.columns:
        df.drop(columns=["id", "profit"], inplace=True)
    else:
        df.drop(columns=["id"], inplace=True)

    # Feature engineering to simplify the values in the feature schooling
    schooling_simplified = {
        "basic.4y": "basic",
        "basic.6y": "basic",
        "basic.9y": "basic",
        "high.school": "high.school",
        "illiterate": "illiterate",
        "professional.course": "professional.course",
        "university.degree": "university.degree",
        "unknown": "unknown",
    }
    df.loc[:, "schooling"] = df["schooling"].replace(schooling_simplified)

    # Imputation of missing values in schooling based on profession
    imputation_mapping = {
        "blue-collar": "basic",
        "self-employed": "illiterate",
        "technician": "professional.course",
        "admin.": "university.degree",
        "services": "high.school",
        "management": "university.degree",
        "retired": "unknown",
        "entrepreneur": "university.degree",
        "unknown": "unknown",
    }
    df["schooling"] = df["schooling"].combine_first(
        df["profession"].map(imputation_mapping)
    )

    # Imputation of age values based on profession
    mean_age_retired = df.loc[df["profession"] == "retired", "custAge"].mean()
    mean_age_student = df.loc[df["profession"] == "student", "custAge"].mean()
    median_age_working = df.loc[
        ~df["profession"].isin(["retired", "student"]), "custAge"
    ].median()

    df["custAge"] = np.where(
        (df["profession"] == "retired") & df["custAge"].isna(),
        mean_age_retired,
        df["custAge"],
    )
    df["custAge"] = np.where(
        (df["profession"] == "student") & df["custAge"].isna(),
        mean_age_student,
        df["custAge"],
    )
    df["custAge"] = np.where(df["custAge"].isna(), median_age_working, df["custAge"])

    # Impute random day for missing 'day_of_week' values
    df.loc[:, "day_of_week"] = df["day_of_week"].apply(
        lambda day: (
            np.random.choice(["mon", "tue", "wed", "thu", "fri"])
            if pd.isna(day)
            else day
        )
    )

    # Drop remaining rows with missing values
    df = df.dropna()

    # Mapping for profession
    profession_mapping = {
        "admin.": "working",
        "services": "working",
        "blue-collar": "working",
        "entrepreneur": "working",
        "technician": "working",
        "retired": "dependant",
        "student": "dependant",
        "unknown": "unknown",
        "unemployed": "unemployed",
        "self-employed": "working",
        "management": "working",
        "housemaid": "working",
    }
    df.loc[:, "profession"] = df["profession"].map(profession_mapping)

    # Mapping for marital
    marital_mapping = {
        "single": "Single&Divorced",
        "divorced": "Single&Divorced",
        "married": "married",
        "unknown": "Unknown",
    }
    df.loc[:, "marital"] = df["marital"].map(marital_mapping)

    # Mapping for schooling
    schooling_mapping = {
        "basic": "basic_education",
        "illiterate": "uneducated",
        "professional.course": "educated",
        "high.school": "basic_education",
        "university.degree": "educated",
        "unknown": "unknown",
    }
    df.loc[:, "schooling"] = df["schooling"].map(schooling_mapping)

    # Label encoding for 'day_of_week'
    day_mapping = {
        "mon": 1,
        "tue": 2,
        "wed": 3,
        "thu": 4,
        "fri": 5,
    }
    df.loc[:, "day_of_week"] = df["day_of_week"].map(day_mapping)

    # Label encoding for 'month'
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    df.loc[:, "month"] = df["month"].map(months)

    # Feature engineering for pdays and pmonths
    df["pdays"] = np.select(
        [(df["pdays"] == 999), (df["pdays"] <= 10), (df["pdays"] > 10)],
        ["first visit", "within 10 days", "greater than 10 days"],
        default="unknown",
    )

    df["pmonths"] = np.select(
        [(df["pmonths"] == 999), (df["pmonths"] <= 0.3), (df["pmonths"] > 0.3)],
        ["first visit", "within 3 months", "greater than 3 months"],
        default="unknown",
    )

    # One-hot encoding for categorical features
    categorical_features = [
        "loan",
        "marital",
        "schooling",
        "default",
        "housing",
        "pdays",
        "pmonths",
        "poutcome",
        "profession",
        "contact",
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Scaling numerical features
    numerical_features = [
        "custAge",
        "campaign",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
        "pastEmail",
    ]
    scaler = StandardScaler()
    df_encoded[numerical_features] = scaler.fit_transform(
        df_encoded[numerical_features]
    )

    return df_encoded


# if __name__ == "__main__":
#     preprocessing_pipeline = Pipeline(
#         [("preprocess", FunctionTransformer(func=preprocess_pipeline))]
#     )
#     joblib.dump(preprocessing_pipeline, "models/preprocessing_pipeline.pkl")
