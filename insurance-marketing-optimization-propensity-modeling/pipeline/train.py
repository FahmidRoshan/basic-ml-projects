import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from preprocess import preprocess_pipeline


def train_model(df):

    X = df.drop(columns=["responded"])
    y = df["responded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train = y_train.values
    y_test = y_test.values

    smoteenn = SMOTEENN(random_state=42)
    X_train_mixsmapled, y_train_mixsmapled = smoteenn.fit_resample(X_train, y_train)

    rf_classifier = RandomForestClassifier(random_state=42)
    svm_classifier = SVC(kernel="rbf", probability=True, random_state=42)

    ensemble_classifier = VotingClassifier(
        estimators=[("rf", rf_classifier), ("svm", svm_classifier)], voting="hard"
    )
    ensemble_classifier.fit(X_train, y_train)

    y_pred = ensemble_classifier.predict(X_test)

    joblib.dump(ensemble_classifier, "models/ensemble_classifier_model.pkl")


if __name__ == "__main__":

    df = pd.read_excel("data/raw/train.xlsx")

    df_encoded = preprocess_pipeline(df)

    train_model(df_encoded)
