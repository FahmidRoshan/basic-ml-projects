import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_pipeline  # Import the function


def evaluate_model(model_path, df):

    model = joblib.load(model_path)

    X_test = df.drop(columns=["responded"])
    y_test = df["responded"]

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{cr}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Responded", "Responded"],
        yticklabels=["Not Responded", "Responded"],
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":

    df = pd.read_excel("data/raw/train.xlsx")

    df_encoded = preprocess_pipeline(df)

    evaluate_model("models/ensemble_classifier_model.pkl", df_encoded)
