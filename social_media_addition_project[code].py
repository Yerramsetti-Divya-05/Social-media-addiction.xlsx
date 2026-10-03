import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np
from flask import Flask, render_template, request
import os

# ================= 1. DATA CLEANING =================
print("STEP 1: DATA CLEANING")

def clean_data():
    df = pd.read_excel("Social media addiction.xlsx")
    print(f"Original Shape: {df.shape}")

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df = df.drop(columns=['Roll no '], errors="ignore")
    df["Name "] = df["Name "].fillna("Unknown")
    df = df.drop(columns=["Self assessed addiction on social media"], errors="ignore")

    df["Hours spending on social media"] = df["Hours spending on social media"].astype(str).str.lower()
    df["Hours spending on social media"] = (
        df["Hours spending on social media"]
        .str.replace("hours", "", regex=False)
        .str.replace("hour", "", regex=False)
        .str.replace("hrs", "", regex=False)
        .str.replace("hr", "", regex=False)
        .str.strip()
    )
    df["Hours spending on social media"] = pd.to_numeric(
        df["Hours spending on social media"], errors="coerce"
    )
    df["Hours spending on social media"] = df["Hours spending on social media"].fillna(
        df["Hours spending on social media"].mean()
    )

    df["Gender"] = df["Gender"].fillna("Female")
    df["Year"] = df["Year"].fillna(df["Year"].mode()[0])
    df["Branch"] = df["Branch"].fillna(df["Branch"].mode()[0])
    df["Most used platform "] = df["Most used platform "].fillna(
        df["Most used platform "].mode()[0]
    )
    df = df.dropna(subset=["Primary purpose of spending time on social media"])

    df["Last sem CGPA"] = df["Last sem CGPA"].astype(str).str.replace(
        r'[^0-9.]', '', regex=True
    )
    df["Last sem CGPA"] = pd.to_numeric(df["Last sem CGPA"], errors="coerce")
    df["Last sem CGPA"] = df["Last sem CGPA"].fillna(df["Last sem CGPA"].median())

    df.rename(columns={
        "Does you feel restless, uncomfortable, stressed, or uneasy if they cannot use social media for some time":
        "Restless without Social media"
    }, inplace=True, errors="ignore")

    df["Restless without Social media"] = df["Restless without Social media"].fillna("No Response")
    df["Restless without Social media"] = (
        df["Restless without Social media"].astype(str).str.strip().str.title()
    )

    df = df.dropna(subset=["Effect on studies(scale 1 to 5)"])
    df["Notifications on before going to bed"] = df[
        "Notifications on before going to bed"
    ].fillna("No")
    df = df.dropna(subset=["Average_Sleep_hours"])

    df["Average_Sleep_hours"] = (
        df["Average_Sleep_hours"].astype(str)
        .str.extract(r'(\d+\.?\d*)')[0]
        .astype(float)
    )
    df["Daily study hours"] = (
        df["Daily study hours"].astype(str)
        .str.extract(r'(\d+\.?\d*)')[0]
        .astype(float)
    )

    df.to_excel("cleaned Social media addiction.xlsx", index=False)
    print("Cleaned file saved!")
    print(f"Cleaned Shape: {df.shape}")
    return df

df = clean_data()

# ================= 2. EDA - VISUALIZATION =================
print("\nSTEP 2: EDA VISUALIZATION")

df.columns = df.columns.str.strip()

platform_counts = df["Most used platform"].value_counts()
top_platforms = platform_counts.head(5)

plt.figure(figsize=(8, 5))
top_platforms.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Most used social media platform", fontsize=14)
plt.xlabel("platform")
plt.ylabel("no. of students")
plt.show()

sns.histplot(df["Hours spending on social media"], bins=10, kde=True)
plt.title("Distribution of hours spent on social media")
plt.show()

sns.scatterplot(x="Hours spending on social media", y="Last sem CGPA", data=df)
plt.title("social media hours vs cgpa")
plt.show()

# ================= 3. CHI-SQUARE TEST =================
print("\nSTEP 3: CHI-SQUARE TEST")

contingency_table = pd.crosstab(
    df["Primary purpose of spending time on social media"],
    df["Effect on studies(scale 1 to 5)"]
)
print("Contingency Table:\n", contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)
print("\nChi-square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("P-value:", p)

if p < 0.05:
    print("There is a significant relationship between primary purpose and effect on studies.")
else:
    print("No significant relationship found.")

# ================= 4. ML MODEL TRAINING =================
print("\nSTEP 4: ML MODEL TRAINING")

df["Effect_Affected"] = df["Effect on studies(scale 1 to 5)"].apply(
    lambda x: 1 if x >= 3 else 0
)

X = df[[
    "Hours spending on social media",
    "Average_Sleep_hours",
    "Daily study hours"
]]
y = df["Effect_Affected"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(model, open("logistic_model.pkl", "wb"))
print("Model saved as logistic_model.pkl")

# ================= 5. FLASK APP =================
print("\nSTEP 5: STARTING FLASK APP")

app = Flask(__name__)

if not os.path.exists("templates"):
    os.makedirs("templates")

html_code = """
<!DOCTYPE html>
<html>
<head>
    <title>Social Media Impact Predictor</title>
    <style>
        body {
            font-family: Arial;
            background: #f2f2f2;
            text-align: center;
            padding: 50px;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            width: 400px;
            margin: auto;
            box-shadow: 0 0 10px #ccc;
        }
        input {
            width: 90%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        button {
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            font-weight: bold;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Check Social Media Impact on Studies</h2>
        <form method="POST">
            <input type="number" step="0.1" name="hours"
                   placeholder="Hours on Social Media" required>
            <input type="number" step="0.1" name="sleep"
                   placeholder="Average Sleep Hours" required>
            <input type="number" step="0.1" name="study"
                   placeholder="Daily Study Hours" required>
            <button type="submit">Predict</button>
        </form>
        <div class="result">{{ prediction }}</div>
    </div>
</body>
</html>
"""

with open("templates/index.html", "w") as f:
    f.write(html_code)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        hours = float(request.form["hours"])
        sleep = float(request.form["sleep"])
        study = float(request.form["study"])

        data = np.array([[hours, sleep, study]])
        pred = model.predict(data)[0]

        prediction = (
            "Social Media IS affecting studies"
            if pred == 1
            else "Social Media is NOT affecting studies"
        )

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    print("\nFlask App Running at http://127.0.0.1:5000")
    app.run(debug=True)
