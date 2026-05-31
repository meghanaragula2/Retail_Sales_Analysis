from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly

import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load Dataset
df = pd.read_csv("data.csv", encoding="latin1")

# Data Preparation
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

# KPI Metrics
total_sales = round(df["Sales"].sum(), 2)
total_profit = round(df["Profit"].sum(), 2)
total_orders = len(df)

# ML Model
X = df[["Quantity", "Discount"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

sample_prediction = round(
    model.predict([[5, 0.1]])[0], 2
)

@app.route("/")
def dashboard():

    # Region Sales Chart
    region_sales = (
        df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig1 = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        title="Region-wise Sales"
    )

    graph1 = json.dumps(
        fig1,
        cls=plotly.utils.PlotlyJSONEncoder
    )

    # Category Sales Chart
    category_sales = (
        df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig2 = px.pie(
        category_sales,
        names="Category",
        values="Sales",
        title="Category Distribution"
    )

    graph2 = json.dumps(
        fig2,
        cls=plotly.utils.PlotlyJSONEncoder
    )

    # Monthly Trend
    monthly_sales = (
        df.groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig3 = px.line(
        monthly_sales,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    graph3 = json.dumps(
        fig3,
        cls=plotly.utils.PlotlyJSONEncoder
    )

    return render_template(
        "index.html",
        total_sales=total_sales,
        total_profit=total_profit,
        total_orders=total_orders,
        prediction=sample_prediction,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3
    )

if __name__ == "__main__":
    app.run(debug=True)