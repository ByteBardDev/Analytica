import streamlit as st
import pandas as pd
import plotly.express as px
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression
import numpy as np

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Analytica", page_icon="📊", layout="wide")

# ---- DARK THEME ----
st.markdown("""
    <style>
        body { background-color: #0E1117; color: #FFFFFF; }
        .chat-container { max-height: 400px; overflow-y: auto; }
        .user-msg { background-color: #2E86C1; color: white; padding: 8px; border-radius: 8px; margin-bottom: 5px; }
        .bot-msg { background-color: #1E222A; color: white; padding: 8px; border-radius: 8px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---- TITLE ----
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Analytica</h1>", unsafe_allow_html=True)
st.caption("Ask. Analyze. Visualize. Predict. Detect anomalies — Chat with your data like never before.")

# ---- SESSION STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data" not in st.session_state:
    st.session_state.data = None
if "context" not in st.session_state:
    st.session_state.context = {}

# ---- SIDEBAR ----
st.sidebar.header("Quick Filters")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file:
    st.session_state.data = pd.read_csv(uploaded_file)

# ---- UTILS ----
def detect_column(possible_names, data_columns):
    """Find best matching column from data"""
    match, score = process.extractOne(possible_names, data_columns)
    return match if score > 70 else None

# ---- PROCESS QUERY ----
def process_query(query):
    data = st.session_state.data
    if data is None:
        return "Please upload a CSV first."

    cols = list(data.columns)
    sales_col = detect_column("sales", cols)
    month_col = detect_column("month", cols)
    category_col = detect_column("category", cols)

    q = query.lower()

    # Follow-up handling
    if "show" in q and st.session_state.context.get("last_query"):
        if month_col:
            for m in data[month_col].astype(str).unique():
                if m.lower() in q:
                    st.session_state.context["last_query"] = {"filter_col": month_col, "filter_val": m}
                    return process_query(f"total sales of {m} month")

    try:
        # Store last query context
        st.session_state.context["last_query"] = {"filter_col": None, "filter_val": None}

        # TOTAL SALES
        if "total" in q and "sales" in q:
            if month_col and "month" in q:
                for m in data[month_col].astype(str).unique():
                    if m.lower() in q:
                        total = data[data[month_col].astype(str).str.lower() == m.lower()][sales_col].sum()
                        st.session_state.context["last_query"] = {"filter_col": month_col, "filter_val": m}
                        fig = px.bar(data[data[month_col].astype(str).str.lower() == m.lower()],
                                     x=category_col, y=sales_col, title=f"Sales Breakdown - {m}", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                        return f"Total sales for {m}: {total:,.2f}"
            total = data[sales_col].sum()
            return f"Total sales: {total:,.2f}"

        # CATEGORY-WISE SALES
        elif category_col and "category" in q and "sales" in q:
            result = data.groupby(category_col)[sales_col].sum().reset_index()
            fig = px.bar(result, x=category_col, y=sales_col, title="Category-wise Sales", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            return result.to_html(index=False)

        # TREND
        elif month_col and "trend" in q:
            result = data.groupby(month_col)[sales_col].sum().reset_index()
            fig = px.line(result, x=month_col, y=sales_col, title="Monthly Sales Trend", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            return "Here’s your monthly sales trend."

        # PREDICTION
        elif month_col and "predict" in q:
            months = pd.factorize(data[month_col])[0].reshape(-1, 1)
            sales = data[sales_col].values
            model = LinearRegression()
            model.fit(months, sales)
            next_month = np.array([[months.max() + 1]])
            prediction = model.predict(next_month)[0]
            return f"Predicted sales for next month: {prediction:,.2f}"

        # ANOMALY DETECTION
        elif "anomaly" in q or "detect" in q:
            mean = data[sales_col].mean()
            std = data[sales_col].std()
            anomalies = data[abs(data[sales_col] - mean) > 2 * std]
            if anomalies.empty:
                return "No anomalies detected in sales."
            else:
                fig = px.scatter(anomalies, x=month_col, y=sales_col, title="Anomalies in Sales", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                return f"Detected {len(anomalies)} anomalies based on sales deviations."

        else:
            return "Try queries like 'total sales of January', 'category-wise sales', 'monthly trend', 'predict next month sales', or 'detect anomalies'."

    except Exception as e:
        return f"Error: {e}"

# ---- CHAT INTERFACE ----
st.subheader("Chat with Analytica")
user_query = st.text_input("Ask a question about your data:")

if user_query:
    response = process_query(user_query)
    st.session_state.messages.append({"role": "user", "text": user_query})
    st.session_state.messages.append({"role": "bot", "text": response})

# ---- DISPLAY CHAT ----
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>You: {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>Analytica: {msg['text']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
