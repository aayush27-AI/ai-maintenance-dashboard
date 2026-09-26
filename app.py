import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import joblib
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Maintenance System", page_icon="", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(10, 20, 30, 0.85), rgba(10, 20, 30, 0.9)),
                    url("https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center top;
        background-repeat: no-repeat;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 15px;
    }
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
    div[data-testid="stForm"] {
        background: rgba(20, 25, 35, 0.75);
        padding: 30px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
    }
    .status-card {
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD CONFIG ----------
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ---------- LOGIN ----------
authenticator.login()

if st.session_state['authentication_status'] is False:
    st.error('Username/password Incorrect Password!!')
elif st.session_state['authentication_status'] is None:
    st.warning('Please login!!')
elif st.session_state['authentication_status']:

    name = st.session_state['name']
    username = st.session_state['username']
    role = config['credentials']['usernames'][username]['role']

    # ---------- LOAD DATA & MODELS ----------
    @st.cache_data
    def load_data():
        return pd.read_csv('ai4i2020_cleaned.csv')

    @st.cache_resource
    def load_anomaly_model():
        return joblib.load('isolation_forest_model.pkl')

    @st.cache_resource
    def load_failure_model():
        return joblib.load('failure_prediction_model.pkl')

    df = load_data()
    anomaly_model = load_anomaly_model()
    failure_model = load_failure_model()

    features = ['air_temperature', 'process_temperature', 'rpm', 'torque', 'tool_wear']

    df['anomaly'] = anomaly_model.predict(df[features])
    df['status'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    df['failure_probability'] = failure_model.predict_proba(df[features])[:, 1]

    def get_risk_level(prob):
        if prob < 0.3:
            return 'Low'
        elif prob < 0.6:
            return 'Medium'
        else:
            return 'High'

    df['risk_level'] = df['failure_probability'].apply(get_risk_level)

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {name}")
        st.markdown(f"**Role:** `{role}`")
        st.divider()
        page = st.radio(
            "Navigate",
            ["Overview", "Machine Lookup", "Sensor Data", "Anomaly Detection", "Failure Prediction", "Trends"]
        )
        st.divider()
        authenticator.logout('Logout', 'sidebar')

    # ---------- PAGE: OVERVIEW ----------
    if page == "Overview":
        st.title(" AI Maintenance & Optimization Dashboard")
        st.caption("Real-time machine health monitoring and predictive maintenance")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Readings", len(df))
        col2.metric("Normal", (df['status'] == 'Normal').sum())
        col3.metric("Anomalies", (df['status'] == 'Anomaly').sum())
        col4.metric("High Risk", (df['risk_level'] == 'High').sum())

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Anomaly Distribution")
            fig1 = px.pie(df, names='status', hole=0.5,
                          color='status',
                          color_discrete_map={'Normal': '#2ecc71', 'Anomaly': '#e74c3c'})
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig1, width='stretch')

        with c2:
            st.subheader("Risk Level Distribution")
            fig2 = px.pie(df, names='risk_level', hole=0.5,
                          color='risk_level',
                          color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'})
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig2, width='stretch')

    # ---------- PAGE: MACHINE LOOKUP ----------
    elif page == "Machine Lookup":
        st.title("🔍 Machine Lookup")
        st.caption("Search a specific machine by its Product ID to see full details")

        search_id = st.text_input("Enter Machine / Product ID (e.g. L47183, M14865, H29424)").strip().upper()

        if search_id:
            result = df[df['product_id'].str.upper() == search_id]

            if len(result) == 0:
                st.error(f"No machine found with ID '{search_id}'. Check the ID and try again.")
            else:
                row = result.iloc[0]

                if row['risk_level'] == 'High':
                    st.markdown("""<div class="status-card" style="background:rgba(231,76,60,0.25); border:2px solid #e74c3c;">
                                🔴 RISKY — Immediate Attention Needed</div>""", unsafe_allow_html=True)
                elif row['risk_level'] == 'Medium':
                    st.markdown("""<div class="status-card" style="background:rgba(243,156,18,0.25); border:2px solid #f39c12;">
                                🟡 CAUTION — Schedule a Checkup Soon</div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="status-card" style="background:rgba(46,204,113,0.25); border:2px solid #2ecc71;">
                                🟢 SAFE — Machine is Operating Normally</div>""", unsafe_allow_html=True)

                st.divider()

                col1, col2, col3 = st.columns(3)
                col1.metric("Machine Type", row['type'])
                col2.metric("Anomaly Status", row['status'])
                col3.metric("Failure Probability", f"{row['failure_probability']*100:.1f}%")

                st.divider()
                st.subheader("Sensor Readings")
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Air Temp", f"{row['air_temperature']:.1f} K")
                s2.metric("Process Temp", f"{row['process_temperature']:.1f} K")
                s3.metric("RPM", f"{row['rpm']:.0f}")
                s4.metric("Torque", f"{row['torque']:.1f} Nm")
                s5.metric("Tool Wear", f"{row['tool_wear']:.0f} min")

                st.divider()
                st.subheader(" Recommendation")
                if row['risk_level'] == 'High':
                    st.error("This machine shows a high failure probability. Recommend immediate inspection by maintenance team.")
                elif row['risk_level'] == 'Medium':
                    st.warning("This machine shows early warning signs. Recommend scheduling a routine checkup within the next few days.")
                else:
                    st.success("This machine is operating within normal parameters. No action needed.")
        else:
            st.info(" Enter a Product ID above to see machine details")

    # ---------- PAGE: SENSOR DATA ----------
    elif page == "Sensor Data":
        st.title(" Sensor Data")
        if role == 'admin':
            st.caption(f"Showing first 1000 of {len(df)} rows")
            st.dataframe(df.head(1000), width='stretch')
        else:
            st.dataframe(
                df[df['status'] == 'Anomaly'][features + ['status', 'failure_probability', 'risk_level']],
                width='stretch'
            )

    # ---------- PAGE: ANOMALY DETECTION ----------
    elif page == "Anomaly Detection":
        st.title("Anomaly Detection")
        anomaly_df = df[df['status'] == 'Anomaly']
        st.metric("Total Anomalies Found", len(anomaly_df))
        st.dataframe(anomaly_df[features + ['status']], width='stretch')

        fig3 = px.scatter(df, x='torque', y='rpm', color='status',
                          color_discrete_map={'Normal': '#2ecc71', 'Anomaly': '#e74c3c'},
                          title="Torque vs RPM — Anomalies Highlighted")
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig3, width='stretch')

    # ---------- PAGE: FAILURE PREDICTION ----------
    elif page == "Failure Prediction":
        st.title(" Failure Prediction")
        high_risk_df = df[df['risk_level'] == 'High'][features + ['failure_probability', 'risk_level']]
        st.metric("High Risk Machines", len(high_risk_df))

        if len(high_risk_df) > 0:
            st.warning("These machines need immediate attention ")
            st.dataframe(high_risk_df, width='stretch')
        else:
            st.success("No high-risk machines right now ")

        fig4 = px.histogram(df, x='failure_probability', nbins=30,
                            title="Failure Probability Distribution")
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig4, width='stretch')

    # ---------- PAGE: TRENDS ----------
    elif page == "Trends":
        st.title("Sensor Trends")
        selected_feature = st.selectbox("Choose a sensor reading", features)
        fig5 = px.line(df.head(300), y=selected_feature, title=f"{selected_feature} over readings")
        fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig5, width='stretch')