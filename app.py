import streamlit as st
import pandas as pd
import base64

# Load the evaluation report
df = pd.read_csv("evaluation_report.csv")

# Clean the dataframe
if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'class'}, inplace=True)

# Page config
st.set_page_config(page_title="Evaluation Report Viewer", layout="wide")
st.title("Cybersecurity Incident Classification - Evaluation Report")

# Show key metrics
st.markdown("## Overall Metrics Summary")
overall_metrics = df[df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])]
st.dataframe(overall_metrics.reset_index(drop=True))

# Show class-wise detailed table
st.markdown("## Class-wise Evaluation Metrics")
with st.expander("Click to view full table"):
    st.dataframe(df[~df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])].reset_index(drop=True))

# Filter by F1-Score threshold
st.markdown("### Filter by F1-Score")
f1_threshold = st.slider("Select minimum F1-Score", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
filtered_df = df[(~df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])) & (df['f1-score'] >= f1_threshold)]
st.write(f"Showing {len(filtered_df)} classes with F1-score >= {f1_threshold}")
st.dataframe(filtered_df.reset_index(drop=True))

# Download option
st.markdown("### Download Report")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Full Evaluation Report",
    data=csv,
    file_name="evaluation_report.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Project: Microsoft Cybersecurity Incident Classification")