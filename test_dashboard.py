#!/usr/bin/env python3
"""Simple test version of the dashboard to debug issues."""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Alpha Seeker Test",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Alpha Seeker Dashboard Test")
    
    try:
        # Test data loading
        st.write("Testing data loading...")
        
        eval_df = pd.read_csv("data/evaluation_results.csv")
        st.success(f"✅ Evaluation data loaded: {len(eval_df)} rows")
        
        price_df = pd.read_csv("data/dataset.csv")
        st.success(f"✅ Price data loaded: {len(price_df)} rows")
        
        # Simple chart test
        st.write("Testing chart creation...")
        
        fig = px.line(eval_df.head(50), x='date', y='actual', title="Sample Chart")
        st.plotly_chart(fig)
        
        st.success("✅ Dashboard components working!")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
