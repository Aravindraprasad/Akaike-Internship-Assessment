import streamlit as st
import requests
import pandas as pd
import json
import time
import os
import subprocess
import threading
import atexit
from io import BytesIO
import base64

# Define API URL - change when deploying
API_URL = "http://localhost:8000"  # Local development
# API_URL = "https://your-huggingface-space-name.hf.space" # For production

# Global variable to hold the API server process
api_process = None

def start_api_server():
    """Start the FastAPI server as a subprocess"""
    global api_process
    # Check if the API server is already running
    try:
        response = requests.get(f"{API_URL}")
        if response.status_code == 200:
            print("FastAPI server is already running!")
            return True
    except requests.exceptions.ConnectionError:
        # Server is not running, we'll start it
        pass
    
    # Start the FastAPI server with uvicorn
    try:
        api_process = subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])
        print("FastAPI server started...")
        
        # Wait for server to be ready
        server_ready = False
        max_retries = 15
        retries = 0
        
        while not server_ready and retries < max_retries:
            try:
                response = requests.get(f"{API_URL}")
                if response.status_code == 200:
                    server_ready = True
                    print("FastAPI server is ready!")
                    return True
                else:
                    retries += 1
                    time.sleep(1)
            except requests.exceptions.ConnectionError:
                retries += 1
                time.sleep(1)
        
        if not server_ready:
            st.error("Failed to start API server. Please run 'uvicorn api:app --reload' manually.")
            return False
    except Exception as e:
        st.error(f"Error starting API server: {str(e)}")
        return False

def cleanup():
    """Clean up process on exit"""
    global api_process
    if api_process:
        print("Shutting down API server...")
        api_process.terminate()
        api_process.wait()

# Register the cleanup function to run when the program exits
atexit.register(cleanup)

def get_sample_companies():
    """Get list of sample companies from API"""
    try:
        response = requests.get(f"{API_URL}/companies")
        return response.json().get("companies", [])
    except Exception as e:
        st.error(f"Error fetching companies: {str(e)}")
        return ["Tesla", "Apple", "Microsoft"]  # Fallback options

def get_audio_html(company_name):
    """Get HTML for audio playback"""
    audio_url = f"{API_URL}/get_audio?company={company_name}"
    return f"""
    <audio controls>
        <source src="{audio_url}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """

def analyze_company(company_name):
    """Send request to API to analyze company news"""
    try:
        response = requests.post(
            f"{API_URL}/analyze", 
            json={"company_name": company_name}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def main():
    # Start API server before doing anything else
    api_started = start_api_server()
    
    if not api_started:
        st.error("Unable to start API server. Some functionality may not work.")
    
    # Set page config
    st.set_page_config(
        page_title="News Sentiment Analysis",
        page_icon="📰",
        layout="wide"
    )
    
    # Header
    st.title("📰 News Sentiment Analysis and Text-to-Speech")
    st.markdown("""
    This application extracts key details from news articles related to a company,
    performs sentiment analysis, conducts comparative analysis,
    and generates a text-to-speech output in Hindi.
    """)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Get sample companies
    companies = get_sample_companies()
    
    # Company input
    company_input_method = st.sidebar.radio(
        "Select input method:", 
        ["Dropdown", "Text Input"]
    )
    
    if company_input_method == "Dropdown":
        company_name = st.sidebar.selectbox(
            "Select a company:",
            companies
        )
    else:
        company_name = st.sidebar.text_input(
            "Enter company name:",
            "Tesla"
        )
    
    # Analyze button
    if st.sidebar.button("Analyze Company News"):
        with st.spinner(f"Analyzing news for {company_name}..."):
            # Call API
            result = analyze_company(company_name)
            
            if result:
                # Display results
                st.success(f"Analysis complete for {company_name}")
                
                # Store results in session state for display
                st.session_state.result = result
    
    # Display results if available
    if hasattr(st.session_state, 'result'):
        result = st.session_state.result
        
        if "simulated data" in result['Final Sentiment Analysis']:
            st.warning("⚠️ Using simulated data due to connection issues with news sources. The analysis shown is for demonstration purposes only.")
            
        # Company info
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header(f"Company: {result['Company']}")
            st.markdown(f"**Final Sentiment Analysis:** {result['Final Sentiment Analysis']}")
        
        with col2:
            st.subheader("Hindi Audio Summary")
            st.markdown(f"**Hindi Summary:** {result['Hindi Summary']}")
            st.markdown(get_audio_html(result['Company']), unsafe_allow_html=True)
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_dist = result["Comparative Sentiment Score"]["Sentiment Distribution"]
        
        # Create a DataFrame for the chart
        sentiment_df = pd.DataFrame({
            'Sentiment': list(sentiment_dist.keys()),
            'Count': list(sentiment_dist.values())
        })
        
        # Display chart
        st.bar_chart(sentiment_df.set_index('Sentiment'))
        
        # Articles
        st.subheader("News Articles")
        
        # Use tabs for each article
        tabs = st.tabs([f"Article {i+1}" for i in range(len(result["Articles"]))])
        
        for i, (tab, article) in enumerate(zip(tabs, result["Articles"])):
            with tab:
                st.markdown(f"### {article['Title']}")
                
                # Sentiment badge
                sentiment = article["Sentiment"]
                if sentiment == "Positive":
                    st.markdown("**Sentiment:** 🟢 Positive")
                elif sentiment == "Negative":
                    st.markdown("**Sentiment:** 🔴 Negative")
                else:
                    st.markdown("**Sentiment:** 🟡 Neutral")
                
                st.markdown("**Summary:**")
                st.markdown(article["Summary"])
                
                st.markdown("**Topics:**")
                st.markdown(", ".join(article["Topics"]))
        
        # Comparative Analysis
        st.subheader("Comparative Analysis")
        
        # Display common topics
        common_topics = result["Comparative Sentiment Score"]["Topic Overlap"].get("Common Topics", [])
        if common_topics:
            st.markdown("**Common Topics Across Articles:**")
            st.markdown(", ".join(common_topics))
        
        # Display coverage differences in an expandable section
        with st.expander("Coverage Differences"):
            coverage_diff = result["Comparative Sentiment Score"]["Coverage Differences"]
            for i, diff in enumerate(coverage_diff):
                st.markdown(f"**Comparison {i+1}:** {diff['Comparison']}")
                st.markdown(f"**Impact:** {diff['Impact']}")
                st.markdown("---")

if __name__ == "__main__":
    main()