# gait_analysis_ui.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal
from io import BytesIO
import base64


COLUMN_NAMES = [
    'CurrentTimeStamps', 'Time(ms)', 'Heel(kPa)', 'M2(kPa)', 
    'Mid(kPa)', 'M1(kPa)', 'IncipMat Sensor1.1', 
    'IncipMat Sensor2.1', 'IncipMat Sensor3.1', 'IncipMat Sensor4.1'
]
# Custom CSS for medical-themed styling
MEDICAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .main {
        background-color: #f0f4f8;
    }
    
    .stButton>button {
        background-color: #2A7AB0;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1E5C87;
        color: white;
    }
    
    .uploaded-file {
        border: 2px dashed #2A7AB0;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .results-section {
        background-color: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .metric-box {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
"""

st.markdown(MEDICAL_CSS, unsafe_allow_html=True)

def process_analysis(df):
    # Modified analysis functions to return plot images instead of showing
    
    def find_threshold(data, sensor):
        hist_vals, bin_edges = np.histogram(data, bins=30)
        peaks, _ = scipy.signal.find_peaks(hist_vals)
        peak_heights = hist_vals[peaks]
        
        if len(peaks) >= 2:
            top_2_peaks = peaks[np.argsort(peak_heights)[-2:]]
            peak1, peak2 = sorted(top_2_peaks)
            valleys, _ = scipy.signal.find_peaks(-hist_vals)
            valleys_between = [v for v in valleys if peak1 < v < peak2]
            
            if valleys_between:
                threshold_idx = min(valleys_between, key=lambda v: hist_vals[v])
                threshold = bin_edges[threshold_idx]
                
                fig = plt.figure(figsize=(8, 5))
                plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
                plt.axvline(threshold, color='red', linestyle='dashed', 
                          label=f'Threshold: {threshold:.2f} kPa')
                plt.xlabel('Pressure')
                plt.ylabel('Frequency')
                plt.title(f'{sensor} Pressure Distribution')
                plt.legend()
                plt.grid(True)
                plt.close()
                return fig, threshold
        return None, None

    def process_sensor(df, sensor, threshold):
        touch_indices = []
        leave_indices = []
        in_contact = False
        
        for i in range(len(df)):
            pressure = df[sensor].iloc[i]
            if not in_contact and pressure > threshold:
                touch_indices.append(i)
                in_contact = True
            elif in_contact and pressure < threshold:
                leave_indices.append(i)
                in_contact = False
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df["Time(ms)"], df[sensor], label=f"{sensor} Pressure", 
                color="blue" if sensor == "M1(kPa)" else "black")
        plt.scatter(
            df.loc[touch_indices, "Time(ms)"],
            df.loc[touch_indices, sensor],
            color="green",
            label="Touch Points",
            zorder=5
        )
        plt.scatter(
            df.loc[leave_indices, "Time(ms)"],
            df.loc[leave_indices, sensor],
            color="red",
            label="Leave Points",
            zorder=5
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Pressure (kPa)")
        plt.title(f"{sensor} Pressure Analysis")
        plt.legend()
        plt.grid(True)
        plt.close()
        return fig, (df["Time(ms)"].iloc[leave_indices].tolist() if sensor == "M1(kPa)" 
                    else df["Time(ms)"].iloc[touch_indices].tolist())

    # Main processing
    results = {}
    
    # Process M1
    m1_fig, m1_threshold = find_threshold(df['M1(kPa)'], "M1")
    m1_plot, m1_release = process_sensor(df, "M1(kPa)", m1_threshold)
    
    # Process Heel
    heel_fig, heel_threshold = find_threshold(df['Heel(kPa)'], "Heel")
    heel_plot, heel_land = process_sensor(df, "Heel(kPa)", heel_threshold)
    
    # Calculate stance time
    stance_times = []
    cycles = min(len(m1_release), len(heel_land))
    for i in range(cycles):
        st = m1_release[i] - heel_land[i]
        stance_times.append(st)
    
    return {
        "m1_threshold": m1_threshold,
        "heel_threshold": heel_threshold,
        "stance_times": stance_times,
        "plots": {
            "m1_dist": m1_fig,
            "heel_dist": heel_fig,
            "m1_analysis": m1_plot,
            "heel_analysis": heel_plot
        }
    }

def main():
    st.title("Gait Analysis System 🦶")
    st.markdown("### Medical Motion Analytics Platform")
    
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    if not st.session_state.processed:
        with st.container():
            st.markdown('<div class="uploaded-file">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload Patient Data", 
                type=["xlsx", "csv"],
                help="Upload Excel/CSV file with sensor data"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file:
                try:
                    # Read file based on type
                    if uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    
                    # Validate and standardize columns
                    if len(df.columns) != len(COLUMN_NAMES):
                        st.error(f"Invalid file format: Expected {len(COLUMN_NAMES)} columns, got {len(df.columns)}")
                        return
                    
                    df.columns = COLUMN_NAMES  # Standardize column names
                    st.session_state.df = df
                    st.success("File uploaded and standardized successfully!")
                    
                    if st.button("🚀 Start Analysis", use_container_width=True):
                        with st.spinner("Analyzing gait patterns..."):
                            st.session_state.results = process_analysis(df)
                            st.session_state.processed = True
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    if st.session_state.processed:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## Analysis Results")
        
        # Metrics Display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("M1 Threshold", 
                     f"{st.session_state.results['m1_threshold']:.2f} kPa")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Heel Threshold", 
                     f"{st.session_state.results['heel_threshold']:.2f} kPa")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            avg_stance = np.mean(st.session_state.results['stance_times'])
            st.metric("Average Stance Time", 
                     f"{avg_stance:.2f} ms")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Plots
        st.markdown("### Pressure Distribution Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(st.session_state.results['plots']['m1_dist'])
        with col2:
            st.pyplot(st.session_state.results['plots']['heel_dist'])
        
        st.markdown("### Temporal Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(st.session_state.results['plots']['m1_analysis'])
        with col2:
            st.pyplot(st.session_state.results['plots']['heel_analysis'])
        
        st.markdown("### Stance Time Details")
        st.dataframe(
            pd.DataFrame({
                "Cycle": range(1, len(st.session_state.results['stance_times'])+1),
                "Stance Time (ms)": st.session_state.results['stance_times']
            }),
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Analyze Another Patient", use_container_width=True):
            st.session_state.processed = False
            st.rerun()

if __name__ == "__main__":
    main()