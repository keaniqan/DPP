import streamlit as st
import os
from datetime import datetime, timedelta
from pathlib import Path

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize session state
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

# Custom CSS for ChatGPT-like sidebar
st.markdown("""
<style>
    /* Sidebar styling */

    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    
    /* Sidebar header styling */
    [data-testid="stSidebar"] h2 {
        color: #ececec !important;
        font-size: 1rem !important;
        padding: 0.5rem 0 !important;
        margin-bottom: 1rem !important;
    }
    
    /* File button container */
    [data-testid="stSidebar"] [data-testid="column"] {
        padding: 0 !important;
    }
    
    /* File name buttons (tertiary type) */
    [data-testid="stSidebar"] button[kind="tertiary"] {
        background-color: transparent !important;
        color: #ececec !important;
        text-align: left !important;
        font-size: 0.75rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    [data-testid="stSidebar"] button[kind="tertiary"] p {
        font-size: 0.75rem !important;

    }
    
    [data-testid="stSidebar"] button[kind="tertiary"]:hover {
        background-color: #2a2b32 !important;
        border-color: transparent !important;
    }
    
    [data-testid="stSidebar"] button[kind="tertiary"]:focus {
        background-color: #2a2b32 !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] button[kind="tertiary"]:active {
        background-color: #3a3b42 !important;
    }
    
    /* Triple dot menu button (popover) - hide by default */
    [data-testid="stSidebar"] [data-testid="column"]:last-child button[data-testid="baseButton-secondary"] {
        background-color: transparent !important;
        border: none !important;
        color: #8e8ea0 !important;
        font-size: 1.5rem !important;
        padding: 0.25rem 0.5rem !important;
        min-height: 0 !important;
        transition: all 0.2s ease !important;
        opacity: 0 !important;
        line-height: 1 !important;
    }
    
    /* Show triple dot button on row hover */
    [data-testid="stSidebar"] [data-testid="column"]:last-child:hover button[data-testid="baseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover [data-testid="column"]:last-child button[data-testid="baseButton-secondary"] {
        opacity: 1 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="column"]:last-child button[data-testid="baseButton-secondary"]:hover {
        color: #ececec !important;
        background-color: #2a2b32 !important;
        border-radius: 0.375rem !important;
    }
    
    /* Popover menu styling */

    /* Delete button inside popover */
    [data-testid="stPopover"] button {
        background-color: transparent !important;
        color: #ececec !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        text-align: left !important;
        font-size: 0.875rem !important;
        border-radius: 0.375rem !important;
        transition: all 0.2s ease !important;
    }
    

    /* Container for file rows */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        gap: 0.25rem !important;
        align-items: center !important;
        margin-bottom: 0.25rem !important;
    }
    
    
    /* Info messages */
    [data-testid="stSidebar"] .stAlert {
        background-color: #2a2b32 !important;
        color: #8e8ea0 !important;
        border: 1px solid #4d4d4f !important;
    }
</style>
""", unsafe_allow_html=True)

# Main content area
st.title("Evaluation Metrics")
uploaded_file = st.file_uploader("Upload the dataset files", accept_multiple_files=True)

if uploaded_file:
    for file in uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"File '{file.name}' uploaded successfully.")

# Display selected file content
if st.session_state.selected_file:
    st.subheader(f"📄 {st.session_state.selected_file}")
    file_path = os.path.join(UPLOAD_FOLDER, st.session_state.selected_file)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            st.text_area("File Content", content, height=400)

# Sidebar - File Management System
with st.sidebar:
    st.header("Uploaded Files")
    
    # Get all files from upload folder
    if os.path.exists(UPLOAD_FOLDER):
        all_files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                all_files.append({
                    'name': filename,
                    'path': filepath,
                })
        
        # Sort by modified time (newest first)
        all_files.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
        
        # Display files
        for file_info in all_files:
            # Use a container for each row to enable hover detection
            with st.container():
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # Create a clickable file item
                    if st.button(f"{file_info['name']}", key=f"file_{file_info['name']}", use_container_width=True, type="tertiary"):
                        st.session_state.selected_file = file_info['name']
                        st.rerun()
                
                with col2:
                    # Triple dot menu button
                    with st.popover("", use_container_width=True):
                        if st.button("🗑️ Delete", key=f"del_{file_info['name']}", use_container_width=True):
                            os.remove(file_info['path'])
                            if st.session_state.selected_file == file_info['name']:
                                st.session_state.selected_file = None
                            st.rerun()
        
        if not all_files:
            st.info("No files found. Upload files to get started.")
    else:
        st.info("Upload folder not found.")