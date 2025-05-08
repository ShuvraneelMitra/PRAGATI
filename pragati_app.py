import streamlit as st
import os
import time
import threading
from datetime import datetime
import tempfile
import sys
from pragati import PRAGATI_pipeline
from agents.schemas import Paper
from agents.states import PaperState, QuestionState, FactCheckerState, TokenTracker

st.set_page_config(
    page_title="PRAGATI Document Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTextArea textarea {
        height: 300px;
    }
    .log-container {
        background-color: #1e1e1e;
        color: #00ff00;
        border-radius: 5px;
        padding: 10px;
        max-height: 150px;
        overflow-y: auto;
        font-family: monospace;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .result-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
    }
    .status-running {
        color: #FF9800;
        font-weight: bold;
    }
    .status-complete {
        color: #4CAF50;
        font-weight: bold;
    }
    /* Auto-scroll for logs */
    .log-container {
        scroll-behavior: smooth;
    }
    </style>
""", unsafe_allow_html=True)

def read_logs():
    try:
        with open("PRAGATI.log", "r") as log_file:
            return log_file.read()
    except FileNotFoundError:
        return "Log file not found. Analysis hasn't started yet."

def append_log(message):
    with open("PRAGATI.log", "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_file.write(f"{timestamp} - {message}\n")
        log_file.flush()  # Ensure the write is flushed to disk

def run_analysis(uploaded_files, topic, paper_title, log_callback=None):
    start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        if log_callback:
            log_callback(f"INFO - Starting to process {len(uploaded_files)} files")
        
        for uploaded_file in uploaded_files:
            if log_callback:
                log_callback(f"INFO - Processing file: {uploaded_file.name}")
            
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
            
            if log_callback:
                log_callback(f"INFO - Successfully saved file: {uploaded_file.name}")
        
        main_file_path = file_paths[0] if file_paths else ""
        
        if log_callback:
            log_callback(f"INFO - Creating Paper object from {os.path.basename(main_file_path)}")

        paper = Paper(
            filepath=main_file_path,
            title=paper_title,
            topic=topic,
            sections=["Introduction", "Methodology", "Conclusion"],  # Default sections
        )

        if log_callback:
            log_callback("INFO - Initializing PRAGATI states")
        
        qa_input = QuestionState(
            messages=[],
            paper=paper,
            num_reviewers=1,
            token_usage=TokenTracker(
                net_input_tokens=0, net_output_tokens=0, net_tokens=0
            ),
            reviewers=[],
            queries=[],
        )

        fc_input = FactCheckerState(paper=paper)

        state = PaperState(
            qa_results=qa_input,
            fact_checker_results=fc_input
        )

        try:
            if log_callback:
                log_callback("INFO - Starting PRAGATI pipeline execution")
 
            pragati_graph = PRAGATI_pipeline()
            
            if log_callback:
                log_callback("INFO - Invoking PRAGATI pipeline")
            
            final_state = pragati_graph.invoke(state, {"recursion_limit": 1000})
            
            if log_callback:
                log_callback("INFO - Pipeline execution complete")
            
            paper_state = PaperState(**final_state)

            factual = paper_state.overall_assesment.factual
            fact_checker_score = paper_state.overall_assesment.fact_checker_score
            publishability = paper_state.overall_assesment.Publishability
            suggestions = paper_state.overall_assesment.Suggestions
            
            if log_callback:
                log_callback(f"INFO - Results generated: Factual={factual}, Score={fact_checker_score}")
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR - An error occurred during analysis: {str(e)}")

            factual = None
            fact_checker_score = None
            publishability = None
            suggestions = "The execution of the pipeline failed. Please check the logs for more details."
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    if log_callback:
        log_callback(f"INFO - Analysis completed in {minutes} minutes {seconds} seconds")
    
    return {
        "factual": factual,
        "fact_checker_score": fact_checker_score,
        "publishability": publishability,
        "suggestions": suggestions,
        "analysis_time": f"{minutes} minutes {seconds} seconds"
    }

if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'log_position' not in st.session_state:
    st.session_state.log_position = 0
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

st.title("PRAGATI - Paper Review and Analysis Tool")
st.write("Upload your paper files, provide topic and title information, and get a comprehensive analysis.")

with st.form("analysis_form"):
    uploaded_files = st.file_uploader("Upload Paper File(s)", accept_multiple_files=True, type=["pdf", "docx", "txt"])
    topic = st.text_input("Paper Topic (e.g., ML and Time Series)", value="ML and Time Series")
    paper_title = st.text_input("Paper Title", value="")
    submit_button = st.form_submit_button("Start Analysis")

def analysis_thread(uploaded_files, topic, paper_title):
    with open("PRAGATI.log", "w") as f:
        pass
    append_log(f"INFO - Starting analysis for '{paper_title}' on topic '{topic}'")
    append_log(f"INFO - Processing file(s): {', '.join([f.name for f in uploaded_files])}")
    results = run_analysis(uploaded_files, topic, paper_title, append_log)
    st.session_state.results = results
    st.session_state.analysis_complete = True
    st.session_state.auto_refresh = False

if submit_button and not st.session_state.processing:
    if not uploaded_files:
        st.error("Please upload at least one file.")
    elif not paper_title:
        st.error("Please provide a paper title.")
    else:
        st.session_state.processing = True
        st.session_state.results = None
        st.session_state.log_position = 0
        st.session_state.analysis_complete = False
        st.session_state.auto_refresh = True
        thread = threading.Thread(
            target=analysis_thread,
            args=(uploaded_files, topic, paper_title)
        )
        thread.daemon = True
        thread.start()

if st.session_state.processing:
    if not st.session_state.analysis_complete:
        st.markdown("<p class='status-running'>⏳ Analysis in progress...</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='status-complete'>✅ Analysis complete!</p>", unsafe_allow_html=True)
    
    st.subheader("Processing Logs")
    logs_container = st.container()
    results_placeholder = st.empty()
    
    with logs_container:
        logs = read_logs()
        st.markdown(f"<div class='log-container' id='log-display'>{logs}</div>", unsafe_allow_html=True)

        st.markdown("""
            <script>
                const logContainer = document.querySelector('#log-display');
                if (logContainer) {
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
            </script>
            """, unsafe_allow_html=True)

    if st.session_state.analysis_complete and st.session_state.results is not None:
        results = st.session_state.results
        with results_placeholder.container():
            st.subheader("Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown(f"**Factual Assessment:** {results['factual']}")
                st.markdown(f"**Fact Checker Score:** {results['fact_checker_score']}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown(f"**Publishability:** {results['publishability']}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("**Suggestions:**")
            st.write(results['suggestions'])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.success(f"Analysis completed in {results['analysis_time']}")

            if st.button("Start New Analysis"):
                st.session_state.processing = False
                st.session_state.results = None
                st.session_state.analysis_complete = False
                st.rerun()
else:
    st.info("Upload your paper files and provide the required information to start the analysis.")

if st.session_state.processing and not st.session_state.analysis_complete and st.session_state.auto_refresh:
    time.sleep(1) 
    st.rerun()

st.markdown("---")
st.caption("PRAGATI - Paper Review and Analysis Tool - © 2025")