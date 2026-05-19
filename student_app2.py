# student_app.py
import streamlit as st
import ollama
import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, TransferFunction, step, impulse
import cv2 
import sympy as sp
import fitz  # PyMuPDF
from PIL import Image
import base64

# --- PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="Oasis Educational Platform", page_icon="🎓", layout="wide")

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# Load Watermark (Update this path if needed)
img_base64 = get_base64_of_bin_file("C:\\Users\\SitaSrinivas\\Downloads\\image (9).png")

# --- CUSTOM DARK MODE THEME & WATERMARK ---
if img_base64:
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: transparent;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: url("data:image/png;base64,{img_base64}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: 40%; /* Adjust logo size */
            filter: brightness(0.5) opacity(0.15); /* Perfect watermark transparency */
            pointer-events: none;
            z-index: -1;
        }}
        </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
        .block-container { padding-top: 3rem; padding-bottom: 1rem; }
        html, body, [class*="css"] { font-size: 13px !important; }
        h1 { font-size: 1.5rem !important; padding-bottom: 0px !important;}
        h3 { font-size: 1.1rem !important; margin-bottom: 0px !important;}
        .specs-table { margin-left: auto; margin-right: auto; width: 100%; text-align: left; border-collapse: collapse; }
        .specs-table th, .specs-table td { border: 1px solid #555; padding: 8px; }
        .specs-table th { background-color: #222; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
# DSP Sandbox States
if "sandbox" not in st.session_state: st.session_state.sandbox = {"np": np, "pd": pd, "plt": plt, "cv2": cv2, "sp": sp, "butter": butter, "filtfilt": filtfilt}
if "messages_ws" not in st.session_state: st.session_state.messages_ws = []
if "params" not in st.session_state: st.session_state.params = []
if "review" not in st.session_state: st.session_state.review = ""
if "last_code" not in st.session_state: st.session_state.last_code = ""
if "exports" not in st.session_state: st.session_state.exports = []
if "can_export_firmware" not in st.session_state: st.session_state.can_export_firmware = False
if "trigger_prompt" not in st.session_state: st.session_state.trigger_prompt = None

# Tutor States
if "messages_tutor" not in st.session_state: st.session_state.messages_tutor = []
if "final_answer" not in st.session_state: st.session_state.final_answer = None
if "rag_results" not in st.session_state: st.session_state.rag_results = []
if "is_exact_match" not in st.session_state: st.session_state.is_exact_match = False

# --- SIDEBAR & FILE HANDLING ---
with st.sidebar:
    st.markdown("### 🔀 Select Platform Mode")
    app_mode = st.radio("Mode:", ["🛠️ Workspace (DSP Sandbox)", "🎓 GATE Tutor (Autonomous AI)"])
    
    st.markdown("### ⚙️ AI Engine")
    model_choice = st.selectbox("Select Inference Model:", ["gemma4:e4b (Fast)", "gemma4:26b (Accurate)"])
    active_model = "batiai/gemma4-e4b:q4" if "e4b" in model_choice else "gemma4:26b"
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV/Document/Image", type=["csv", "png", "jpg", "pdf"])
    
    if uploaded_file:
        save_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
        
        try:
            if save_path.lower().endswith('.pdf'):
                doc = fitz.open(save_path)
                page = doc.load_page(0) 
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1)) 
                img_path = save_path.replace('.pdf', '.png')
                pix.save(img_path)
                st.session_state["file_path"] = img_path 
                st.success("PDF Loaded and Optimized!")
            elif save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(save_path)
                img.thumbnail((800, 800)) 
                img.save(save_path)
                st.session_state["file_path"] = save_path
                st.success("Image Loaded and Optimized!")
            else:
                st.session_state["file_path"] = save_path
                st.success("CSV Data Loaded!")
                df_prev = pd.read_csv(save_path)
                st.write(f"📊 **Rows:** {len(df_prev)} | **Cols:** {len(df_prev.columns)}")
                st.dataframe(df_prev.head(3))
        except Exception as e:
            st.error(f"Error processing file: {e}")
        
    if "file_path" in st.session_state and st.sidebar.button("🗑️ Remove File", use_container_width=True):
        del st.session_state["file_path"]
        st.rerun()

# ==========================================
# MODE 1: 🛠️ WORKSPACE (Interactive DSP Lab)
# ==========================================
if app_mode == "🛠️ Workspace (DSP Sandbox)":
    
    sys_prompt_dsp = """You are an Advanced DSP Python Engine.
    MANDATORY OUTPUT STRUCTURE:
    You MUST output your response exactly in this order. Do not skip the [PARAM] tags!
    
    1. UI PARAMETERS (Extract the tuning variables the user might want to change):
    [PARAM: Cutoff (Hz) | 60 | 10~500]
    [PARAM: Filter Type | Lowpass | Lowpass, Highpass, Bandpass]
    
    2. PYTHON CODE (Use pandas, scipy, matplotlib):
    ```python
    import matplotlib.pyplot as plt
    plt.clf()
    # ... your dsp code ...
    plt.savefig('output.png', bbox_inches='tight')
    ```
    """
    if not st.session_state.messages_ws: st.session_state.messages_ws = [{"role": "system", "content": sys_prompt_dsp}]

    st.markdown("<h1>🛠️ Interactive DSP Workspace</h1>", unsafe_allow_html=True)
    col_main, col_right = st.columns([7.5, 2.5])
    user_input = st.chat_input("E.g., Apply a 60Hz low pass filter to the data...")
    active_prompt = st.session_state.trigger_prompt or user_input

    with col_main:
        status_placeholder = st.empty()
        plot_placeholder = st.empty()
        
        if os.path.exists("output.png") and not active_prompt:
            plot_placeholder.image("output.png", use_container_width=True)
            
        if st.session_state.last_code and not active_prompt:
            with st.expander("🧑‍💻 View & Edit Python Sandbox Code"):
                edited_code = st.text_area("Modify code:", value=st.session_state.last_code, height=200)
                if st.button("▶️ Run Edited Code", type="primary"):
                    try:
                        if os.path.exists("output.png"): os.remove("output.png")
                        exec(edited_code, st.session_state.sandbox)
                        st.session_state.last_code = edited_code
                        st.rerun()
                    except Exception as e:
                        st.error(f"Manual Error: {e}")

    # --- DSP EXECUTION ENGINE ---
    if active_prompt:
        st.session_state.trigger_prompt = None
        plot_placeholder.empty()
        
        # INJECT CSV DATA FOR SPEED AND ACCURACY
        prompt_with_context = active_prompt
        if "file_path" in st.session_state and st.session_state["file_path"].endswith('.csv'):
            try:
                df = pd.read_csv(st.session_state["file_path"])
                prompt_with_context += f"\n\n[DATA CONTEXT: The user uploaded a CSV. First 3 rows:\n{df.head(3).to_string()}\nUse pd.read_csv('{st.session_state['file_path']}')]"
            except: pass

        st.session_state.messages_ws.append({"role": "user", "content": prompt_with_context})
        
        with col_main:
            start_time = time.time()
            full_response = ""
            with st.spinner("🧠 AI Engine Computing (Fast Mode)..."):
                # SPEED OPTIMIZATIONS IN OPTIONS
                for chunk in ollama.chat(
                    model=active_model, 
                    messages=st.session_state.messages_ws, 
                    stream=True,
                    options={"num_ctx": 1500, "num_predict": 500, "temperature": 0.1, "num_thread": 4}
                ):
                    full_response += chunk['message']['content']
            
            st.session_state.messages_ws.append({"role": "assistant", "content": full_response})
            
            # --- SMART SLIDER & DROPDOWN PARSER ---
            param_matches = re.findall(r"\[?PARAM:\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\]?", full_response, re.IGNORECASE)
            if param_matches:
                parsed_params = []
                for p in param_matches:
                    name = p[0].strip()
                    curr = p[1].strip()
                    opts_raw = p[2].strip()
                    
                    if "~" in opts_raw: # It's a Slider
                        try:
                            min_val, max_val = map(float, opts_raw.split("~"))
                            parsed_params.append({"type": "slider", "name": name, "current": float(curr), "min": min_val, "max": max_val})
                        except: pass
                    else: # It's a Dropdown
                        opts_list = [o.strip() for o in opts_raw.split(",")]
                        if curr not in opts_list: opts_list.insert(0, curr)
                        parsed_params.append({"type": "dropdown", "name": name, "current": curr, "options": opts_list})
                st.session_state.params = parsed_params

            code_match = re.search(r"```python(.*?)```", full_response, re.DOTALL | re.IGNORECASE)
            if code_match:
                python_code = code_match.group(1).strip()
                st.session_state.last_code = python_code 
                try:
                    if os.path.exists("output.png"): os.remove("output.png")
                    exec(python_code, st.session_state.sandbox)
                    status_placeholder.success(f"✅ Plotted in {time.time() - start_time:.2f}s.")
                    time.sleep(1)
                    status_placeholder.empty()
                except Exception as e:
                    status_placeholder.error(f"⚠️ Sandbox Error: {e}")
            st.rerun()

    # --- DSP CONTROL PANEL (SLIDERS & DROPDOWNS) ---
    if st.session_state.params and not active_prompt:
        with col_right:
            st.markdown("### 🎛️ Tune Settings")
            new_values = {}
            for i, p in enumerate(st.session_state.params):
                if p['type'] == 'slider':
                    new_val = st.slider(p['name'], min_value=float(p['min']), max_value=float(p['max']), value=float(p['current']), key=f"sl_{i}")
                    new_values[p['name']] = new_val
                else:
                    new_val = st.selectbox(p['name'], options=p['options'], index=p['options'].index(p['current']), key=f"sel_{i}")
                    new_values[p['name']] = new_val
                    
            if st.button("🔄 Apply Changes", use_container_width=True, type="primary"):
                changes = [f"Change {p['name']} to {new_values[p['name']]}" for p in st.session_state.params if str(new_values[p['name']]) != str(p['current'])]
                if changes:
                    st.session_state.trigger_prompt = "Update: " + " and ".join(changes) + ". Output new code and [PARAM] tags."
                    st.rerun()


# ==========================================
# MODE 2: 🎓 GATE TUTOR (Autonomous RAG)
# ==========================================
elif app_mode == "🎓 GATE Tutor (Autonomous AI)":
    
    # RAG DATABASE IS ONLY LOADED HERE TO SAVE MEMORY!
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    @st.cache_resource
    def get_rag():
        try:
            return FAISS.load_local("gate_faiss_index", HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
        except: return None
    vectorstore = get_rag()

    st.markdown("<h1>🎓 GATE ECE Educator</h1>", unsafe_allow_html=True)
    
    sys_prompt_tutor = """You are an expert GATE ECE Tutor.
    Read the user's image. You MUST format your response EXACTLY in this order:
    [TOPIC: Write 1-2 words identifying the core subject]
    [TRANSCRIBED: Write out the actual text/equation of the question]
    [REASONING: Write your step-by-step mathematical reasoning]
    [ANSWER: Option X or final value]
    """
    if not st.session_state.messages_tutor: st.session_state.messages_tutor = [{"role": "system", "content": sys_prompt_tutor}]

    col_left, col_right = st.columns([4, 6])
    
    with col_left:
        st.markdown("### 📄 Uploaded Question")
        if "file_path" in st.session_state and st.session_state["file_path"].endswith(('.png', '.jpg')):
            st.image(st.session_state["file_path"], use_container_width=True)
        else:
            st.info("👈 Upload a PDF or screenshot of a GATE question.")
            
        st.markdown("---")
        st.markdown("### 📊 Autonomous Analytics")
        if st.session_state.rag_results:
            if st.session_state.is_exact_match:
                st.markdown("<div style='background-color:#e3f2fd; padding:10px; border-left:5px solid #1565c0;'>🚨 PREVIOUS YEAR QUESTION DETECTED!</div><br>", unsafe_allow_html=True)
            top_match = st.session_state.rag_results[0][0].metadata 
            st.write(f"**AI Detected Topic:** {st.session_state.detected_topic}")
            st.write(f"**Subject:** {top_match.get('subject', 'N/A')}")
            
            st.markdown("---")
            st.markdown("**📚 Related Past Questions:**")
            for doc, score in st.session_state.rag_results:
                meta = doc.metadata
                with st.expander(f"GATE {meta.get('year', 'N/A')} - {meta.get('topic', 'N/A')}"):
                    st.markdown(f"**Question:** {meta.get('question_text', 'N/A')}")
                    st.markdown(f"**Solution:** {meta.get('detailed_solution', 'N/A')}")
        else:
            st.caption("Analytics will appear automatically after the AI analyzes the question.")

    with col_right:
        st.markdown(f"### 🧠 AI Reasoning Engine ({active_model})")
        for msg in st.session_state.messages_tutor:
            if msg["role"] == "assistant":
                clean_text = re.sub(r"\[?TOPIC:\s*(.*?)\]?", "", msg["content"], flags=re.IGNORECASE)
                clean_text = re.sub(r"\[?TRANSCRIBED:\s*(.*?)\]?", "", clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r"\[?REASONING:\s*", "", clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r"\[?ANSWER:\s*(.*?)\]?", "", clean_text, flags=re.IGNORECASE)
                st.markdown(clean_text.strip())
                
        if st.session_state.final_answer:
            st.markdown(f"<div style='background-color:#e8f5e9; padding:15px; border-radius:8px; text-align:center; font-size:24px; color:#2e7d32; font-weight:bold;'>✅ Final Answer: {st.session_state.final_answer}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Click to solve...", key="tutor_input")

    if user_input:
        user_msg = {"role": "user", "content": user_input}
        if "file_path" in st.session_state and st.session_state["file_path"].endswith(('.png', '.jpg')):
            user_msg["images"] = [st.session_state["file_path"]]
            
        st.session_state.messages_tutor.append(user_msg)
        
        with col_right:
            status = st.empty()
            start_time = time.time()
            full_response = ""
            
            with st.spinner("👁️ AI is digesting the image and calculating..."):
                for chunk in ollama.chat(model=active_model, messages=st.session_state.messages_tutor, stream=True, options={"num_ctx": 3072, "num_predict": 250, "temperature": 0.1, "num_thread": 4}):
                    full_response += chunk['message']['content']
                    status.markdown(f"**⏱️ {time.time() - start_time:.1f}s** \n\n" + re.sub(r"\[.*?\]", "", full_response) + "▌")
            status.empty()
            
            # Extract tags for RAG
            topic_match = re.search(r"\[?TOPIC:\s*(.*?)\]?", full_response, re.IGNORECASE)
            st.session_state.detected_topic = topic_match.group(1).strip() if topic_match else "Unknown Topic"
            ans_match = re.search(r"\[?ANSWER:\s*(.*?)\]?", full_response, re.IGNORECASE)
            st.session_state.final_answer = ans_match.group(1).strip() if ans_match else None
            
            # Execute FAISS RAG Search
            if vectorstore:
                results = vectorstore.similarity_search_with_score(st.session_state.detected_topic, k=3)
                st.session_state.rag_results = results
                st.session_state.is_exact_match = (results and results[0][1] < 0.6)
                
            st.session_state.messages_tutor.append({"role": "assistant", "content": full_response})
            st.rerun()