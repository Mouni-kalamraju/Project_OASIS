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
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convert your logo once at startup
try:
    img_base64 = get_base64_of_bin_file("C:\\Users\\SitaSrinivas\\Downloads\\image (9).png")
except:
    img_base64 = None
    

# --- CUSTOM DARK MODE THEME ---
if img_base64:
    st.markdown(f"""
        <style>
        /* 1. Apply watermark to the main container area */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-repeat: no-repeat;
            background-position: center center;
            background-size: 100%; /* Adjust this to make the logo larger or smaller */
            background-attachment: fixed;
        }}

        /* 2. Dim the logo so it's a watermark */
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/png;base64,{img_base64}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: 100%;
            filter: brightness(0.1) opacity(0.9); /* This controls the watermark look */
            pointer-events: none; /* Allows user to click through the watermark */
            z-index: 0;
        }}

        /* 3. Ensure your chat input and main content float ABOVE the watermark */
        [data-testid="stMainBlockContainer"] {{
            position: relative;
            z-index: 1;
        }}
        </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
        .block-container { padding-top: 3rem; padding-bottom: 1rem; }
        html, body, [class*="css"] { font-size: 13px !important; }
        h1 { font-size: 1.5rem !important; padding-bottom: 0px !important;}
        h3 { font-size: 1.1rem !important; margin-bottom: 0px !important;}
        .specs-table { margin-left: auto; margin-right: auto; width: 90%; text-align: center; border-collapse: collapse; }
        .specs-table th, .specs-table td { border: 1px solid #ddd; padding: 6px; }
        .specs-table th { background-color: #333; color: white; }
    </style>
""", unsafe_allow_html=True)
# --- PAGE CONFIGURATION & CUSTOM CSS ---
# st.set_page_config(page_title="Oasis Educational Platform", page_icon="🎓", layout="wide")


# --- LOAD FAISS RAG DB (FOR TUTOR MODE) ---
@st.cache_resource
def load_rag_db():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local("gate_faiss_index", embeddings, allow_dangerous_deserialization=True)
    except:
        return None
vectorstore = load_rag_db()

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
if "current_file" not in st.session_state: st.session_state.current_file = None

# Tutor States
if "messages_tutor" not in st.session_state: st.session_state.messages_tutor = []
if "final_answer" not in st.session_state: st.session_state.final_answer = None
if "rag_results" not in st.session_state: st.session_state.rag_results = []
if "is_exact_match" not in st.session_state: st.session_state.is_exact_match = False
if "detected_topic" not in st.session_state: st.session_state.detected_topic = ""

# --- SIDEBAR & FILE HANDLING ---
with st.sidebar:
    st.markdown("### 🔀 Select Platform Mode")
    app_mode = st.radio("Mode:", ["🛠️ Workspace ", "🎓 GATE Tutor (Autonomous AI)"])
    
    st.markdown("### ⚙️ AI Engine")
    model_choice = st.selectbox("Select Inference Model:", ["gemma4:e4b (Fast)", "gemma4:26b (Accurate)"])
    active_model = "batiai/gemma4-e4b:q4" if "e4b" in model_choice else "gemma4:26b"
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV/Document/Image", type=["csv", "png", "jpg", "pdf"])
    
    if uploaded_file:
        save_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
        
        # Optimize PDF / Image
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
                
                # --- INSTANT NATIVE PREVIEW (Zero AI wait time!) ---
                df_prev = pd.read_csv(save_path)
                st.write(f"📊 **Rows:** {len(df_prev)} | **Cols:** {len(df_prev.columns)}")
                st.dataframe(df_prev.head(3))
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

        # --- REMOVED THE AUTO-TRIGGER! The AI will now only think when YOU type a prompt. ---
        
    if "file_path" in st.session_state and st.sidebar.button("Remove", use_container_width=True):
        del st.session_state["file_path"]
        st.session_state.current_file = None
        st.rerun()

# ==========================================
# MODE 1: 🛠️ WORKSPACE (Interactive DSP Lab)
# ==========================================
if app_mode == "🛠️ Workspace ":
    
    sys_prompt_dsp = """You are an Advanced Data Analysis and DSP Engine. Variables persist in memory.
    RULES:
    1. Output ONLY Python code in ```python ``` blocks. Keep code fast and minimal.
    2. PLOTTING: Always start with plt.clf(). End with plt.savefig('output.png', bbox_inches='tight').
    3. PARAMETERS: Deduce tuning parameters. Format: [PARAM: Name | Current | Opt1, Opt2, Custom]
    4. REVIEW: Output [REVIEW: Your 1-sentence critique].
    5. EXPORTS: If data is processed, save it (e.g., df.to_csv('processed.csv')) and output [EXPORT: processed.csv].
    6. FIRMWARE: Output [FIRMWARE: YES] if math can be exported to C coefficients. Otherwise [FIRMWARE: NO].
    """
    if not st.session_state.messages_ws: st.session_state.messages_ws = [{"role": "system", "content": sys_prompt_dsp}]

    st.markdown("<h1>🛠️ Interactive DSP Workspace</h1>", unsafe_allow_html=True)
    col_main, col_right = st.columns([7.5, 2.5])
    user_input = st.chat_input("E.g., Apply a 50Hz notch filter to the data...")
    active_prompt = st.session_state.trigger_prompt or user_input

    # --- MAIN DSP SCREEN ---
    with col_main:
        status_placeholder = st.empty()
        plot_placeholder = st.empty()
        
        if os.path.exists("output.png") and not active_prompt:
            plot_placeholder.image("output.png", use_container_width=True)
            
        if st.session_state.params and not active_prompt:
            table_html = "<table class='specs-table'><tr><th>Parameter</th><th>Current Value</th></tr>"
            for p in st.session_state.params:
                table_html += f"<tr><td>{p['name']}</td><td><b>{p['current']}</b></td></tr>"
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)
            
        if st.session_state.review and not active_prompt:
            st.info(f"💡 **AI Profiling:** {st.session_state.review}")
            
        if st.session_state.last_code and not active_prompt:
            with st.expander("🧑‍💻 View & Edit Python Sandbox Code"):
                edited_code = st.text_area("Modify and execute sandbox code:", value=st.session_state.last_code, height=200)
                if st.button("▶️ Run Edited Code", type="primary"):
                    try:
                        if os.path.exists("output.png"): os.remove("output.png")
                        exec(edited_code, st.session_state.sandbox)
                        st.session_state.last_code = edited_code
                        st.success("Execution applied!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Manual Error: {e}")

    # --- DSP EXECUTION ENGINE ---
    if active_prompt:
        st.session_state.trigger_prompt = None
        plot_placeholder.empty()
        st.session_state.messages_ws.append({"role": "user", "content": active_prompt})
        
        with col_main:
            start_time = time.time()
            full_response = ""
            with st.spinner("🧠 AI Engine Computing..."):
                for chunk in ollama.chat(model=active_model, messages=st.session_state.messages_ws, stream=True):
                    full_response += chunk['message']['content']
            
            st.session_state.messages_ws.append({"role": "assistant", "content": full_response})
            
            # Extract Tags
            param_matches = re.findall(r"\[?PARAM:\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\]?", full_response, re.IGNORECASE)
            if param_matches:
                parsed_params = []
                seen_names = set() 
                for p in param_matches:
                    name = p[0].strip()
                    if name in seen_names: continue
                    seen_names.add(name)
                    opts = [o.strip() for o in p[2].split(",")]
                    curr = p[1].strip()
                    if "Custom" not in opts: opts.append("Custom")
                    if curr not in opts: opts.insert(0, curr)
                    parsed_params.append({"name": name, "current": curr, "options": opts})
                st.session_state.params = parsed_params
                
            review_match = re.search(r"\[?REVIEW:\s*(.*?)\]?", full_response, re.IGNORECASE)
            if review_match: st.session_state.review = review_match.group(1).strip()
                
            fw_match = re.search(r"\[?FIRMWARE:\s*(YES|NO)\]?", full_response, re.IGNORECASE)
            if fw_match: st.session_state.can_export_firmware = (fw_match.group(1).upper() == "YES")

            export_matches = re.findall(r"\[?EXPORT:\s*(.*?)\]?", full_response, re.IGNORECASE)
            st.session_state.exports = [m.strip() for m in export_matches]

            code_match = re.search(r"```python(.*?)```", full_response, re.DOTALL | re.IGNORECASE)
            if code_match:
                python_code = code_match.group(1).strip()
                st.session_state.last_code = python_code 
                status_placeholder.warning("⚙️ Executing in Stateful Sandbox...")
                try:
                    if os.path.exists("output.png"): os.remove("output.png")
                    exec(python_code, st.session_state.sandbox)
                    if os.path.exists("output.png"):
                        status_placeholder.success(f"✅ Completed in {time.time() - start_time:.2f}s.")
                    time.sleep(0.8)
                    status_placeholder.empty()
                except Exception as e:
                    status_placeholder.error(f"⚠️ Error: {e}")
            st.rerun()

    # --- DSP CONTROL PANEL (RIGHT) ---
    if st.session_state.params and not active_prompt:
        with col_right:
            st.markdown("### 🎛️ Tune Settings")
            new_values = {}
            for i, p in enumerate(st.session_state.params):
                selected = st.selectbox(p['name'], options=p['options'], index=p['options'].index(p['current']), key=f"sel_{i}_{p['name']}")
                if selected == "Custom":
                    custom_val = st.text_input(f"Enter custom {p['name']}:", key=f"txt_{i}_{p['name']}")
                    new_values[p['name']] = custom_val if custom_val else selected
                else:
                    new_values[p['name']] = selected
                    
            if st.button("🔄 Update Model", use_container_width=True, type="primary"):
                changes = [f"Change {p['name']} to {new_values[p['name']]}" for p in st.session_state.params if new_values.get(p['name']) and new_values[p['name']] != "Custom" and new_values[p['name']] != p['current']]
                if changes:
                    st.session_state.trigger_prompt = "Update: " + " and ".join(changes) + ". Use memory. Output [PARAM], [REVIEW], [EXPORT], [FIRMWARE]."
                    st.rerun()
                    
            st.markdown("---")
            st.markdown("### 💾 Workspace Exports")
            for export_file in st.session_state.exports:
                if os.path.exists(export_file):
                    with open(export_file, "rb") as f:
                        file_ext = export_file.split('.')[-1].upper()
                        st.download_button(f"📥 Download {file_ext}", data=f, file_name=export_file, use_container_width=True)
                        
            if st.session_state.can_export_firmware:
                if st.button("🚀 Gen .c Header", use_container_width=True):
                    st.session_state.trigger_prompt = "Write filter coefficients to 'firmware.h'."
                    st.rerun()
                if os.path.exists("firmware.h"):
                    with open("firmware.h", "rb") as f:
                        st.download_button("📥 Download firmware.h", data=f, file_name="firmware.h", type="primary", use_container_width=True)


# ==========================================
# MODE 2: 🎓 GATE TUTOR (Autonomous VLM + RAG)
# ==========================================
elif app_mode == "🎓 GATE Tutor (Autonomous AI)":
    st.markdown("<h1>🎓 GATE ECE Educator</h1>", unsafe_allow_html=True)
    
    sys_prompt_tutor = """You are an expert GATE ECE Tutor.
    Read the user's image. You MUST format your response EXACTLY in this order:
    [TOPIC: Write 1-2 words identifying the core subject, e.g., BJT Biasing, Nyquist Plot]
    [TRANSCRIBED: Write out the actual text/equation of the question you see in the image]
    [REASONING: Write your step-by-step mathematical reasoning in Markdown/LaTeX]
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
                st.markdown("<div class='exact-match'>🚨 PREVIOUS YEAR QUESTION DETECTED!</div>", unsafe_allow_html=True)
            top_match = st.session_state.rag_results[0][0].metadata 
            st.write(f"**AI Detected Topic:** {st.session_state.detected_topic}")
            st.write(f"**Subject:** {top_match.get('subject', 'N/A')}")
            if st.session_state.is_exact_match:
                st.write(f"**Originally Asked By:** {top_match.get('iit_institution', 'Unknown')} (GATE {top_match.get('year', 'Unknown')})")
            
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
            st.markdown(f"<div class='big-answer'>✅ Final Answer: {st.session_state.final_answer}</div>", unsafe_allow_html=True)

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
                for chunk in ollama.chat(model=active_model, messages=st.session_state.messages_tutor, stream=True):
                    full_response += chunk['message']['content']
                    elapsed = time.time() - start_time
                    display_text = re.sub(r"\[.*?\]", "", full_response) 
                    status.markdown(f"**⏱️ {elapsed:.1f}s** \n\n" + display_text + "▌")
            status.empty()
            
            topic_match = re.search(r"\[?TOPIC:\s*(.*?)\]?", full_response, re.IGNORECASE)
            st.session_state.detected_topic = topic_match.group(1).strip() if topic_match else "Unknown Topic"
            
            transcribed_match = re.search(r"\[?TRANSCRIBED:\s*(.*?)\]?", full_response, re.IGNORECASE)
            search_text = transcribed_match.group(1).strip() if transcribed_match else st.session_state.detected_topic
            
            ans_match = re.search(r"\[?ANSWER:\s*(.*?)\]?", full_response, re.IGNORECASE)
            st.session_state.final_answer = ans_match.group(1).strip() if ans_match else None
            
            if vectorstore and search_text:
                results = vectorstore.similarity_search_with_score(search_text, k=3)
                st.session_state.rag_results = results
                st.session_state.is_exact_match = False
                if results and results[0][1] < 0.6: 
                    st.session_state.is_exact_match = True
                
            st.session_state.messages_tutor.append({"role": "assistant", "content": full_response})
            st.rerun()