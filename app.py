import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- 1. STUDENT MODEL FUNCTIONS ---
# This section remains unchanged.
DATA_DIR = "student_data"

def get_default_student_data():
    return {"knowledge_graph": {}}

def load_student_data(user_id):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    filepath = os.path.join(DATA_DIR, f"{user_id}.json")
    if not os.path.exists(filepath):
        default_data = get_default_student_data()
        save_student_data(user_id, default_data)
        return default_data
    try:
        with open(filepath, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return get_default_student_data()

def save_student_data(user_id, data):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    filepath = os.path.join(DATA_DIR, f"{user_id}.json")
    with open(filepath, 'w') as f: json.dump(data, f, indent=4)

def update_student_mastery(user_id, topic, is_correct):
    data = load_student_data(user_id)
    graph = data.get("knowledge_graph", {})
    if topic not in graph:
        graph[topic] = {'attempts': 0, 'correct': 0, 'mastery': 0.0}
    graph[topic]['attempts'] += 1
    if is_correct: graph[topic]['correct'] += 1
    attempts, correct = graph[topic]['attempts'], graph[topic]['correct']
    graph[topic]['mastery'] = (correct / attempts) if attempts > 0 else 0.0
    data['knowledge_graph'] = graph
    save_student_data(user_id, data)

# --- 2. RAG PIPELINE SETUP ---
# This function is now cached. When you upload a new file, its path will be different,
# and Streamlit will automatically re-run this function for the new file.
@st.cache_resource
def setup_rag_chain(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.from_documents(docs, embeddings)
        return db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error setting up RAG pipeline: {e}")
        return None

# --- 3. AGENT FUNCTIONS ---

def get_content_explanation(rag_retriever, llm, topic):
    if rag_retriever:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=rag_retriever)
        prompt = f"Explain the key concepts of '{topic}' in a clear, simple, and detailed way for a high school student. Use analogies if helpful."
        return qa_chain.run(prompt)
    return "Error: RAG retriever not available."

def get_refined_explanation(topic, previous_explanation, llm):
    prompt = f"A student found this explanation for '{topic}' unhelpful:\n---\n{previous_explanation}\n---\nPlease re-explain the same concept but use a completely different analogy or a simpler, step-by-step approach."
    response = llm.invoke(prompt)
    return response.content

def generate_quiz_question(rag_retriever, llm, topic):
    if rag_retriever:
        context_docs = rag_retriever.get_relevant_documents(f"Information about {topic}")
        context_text = " ".join([doc.page_content for doc in context_docs])
        prompt = f"Based on the provided context, create one multiple-choice question about '{topic}'. Provide 4 options (A, B, C, D) and specify the correct answer in this exact JSON format:\n{{\"question\": \"...\", \"options\": {{\"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\"}}, \"correct_answer\": \"A\"}}\n\nContext:\n---\n{context_text}\n---"
        response = llm.invoke(prompt)
        try:
            clean_response = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except (json.JSONDecodeError, TypeError): return None
    return None

def get_adaptive_recommendation(student_data):
    graph = student_data.get("knowledge_graph", {})
    if not graph: return {"action": "start", "message": "Welcome! 📚 Upload a PDF to begin your learning journey."}
    weakest_topic, min_mastery = None, 1.0
    for topic, data in graph.items():
        if data.get('mastery', 0) < min_mastery: min_mastery, weakest_topic = data.get('mastery', 0), topic
    if weakest_topic is None or min_mastery >= 0.8: return {"action": "new_topic", "message": "🎉 You've mastered all your current topics! Enter a new one to continue."}
    elif min_mastery < 0.5: return {"action": "explain", "topic": weakest_topic, "message": f"💡 Recommendation: Let's review '{weakest_topic}'. You're still building a foundation here."}
    else: return {"action": "quiz", "topic": weakest_topic, "message": f"🎯 Recommendation: You're close to mastering '{weakest_topic}'! Let's try a quiz question to solidify it."}


# --- 4. STREAMLIT APP ---
st.set_page_config(page_title="Adaptive AI Tutor", layout="wide")

api_key = os.getenv("GOOGLE_API_KEY")
if "GOOGLE_API_KEY" not in os.environ:
    st.error("🚨 Google API Key not found. Please add it to your .env file.")
    st.stop()

USER_ID = "user_123"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, convert_system_message_to_human=True)
if 'current_quiz' not in st.session_state: st.session_state.current_quiz = None
if 'explanation' not in st.session_state: st.session_state.explanation = None
if 'feedback_given' not in st.session_state: st.session_state.feedback_given = False

# --- ### NEW ### Sidebar for PDF Upload and Dashboard ---
with st.sidebar:
    st.title("📚 Course Material")
    uploaded_file = st.file_uploader("Upload your PDF textbook", type="pdf")
    
    st.title("📊 Mastery Dashboard")
    student_data = load_student_data(USER_ID)
    graph = student_data.get("knowledge_graph")
    if graph:
        df = pd.DataFrame.from_dict(graph, orient='index')
        df['topic'] = df.index
        df['mastery'] = df['mastery'].clip(0, 1)
        fig = px.bar(df, x='topic', y='mastery', title="Mastery by Topic", color='mastery', range_y=[0,1], color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Your dashboard will appear here once you start a quiz!")

#  Main Page Logic now depends on file upload ---
st.title("🎓 Adaptive AI Tutor")
st.markdown("Your personalized learning assistant, powered by Google Gemini.")

# Only proceed if a PDF has been uploaded
if uploaded_file is not None:
    # Save uploaded file to a temporary location
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Setup RAG pipeline with the uploaded file
    rag_retriever = setup_rag_chain(temp_file_path)

    if rag_retriever:
        # Display adaptive recommendation
        recommendation = get_adaptive_recommendation(student_data)
        st.info(recommendation['message'])

        # User Input
        topic_input = st.text_input("Enter a topic you want to learn or practice:", placeholder="e.g., Photosynthesis")

        col1, col2 = st.columns(2)
        if col1.button("📚 Explain Topic", use_container_width=True):
            if topic_input:
                with st.spinner(f"Generating explanation for '{topic_input}'..."):
                    st.session_state.explanation = {'topic': topic_input, 'text': get_content_explanation(rag_retriever, llm, topic_input)}
                    st.session_state.current_quiz = None
                    st.session_state.feedback_given = False
            else:
                st.warning("Please enter a topic.")

        if col2.button("❓ Quiz Me", use_container_width=True):
            if topic_input:
                with st.spinner(f"Generating a quiz for '{topic_input}'..."):
                    st.session_state.current_quiz = generate_quiz_question(rag_retriever, llm, topic_input)
                    st.session_state.explanation = None
            else:
                st.warning("Please enter a topic.")

        # Display Area for Explanation & Feedback
        if st.session_state.explanation:
            st.markdown("---")
            with st.container(border=True):
                st.subheader(f"Explanation on: {st.session_state.explanation['topic']}")
                st.markdown(st.session_state.explanation['text'])
                if not st.session_state.feedback_given:
                    f_col1, f_col2, _ = st.columns([1,1,3])
                    if f_col1.button("👍 Helpful", use_container_width=True):
                        st.session_state.feedback_given = True; st.success("Thanks for your feedback!"); st.rerun()
                    if f_col2.button("👎 Not Helpful", use_container_width=True):
                        st.session_state.feedback_given = True
                        with st.spinner("Rethinking the explanation..."):
                            st.session_state.explanation['text'] = get_refined_explanation(st.session_state.explanation['topic'], st.session_state.explanation['text'], llm)
                        st.rerun()

        # Display Area for Quiz
        if st.session_state.current_quiz:
            st.markdown("---")
            with st.container(border=True):
                quiz, topic = st.session_state.current_quiz, topic_input
                st.subheader(f"Quiz on: {topic}")
                with st.form("quiz_form"):
                    st.write(quiz['question'])
                    options = quiz['options']
                    user_answer = st.radio("Choose your answer:", options.keys(), format_func=lambda key: f"{key}: {options[key]}")
                    if st.form_submit_button("Submit Answer"):
                        is_correct = (user_answer == quiz['correct_answer'])
                        update_student_mastery(USER_ID, topic, is_correct)
                        if is_correct: st.success("Correct! Well done! 🎉")
                        else: st.error(f"Not quite. The correct answer was: {quiz['correct_answer']}: {options[quiz['correct_answer']]}")
                        st.session_state.current_quiz = None; st.rerun()
else:
    #Show this message if no file is uploaded yet
    st.warning("Please upload a PDF file in the sidebar to get started.")