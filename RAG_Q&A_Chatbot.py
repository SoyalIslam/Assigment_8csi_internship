import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import streamlit as st
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def load_data():
    try:
        df = pd.read_csv('Training Dataset.csv')
        df.fillna({'Loan_ID': 'Unknown', 'Gender': 'Unknown', 'Married': 'Unknown', 'Dependents': 'Unknown', 
                   'Education': 'Unknown', 'Self_Employed': 'Unknown', 'ApplicantIncome': 0, 
                   'CoapplicantIncome': 0, 'LoanAmount': 0, 'Loan_Amount_Term': 0, 
                   'Credit_History': 'Unknown', 'Property_Area': 'Unknown', 'Loan_Status': 'Unknown'}, 
                  inplace=True)
        df['text'] = df.apply(
            lambda row: f"Loan ID: {row['Loan_ID']}, Gender: {row['Gender']}, Married: {row['Married']}, "
                        f"Dependents: {row['Dependents']}, Education: {row['Education']}, "
                        f"Self Employed: {row['Self_Employed']}, Applicant Income: {row['ApplicantIncome']}, "
                        f"Coapplicant Income: {row['CoapplicantIncome']}, Loan Amount: {row['LoanAmount']}, "
                        f"Loan Amount Term: {row['Loan_Amount_Term']}, Credit History: {row['Credit_History']}, "
                        f"Property Area: {row['Property_Area']}, Loan Status: {row['Loan_Status']}", 
            axis=1
        )
        logger.debug("Dataset loaded successfully with %d rows.", len(df))
        return df
    except FileNotFoundError:
        st.error("Dataset file 'Training Dataset.csv' not found. Please ensure it is in the same directory as the script.")
        logger.error("Dataset file not found.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def setup_retrieval(df):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['text'].tolist(), convert_to_tensor=True)
        logger.debug("Retrieval model and embeddings initialized.")
        return model, embeddings
    except Exception as e:
        st.error(f"Error setting up retrieval: {str(e)}")
        logger.error(f"Error setting up retrieval: {str(e)}")
        return None, None

def retrieve_relevant_docs(query, model, embeddings, df, top_k=3, threshold=0.3):
    try:
        query_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_results = cos_scores.topk(top_k)
        relevant_docs = []
        indices = top_results.indices.cpu().numpy().tolist()
        logger.debug(f"Top indices: {indices}, Scores: {top_results.values.cpu().numpy().tolist()}")
        for idx, score in zip(indices, top_results.values):
            if idx < len(df) and score.item() > threshold:
                relevant_docs.append(df.iloc[idx].to_dict())
        logger.debug(f"Retrieved {len(relevant_docs)} relevant documents for query: {query}, max score: {max(top_results.values.cpu().numpy()) if relevant_docs else 0.0}")
        return relevant_docs, max(top_results.values.cpu().numpy()) if relevant_docs else 0.0
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        logger.error(f"Error retrieving documents: {str(e)}")
        return [], 0.0

def generate_rag_response(query, relevant_docs):
    try:
        qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
        context = " ".join([doc['text'] for doc in relevant_docs])
        result = qa_model(question=query, context=context)
        logger.debug(f"RAG response: {result['answer']}, confidence: {result['score']}")
        return result['answer'], result['score']
    except Exception as e:
        st.error(f"Error generating RAG response: {str(e)}")
        logger.error(f"Error generating RAG response: {str(e)}")
        return "Sorry, I couldn't generate a response.", 0.0

def is_dataset_related(query):
    keywords = ['loan', 'credit', 'income', 'approval', 'history', 'amount', 'status', 'applicant', 'borrower', 
                'gender', 'married', 'dependents', 'education', 'self employed', 'self employment', 'property', 
                'area', 'term', 'loan_id', 'id', 'finance', 'mortgage', 'application']
    matched_keywords = [kw for kw in keywords if kw in query.lower()]
    logger.debug(f"Query: {query}, Matched keywords: {matched_keywords}")
    return any(keyword in query.lower() for keyword in keywords)

def is_aggregation_query(query):
    aggregation_keywords = ['how many', 'count', 'number of', 'total']
    return any(keyword in query.lower() for keyword in aggregation_keywords)

def handle_aggregation_query(query, df):
    query = query.lower()
    if 'how many' in query and 'approved' in query:
        approved_count = df[df['Loan_Status'] == 'Y'].shape[0]
        logger.debug(f"Aggregation query detected: {query}, Approved loans: {approved_count}")
        return f"{approved_count} loans are approved.", 1.0
    return None, 0.0

def preprocess_query(query):
    query = query.lower().strip()
    replacements = {
        'self employment': 'self employed',
        'self-employment': 'self employed',
        'property area': 'property',
        'loan term': 'loan amount term',
        'applicant': 'applicant income',
        'co-applicant': 'coapplicant income',
        'borrower': 'applicant'
    }
    for old, new in replacements.items():
        query = query.replace(old, new)
    return query

def main():
    st.set_page_config(page_title="Loan Approval Q&A Chatbot", layout="wide")
    st.markdown("""
        <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            background-color: #f7fafc;
            margin-bottom: 16px;
        }
        .user-message {
            background-color: #bee3f8;
            padding: 10px;
            border-radius: 8px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            color: #2b6cb0;
        }
        .bot-message {
            background-color: #e2e8f0;
            padding: 10px;
            border-radius: 8px;
            margin: 8px 0;
            max-width: 80%;
            color: #2d3748;
        }
        .input-container {
            display: flex;
            gap: 8px;
            align-items: center;
            justify-content: center;
            width: 80%;
            margin-left: auto;
            margin-right: auto;
        }
        .stButton>button {
            background-color: #3182ce;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #2b6cb0;
        }
        .clear-button {
            background-color: #e53e3e;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .clear-button:hover {
            background-color: #c53030;
        }
        .response-container {
            width: 80%;
            margin-left: auto;
            margin-right: auto;
            margin-top: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Loan Approval Q&A Chatbot")
    st.markdown("Ask questions about loan approvals! Examples: 'How many loans are approved?'")

    df = load_data()
    if df is None:
        return

    model, embeddings = setup_retrieval(df)
    if model is None or embeddings is None:
        return

    with st.sidebar:
        st.header("Chat History")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message['role'] == 'User':
                    st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message"><strong>Bot:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown("No chat history yet.")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Clear Chat History", key="clear_button"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    query = st.text_input("Your question:", key="query_input", placeholder="Type your question here...")
    send_button = st.button("Send")
    st.markdown('</div>', unsafe_allow_html=True)

    if send_button and query:
        st.session_state.chat_history.append({"role": "User", "content": query})
        with st.spinner("Generating response..."):
            if query.lower().strip() in ['hi', 'hello', 'hey']:
                response = "Hello! How can I assist you with loan approvals?"
                st.session_state.chat_history.append({"role": "Bot", "content": response})
            elif is_aggregation_query(query):
                response, confidence = handle_aggregation_query(query, df)
                if response:
                    st.session_state.chat_history.append({"role": "Bot", "content": response})
                else:
                    response = "I'm focused on loan approval questions. Please ask about loan approvals."
                    st.session_state.chat_history.append({"role": "Bot", "content": response})
            else:
                query = preprocess_query(query)
                if is_dataset_related(query):
                    relevant_docs, max_score = retrieve_relevant_docs(query, model, embeddings, df)
                    if relevant_docs and max_score > 0.3:
                        response, confidence = generate_rag_response(query, relevant_docs)
                        if confidence > 0.1:
                            st.session_state.chat_history.append({"role": "Bot", "content": response})
                        else:
                            response = "I'm focused on loan approval questions. Please ask about loan approvals."
                            st.session_state.chat_history.append({"role": "Bot", "content": response})
                    else:
                        response = "I'm focused on loan approval questions. Please ask about loan approvals."
                        st.session_state.chat_history.append({"role": "Bot", "content": response})
                else:
                    response = "I'm focused on loan approval questions. Please ask about loan approvals."
                    st.session_state.chat_history.append({"role": "Bot", "content": response})
        st.markdown(f'<div class="response-container bot-message"><strong>Bot:</strong> {response}</div>', unsafe_allow_html=True)
        st.rerun()

if __name__ == "__main__":
    main()