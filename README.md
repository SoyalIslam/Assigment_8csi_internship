
# 🤖 Loan Approval Q&A Chatbot

A **Streamlit-based chatbot** that answers questions about loan approvals using **Retrieval-Augmented Generation (RAG)**. It combines semantic search using Sentence-BERT and question answering with DistilBERT to deliver relevant answers from a structured dataset.

---

## 🧠 Features

- 💬 Ask natural questions about loan applications
- 🧭 Semantic search using SentenceTransformers
- 📊 Intelligent answers from tabular data with Hugging Face Transformers
- 🗃️ Handles both factual and aggregation-based queries
- 💻 Streamlit UI with styled chat interface
- 🧼 Clear chat history with one click

---

## 📁 Project Structure

```
📦 Loan-QA-Chatbot/
├── chatbot.py                # 🔧 Main Streamlit app
├── Training Dataset.csv      # 📊 Dataset used for retrieval
├── requirements.txt          # 📦 Project dependencies
└── README.md                 # 📄 Project documentation
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SoyalIslam/Assigment_8csi_internship.git
cd Assigment_8csi_internship
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App

```bash
streamlit run chatbot.py
```

It will open in your browser at [http://localhost:8501](http://localhost:8501).

---

## 💬 Example Questions

- "How many loans are approved?"
- "What is the credit history for loan ID LP001003?"
- "What’s the applicant income for the approved loans?"
- "Who are self-employed borrowers?"

---

## 🔍 How It Works

| Component       | Description |
|----------------|-------------|
| 📥 Data Loader  | Reads and preprocesses the loan dataset |
| 🧠 Embedding Model | `all-MiniLM-L6-v2` from SentenceTransformers |
| 🔎 Retrieval    | Finds relevant rows using cosine similarity |
| 🗣️ QA Model     | Uses `distilbert-base-uncased-distilled-squad` to generate answers |
| 📊 Aggregation  | Handles simple "how many" type queries with custom logic |

---

## 📊 Dataset Fields

The dataset includes:

- `Loan_ID`, `Gender`, `Married`, `Dependents`
- `Education`, `Self_Employed`, `ApplicantIncome`
- `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`
- `Credit_History`, `Property_Area`, `Loan_Status`

Missing values are handled gracefully using default replacements.

---

## 📦 Dependencies

Key packages (see `requirements.txt`):

- `streamlit`
- `pandas`
- `sentence-transformers`
- `transformers`
- `torch`

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🎯 Improvements Roadmap

- 📈 Visual dashboards of loan stats
- 🌐 Deploy to Streamlit Cloud or Hugging Face Spaces
- 🧠 Better model fine-tuning & custom QA logic
- 📄 Export chat logs as PDF/CSV

---

## 👤 Author

**Soyal Islam**  
Developed as part of the 8CSI Internship Program  
🔗 GitHub: [@SoyalIslam](https://github.com/SoyalIslam)

---

## 📜 License

Currently no license specified. Consider adding one like MIT or Apache-2.0.

---

> ⭐️ Found this project useful? Star the repo to support development!
