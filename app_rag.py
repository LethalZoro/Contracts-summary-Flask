from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from operator import itemgetter
import uuid
from dotenv import load_dotenv
from datetime import datetime, timedelta
import threading
import time
import shutil
from langchain_core.runnables import RunnableLambda


load_dotenv()

app = Flask(__name__)
CORS(app)

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=300)


persist_directory = "vector_store"
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=300)

# Initialize Chroma vector store
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
) if os.path.exists(persist_directory) else Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def print_me(x):
    print(x)
    return x


def get_retriever(contract_id):
    num_docs = len(vectordb.get()["documents"])
    if num_docs == 0:
        return None  # or handle appropriately
    return vectordb.as_retriever(search_kwargs={'filter': {'contract_id': contract_id}, "k": 20})


# LLM and prompt
llm = ChatOpenAI(model="gpt-4o")

Summary_prompt = """
                History:
                {history}

                Context:
                {context}

                Role: You are a contract law expert specializing in UK construction contracts. Your audience is made up of non-technical construction professionals who are not experts in contracts.

                Task: Using the provided context, produce a detailed summary of the contract. Write in clear, simple, everyday language, and explain any technical terms so that a layperson can easily understand. Your summary should include all the essential points and explanations, and also highlight any key risks in the contract.

                Please address the following areas:

                1. Documents:
                - List all the uploaded documents (e.g. contract, order, minutes, etc.).

                2. Contract Form:
                - Identify the form of contract (e.g. JCT 2016 Design & Build, JCT Intermediate 2016, etc.).

                3. Payments:
                - Describe the payment terms in simple language.
                - Explain how long before the final due date a pay-less notice can be issued.

                4. Termination:
                - Summarize the termination clauses.
                - Explain what costs are involved if the contract is terminated.

                5. Suspension:
                - Clarify under what circumstances the works can be suspended.
                - Detail what notice is required and what information must be included in the suspension notification.
                - Explain if and how the subcontractor can charge for resuming work after a suspension, including any costs or limits.

                6. Variations:
                - Summarize the clauses related to contract variations.
                - Explain the time limit the subcontractor has to submit a variation.
                - Clarify whether the subcontractor must obtain approval before starting a variation.
                - Identify who is authorized to sign off on variations.
                - State if the subcontractor is required to proceed with variations without prior sign-off, and under what conditions a variation may be invalidated or not paid for.

                7. Day Works:
                - Describe the daywork rates and any percentage calculations.
                - Explain whether these rates include supervisors, skilled labour, unskilled labour, and plant, and note any exclusions.

                8. Extensions of Time:
                - Outline the grounds for an extension of time.
                - Summarize the key information that must be included in an Extension of Time submission.

                9. Retention:
                - State the percentage of retention held.
                - If the contract sum is provided, calculate the total retention amount.
                - Explain the duration of the defects period.
                - Clarify whether the defects period starts upon practical completion of the subcontractor’s work or upon completion of the main contract work.

                10. Adjudication:
                    - Clarify whether the subcontractor has the right to adjudication (sometimes referred to as 'smash and grab').
                    - If applicable, note whether adjudicator fees are fixed or capped.

                11. Entire Agreement Clause:
                    - Explain if the subcontractor’s tender documents are part of the contract, or if the sub-contract documents and the main contract together form the entire agreement.

                12. Programme:
                    - Based on the programme duration and contract sum, calculate the average value of work that needs to be completed each week.
                    - Confirm whether the programme is a numbered document.
                    - Identify the value of Liquidated and Ascertained Damages (LADs).
                    - Determine if the LADs exceed 1% of the agreed contract value for a maximum period of 10 weeks.

                13. Key Risks:
                    - Highlight the main risks in the contract in plain language.

                Questions:
                {question}
                """

chat_prompt = """
                Role: You are a cautious legal assistant specializing in UK construction contracts.

                History:
                {history}

                Context:
                {context}

                Core Principle: "If it's not explicitly stated in the documents, do not guess – instead, explain what is missing."
                
                2. Answer Structure:
                a) [Found in Contract]:
                    - When citing clauses, use: "Section [X] states..." followed by a plain English explanation.
                    - Explain the practical consequences using: "This means..."
                    - Bold key numbers/dates. For example: "**7-day deadline** to dispute invoices (Section 4.2)".

                b) [Not Found]:
                    - Clearly state: "This contract doesn't specify..."
                    - Optionally add: "Typically in JCT contracts..." only if the user asks.
                    - Advise: "You should request written clarification on..." when necessary.

                3. Risk Mitigation:
                - Prefix high-stakes items (e.g., penalties, short deadlines) with a 🚨 symbol.
                - For termination clauses, calculate potential costs if a contract sum is provided.

                User Safety Protocols:
                - For non-contract scenarios, respond: 
                "While this contract doesn't address [X], general practice suggests... **Consult your solicitor for your specific case.**"
                - For conflicting clauses, note:
                "Section [Y] and Section [Z] appear to overlap. **Recommend:** Ask your legal advisor to reconcile these before proceeding."

                
                Question: {question}
                """


prompt_summary = ChatPromptTemplate.from_messages([
    ("system", Summary_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


prompt_chat = ChatPromptTemplate.from_messages([
    ("system", chat_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain_summary = (
    {
        "context": RunnableLambda(
            lambda inputs: format_docs(
                get_retriever(inputs["contract_id"]).invoke(inputs["question"])
            )
        ),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt_summary
    | llm
    | StrOutputParser()
)

chain_chat = (
    {
        "context": RunnableLambda(
            lambda inputs: format_docs(
                get_retriever(inputs["contract_id"]).invoke(inputs["question"])
            )
        ),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt_chat
    | llm
    | StrOutputParser()
)

# Session management
message_histories = {}
session_creation_times = {}


def get_message_history(contract_id: str) -> ChatMessageHistory:
    if contract_id not in message_histories:
        message_histories[contract_id] = ChatMessageHistory()
        message_histories[contract_id].add_ai_message("How can I help you?")
    return message_histories[contract_id]


chain_with_history_summary = RunnableWithMessageHistory(
    chain_summary,
    get_message_history,
    input_messages_key="question",
    history_messages_key="history",
)

chain_with_history_chat = RunnableWithMessageHistory(
    chain_chat,
    get_message_history,
    input_messages_key="question",
    history_messages_key="history",
)


@app.route('/upload', methods=['POST'])
def upload_file():
    cwd = os.getcwd()
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files accepted"}), 400

    try:
        # Save and process PDF
        contract_id = str(uuid.uuid4())

        temp_dir = os.path.join("temp_uploads", contract_id)
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        print(temp_path)
        file.save(temp_path)

        # Validate file size (optional)
        if os.path.getsize(temp_path) == 0:
            return jsonify({"error": "Uploaded file is empty"}), 400

        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)

        for doc in texts:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["contract_id"] = contract_id

        # Add to a unique vector database for this contract
        # vectordb = get_vectordb(contract_id)
        vectordb.add_documents(texts)
        vectordb.persist()

        get_message_history(contract_id)  # Initialize history
        session_creation_times[contract_id] = datetime.now()

        # Generate summary
        try:
            summary = chain_with_history_summary.invoke(
                {"question": "Generate a full contract breakdown covering all sections...",
                    "contract_id": contract_id},
                config={"configurable": {"session_id": contract_id}}
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({"contract_id": contract_id, "summary": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    if not data or 'question' not in data or 'contract_id' not in data:
        return jsonify({"error": "Missing question or contract_id"}), 400

    contract_id = data['contract_id']
    # vectordb = get_vectordb(contract_id)

    # Check if the vector store is empty
    if len(vectordb.get()["documents"]) == 0:
        return jsonify({"error": "No documents found for this contract"}), 400

    try:
        response = chain_with_history_chat.invoke(
            {"question": data['question'], "contract_id": contract_id},
            config={"configurable": {"session_id": contract_id}}
        )

        return jsonify({"answer": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/contracts', methods=['GET'])
def get_contracts():
    """Endpoint to list all uploaded contracts"""
    if not message_histories:
        return jsonify({"message": "Please upload the documents so I can assist you in answering the questions related to the contract. Without the documents, I'm unable to provide specific answers or insights."}), 404

    contracts_list = [
        {
            "contract_id": contract_id,
            "summary": next((msg.content for msg in history.messages if msg.type == "ai"), "")
        }
        for contract_id, history in message_histories.items()
    ]

    return jsonify(contracts_list), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
