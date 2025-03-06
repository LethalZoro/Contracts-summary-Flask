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

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize components
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


def print_me(x):
    print(x)
    return x


# Dynamic k value for retriever
num_docs = len(vectordb.get()["documents"])
retriever = vectordb.as_retriever(search_kwargs={"k": min(1, num_docs)})

# LLM and prompt
llm = ChatOpenAI(model="gpt-4o")

Summary_prompt = """
                History:
                {history}
                Context:
                {context}
                Role: you are a contract law expert in the UK construction sector.
                Task: Use the following pieces of retrieved context to answer the question regarding the contract.
                Target Audience: non-technical construction professionals who do not have contract expertise
                Tone: use simple, direct, and everyday language that a layman could understand.
                Length of response:
                Identify what documents have been uploaded (e.g. contract, order, minutes, etc)
                Identify the form of contract (e.g. JCT 2016 Design & build, JCT intermediate 2016, etc)
                Payments
                - What are the payment terms
                - How long before the final due date for payment can a pay less notice be issued
                Termination
                - What are the clauses for termination
                - What costs are involved in termination
                Suspension
                - Under what circumstances can works be suspended? What notice is needed, and what needs to be included in the notification?
                - Can the subcontractor charge for resuming work after a suspension? If so, how much?
                -
                Variations
                - Summarise the clauses for variations
                - How long does the subcontractor have to submit a variation
                - Does the subcontractor have to get sign off before proceeding with a variation
                - Who can sign off variations
                - Is the subcontractor obliged to carry out variations without prior sign off or agreement from client
                - Under what circumstances can a variation be invalidated, or not paid for
                Day works
                - What are the daywork rates
                - What the daywork percentages
                - Do the day work rates include rates for supervisors, skilled labour, unskilled labour, and plant? If not, specify what is missing
                - How long does the subcontractor have to submit a variation
                Extensions of Time
                - What are the grounds for extension of time
                - Summarise what needs to be included in an Extension of Time submission
                Retention
                - What is the percentage of Retention held as a percentage?
                - If the contract sum is given, what value does this retention total?
                - How long is the defects period?
                - Does the defects period begin upon practical completion of the subcontractor works, or on practical completion of the main contract works?
                Adjudication
                - Does the subcontractor have the right to adjudication? (also known as ‘smash and grab’)
                - If yes, are the adjudicator fees fixed or capped?
                Entire Agreement Clause
                - Do the subcontractor tender documents form part of contract (or do the sub-contract docs and main contract constitute the entire agreement and supercede all others?)
                Programme
                - Given the duration of the programme and the contract sum, what is the value of work that needs to be completed on average per week?
                - Is the programme a numbered document?
                - What is the value of the Liquidated and ascertained damages (LAD's)?
                - Are the Liquidated and ascertained damages (LAD's)? greater than 1% of the agreed contract value for a maximum of 10 weeks?
                *and highlight the key risks in this contract.
                
                Questions:
                {question}
                """


chat_prompt = """ 
                History:
                {history}
                Context:
                {context}
                
                You are a legal contract assistant. 
                Use strictly the above pieces of retrieved context to answer the question.
                If information isn't in the contract, say so. 
                If the user asks any question not in the contract, 
                say so explicitly and then you can answer to the best of your knowledge.
                Also keep in mind the history of the conversation.
                
                Question:
                {question}"""


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
        "context": itemgetter("question") | retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt_summary
    | llm
    | StrOutputParser()
)


chain_chat = (
    {
        "context": itemgetter("question") | retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
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

# Background task to clean up old sessions


def cleanup_old_sessions():
    while True:
        now = datetime.now()
        for contract_id, creation_time in list(session_creation_times.items()):
            if now - creation_time > timedelta(hours=1):  # 1-hour expiration
                del message_histories[contract_id]
                del session_creation_times[contract_id]
        time.sleep(3600)  # Run every hour


# Start the cleanup task in a separate thread
threading.Thread(target=cleanup_old_sessions, daemon=True).start()

# Endpoints


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files accepted"}), 400

    # summary_prompt = request.form.get("summary_prompt")
    # # print(summary_prompt)
    # if not summary_prompt:
    #     return jsonify({"error": "Please provide a summary prompt"}), 400

    try:
        # Save and process PDF
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        # Validate file size (optional)
        if os.path.getsize(temp_path) == 0:
            return jsonify({"error": "Uploaded file is empty"}), 400

        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)

        # Add to vector store
        vectordb.add_documents(texts)
        vectordb.persist()

        # global retriever
        # retriever = vectordb.as_retriever(
        #     search_kwargs={"k": min(1, num_docs)})

        contract_id = str(uuid.uuid4())
        get_message_history(contract_id)  # Initialize history
        session_creation_times[contract_id] = datetime.now()

        # generate summary
        try:
            summary = chain_with_history_summary.invoke(
                {"question": Summary_prompt},
                config={"configurable": {"session_id": contract_id}}
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({"contract_id": contract_id,
                        "summary": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    if not data or 'question' not in data or 'contract_id' not in data:
        return jsonify({"error": "Missing question or contract_id"}), 400

    # Check if the vector store is empty
    if len(vectordb.get()["documents"]) == 0:
        return jsonify({"error": "No documents have been uploaded yet. Please upload documents first."}), 400

    try:
        response = chain_with_history_chat.invoke(
            {"question": data['question']},
            config={"configurable": {"session_id": data['contract_id']}}
        )
        return jsonify({"answer": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route('/history/<contract_id>', methods=['GET'])
# def get_history(contract_id):
#     history = message_histories.get(contract_id)
#     if not history:
#         return jsonify({"error": "Session not found"}), 404

#     messages = [{"type": msg.type, "content": msg.content}
#                 for msg in history.messages]
#     return jsonify({"messages": messages}), 200

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


# @app.route('/status', methods=['GET'])
# def get_status():
#     vector_store_info = vectordb.get()
#     return jsonify({
#         "num_documents": len(vector_store_info["documents"]),
#         "num_chunks": len(vector_store_info["embeddings"]),
#         "persist_directory": persist_directory
#     }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
