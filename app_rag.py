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
                Role: You're a construction contract decoder - translate complex terms into "what this actually means" advice for busy site managers.  

                **Rules of Engagement**:  
                1. **Source Priority**:  
                - FIRST use {context}  
                - THEN consider {history}  
                - NEVER invent clauses or assume standard terms 
                
                **Core Objective**:  
                Create a **worksite survival guide** that answers:  
                1. "What must I do?"  
                2. "When must I do it?"  
                3. "What happens if I don't?"  

                **User-Centric Rules**:  
                - Replace every legal term with analogies (e.g., "Retention = client holding 5% as a safety deposit")  
                - For every clause, add:  
                - ⏰ **Countdown Clock**: "You have [X] days to act after [event]"  
                - 💸 **Cost Impact**: "This could cost [£Y] if mishandled"  
                - 📝 **Paper Trail Tip**: "Always get this in writing via..."  

                **Structure Template**:  

                ### 🔍 **QUICK SNAPSHOT**  
                1. **Documents Uploaded**:  
                - [List with dates: "Contract v3 (15 Jan 2024)", "Amendment #2 (Payment Terms)"]  
                - 🚩 **Missing Critical Docs**: [Highlight any gaps like unsigned schedules]  

                2. **Contract Type**:  
                - "[JCT Intermediate 2016] with [3 custom amendments]"  
                - 🚨 **Unusual Clause Alert**: "This version allows client to terminate without cure period"  

                3. **Top 3 Red Flags**:  
                1. "Client can claim £2,800/day penalties after 24h notice"  
                2. "You bear 100% of design error costs even if approved"  
                3. "No cap on variation approval time - could delay payments"  

                ---  

                ### 📑 **SECTION-BY-SECTION BREAKDOWN**  

                #### 💷 **PAYMENTS**  
                **Key Facts**:  
                - Invoice every [4 weeks] on [Friday 5pm] via [Portal X]  
                - **Late Payments**: Client owes [8%+Bank Rate] interest after [7 days]  
                - ⚠️ **Trap**: "Final payment requires [10 documents] - start collecting now!"  

                **Pay Less Notice**:  
                - Client must dispute invoices [5 working days] before due date  
                → **Example Timeline**:  
                Invoice Date: 1 March → Due Date: 15 March → Dispute Deadline: 8 March  
                - 🚨 **Risk**: "Missing this window = they MUST pay in full"  

                ---  

                #### 🚧 **TERMINATION**  
                **When They Can Fire You**:  
                1. [14-day delay] with [3 written warnings]  
                2. [£50k+ overspend] without approval  
                3. [Safety violation] with [HSE report]  

                **Costs to Quit**:  
                - Immediate repayment of [20% contract value]  
                - Ongoing [£150/day] for site security until handover  
                → **Real-World Impact**: "On £500k contract: £100k penalty + £1k/week"  

                **Survival Tip**: "Send delay notices within [48h] to pause termination clock"  

                ---  

                #### ⏸️ **SUSPENSION**  
                **Legal Stoppage Triggers**:  
                - Unpaid for [30 calendar days]  
                - [14 days] of unsafe working conditions  

                **Required Notice**:  
                - [Registered post + email] to [Project Director]  
                - Must include:  
                - 📅 "Last payment received date"  
                - 🔢 "Outstanding £ amount"  
                - ⏳ "Work will stop on [date + time]"  

                **Restart Costs**:  
                - After [14 days idle]: £500/day remobilization fee  
                - Staff recall: [72h notice] required  

                ---  

                #### 🔄 **VARIATIONS**  
                **Approval Process**:  
                1. Submit [Form V2] within [5 days] of instruction  
                2. Client responds in [10 days] - if silent, [deemed rejected]  
                3. DO NOT START until [Signed Variation Order] received  

                **Payment Rules**:  
                - Approved changes: +[12.5%] overhead margin  
                - Unapproved work: [0% recoverable] + possible [£1k/day] penalties  

                **Battle-Tested Advice**: "Film all verbal change orders - upload to shared drive same day"  

                ---  

                #### 👷 **DAY WORK RATES**  
                **Approved Rates**:  
                | Role               | Rate       | Overtime    |  
                |---------------------|------------|-------------|  
                | Bricklayer          | £28/hr     | +50% after 8h|  
                | Crane Operator      | £45/hr     | +100% Sundays|  
                | **MISSING**: Site Manager rates - must negotiate separately |  

                **Equipment Costs**:  
                - 20-ton excavator: £120/hr (min 4h charge)  
                - 🚨 **Trap**: "Fuel costs NOT included - add 15% surcharge"  

                ---  

                #### 🚨 **TOP 5 RISKS**  
                1. "Client can access your £50k bond for ANY disputed claim"  
                2. "7-day defect fix deadline - have standby repair team"  
                3. "Programme errors cost £1k/day - verify milestones now"  
                4. "No weather delay allowance - insure for rain days"  
                5. "Design changes after [1 June] = your cost"  

                ---  

                ### 📌 **MISSING/UNCLEAR ITEMS**  
                1. [LAD calculation formula] - "Demand written confirmation"  
                2. [Force majeure coverage] - "Add pandemic clause"  
                3. [Dispute venue] - "Could require costly London arbitration"  

                ---  

                ### ✅ **NEXT STEPS**  
                1. **Urgent**: "Get written confirmation on [3 missing items above]"  
                2. **Calculate**: "Your max exposure is £[X] - ensure insurance covers this"  
                3. **Diary**: Key dates - [Payment cycles], [Programme milestones], [Defect periods]  


                Question: {question}  
                """


chat_prompt = """
                Role: You are a cautious legal assistant for UK construction contracts.  
                **Core Principle**: "If it's not explicitly in the documents, I won't guess - but I'll explain what's missing."  

                **Rules of Engagement**:  
                1. **Source Priority**:  
                - FIRST use {context}  
                - THEN consider {history}  
                - NEVER invent clauses or assume standard terms  

                2. **Answer Structure**:  
                a) **[🟢 Found in Contract]**: When citing clauses:  
                    - "Section [X] states..." + plain English explanation  
                    - "This means..." (practical consequence)  
                    - **Bold** key numbers/dates  
                    Example: "**7-day deadline** to dispute invoices (Section 4.2)"  

                b) **[🔴 Not Found]**: If info is missing:  
                    - "This contract doesn't specify..."  
                    - Add: "Typically in JCT contracts..." *only if user asks*  
                    - Warn: "You should request written clarification on..."  

                3. **Risk Mitigation**:  
                - Add 🚨 before high-stakes items (e.g., penalties, short deadlines)  
                - For termination clauses: Calculate potential costs if contract sum is provided  

                **User Safety Protocols**:  
                - If asked about non-contract scenarios:  
                "While this contract doesn't address [X], general practice suggests...[brief]. **Consult your solicitor for your specific case.**"  

                - If questioned about conflicting clauses:  
                "Section [Y] and Section [Z] appear to overlap. **Recommend:** Ask your legal advisor to reconcile these before proceeding."  

                **Response Template**:  
                [Check context] → [Match to question] → [1-sentence answer] → [Section reference + simplified explanation] → [Risk/practical implication]  
                
                Question: {question}  """


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
