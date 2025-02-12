from flask import Flask, request, jsonify
import pdfplumber
from openai import OpenAI
import uuid
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit

# Initialize OpenAI API key from environment variables

# In-memory storage for contract texts (use database in production)
contracts = {}

client = OpenAI(api_key="")


assistant = client.beta.assistants.create(
    name="legal contract analyst",
    instructions="""You are a legal contract analyst. 
                Summarize this UK construction contract and extract these key sections:
                - Payment Terms
                - Termination Clauses
                - Variations
                - Notifications
                - Retention
                Provide a concise summary followed by clear section details.""",
    tools=[{"type": "file_search"}],
    model="gpt-4o",
)

assistant_Questions = client.beta.assistants.create(
    name="legal contract analyst",
    instructions="""You are a legal contract assistant. Answer questions based strictly on this information give in the contract. 
    If the user asks any question not in the contract, 
    say so explicitly and then you can answer to the best of your knowledge.""",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


def extract_text_from_pdf(file):
    """Extract text from PDF using pdfplumber"""
    text = []
    try:
        with pdfplumber.open(file.stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        raise RuntimeError("Failed to extract text from PDF") from e


def generate_summary(file, vector_store, assistant):
    """Generate contract summary using OpenAI"""
    try:

        message_file = client.files.create(
            file=open(file, "rb"), purpose="assistants"
        )

        # Create a thread and attach the file to the message
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": """Summarize the contract by addressing the user as user.
            Include key sections such as Payment Terms, Termination Clauses, Variations, Notifications, and Retention.
            Provide a concise summary followed by clear section details.""",
                    # Attach the new file to the message.
                    "attachments": [
                        {"file_id": message_file.id, "tools": [
                            {"type": "file_search"}]}
                    ],
                }
            ]
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant.id
        )

        messages = list(client.beta.threads.messages.list(
            thread_id=thread.id, run_id=run.id))

        message_content = messages[0].content[0].text
        # annotations = message_content.annotations
        # citations = []
        # for index, annotation in enumerate(annotations):
        #     message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        #     if file_citation := getattr(annotation, "file_citation", None):
        #         cited_file = client.files.retrieve(file_citation.file_id)
        #         citations.append(f"[{index}] {cited_file.filename}")

        # thread = client.beta.threads.create()
        # message = client.beta.threads.messages.create(
        #     thread_id=thread.id,
        #     role="user",
        #     content=text
        # )
        # run = client.beta.threads.runs.create_and_poll(
        #     thread_id=thread.id,
        #     assistant_id=assistant.id,
        #     instructions="""Summarize the contract by addressing the user as user.
        #     Include key sections such as Payment Terms, Termination Clauses, Variations, Notifications, and Retention.
        #     Provide a concise summary followed by clear section details."""
        # )
        if run.status == 'completed':
            messages = list(client.beta.threads.messages.list(
                thread_id=thread.id, run_id=run.id))
            print(message_content.value)
            # print(messages)
        else:
            # print(run.status)
            # print(run.last_error)
            raise RuntimeError("AI processing failed in summary")

        # return messages.data[0].content[0].text.value
        return message_content.value

    except Exception as e:
        raise RuntimeError(
            f"AI processing failed in summary. Error: {e}") from e


def generate_answer(contract_text, question):
    """Generate AI response for chat questions"""
    try:
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_Questions.id,
            instructions=f"""You are a legal contract assistant. 
                Answer questions based strictly on this contract:
                {contract_text}
                If information isn't in the contract, say so. 
                If the user asks any question not in the contract, 
                say so explicitly and then you can answer to the best of your knowledge."""
        )
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            return messages.data[0].content[0].text.value
        else:
            # print(run.status)
            # print(run.last_error)
            raise RuntimeError("AI processing failed in chat")
    except Exception as e:
        raise RuntimeError("Failed to generate answer") from e


@app.route('/upload', methods=['POST'])
def upload_contract():
    """Endpoint for PDF upload and processing"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:

        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join(
            "D:/Coding/Job/Contracts Summary AI/temp", file.filename)
        file.save(temp_file_path)

        # Extract text from PDF
        text = extract_text_from_pdf(file)
        # Create a vector store caled "Financial Statements"

        vector_store = client.beta.vector_stores.create(
            name="legal contract analyst")

        # Ready the files for upload to OpenAI
        file_paths = [temp_file_path]
        file_streams = [open(path, "rb") for path in file_paths]

        # Use the upload and poll SDK helper to upload the files, add them to the vector store,
        # and poll the status of the file batch for completion.
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
        )

        # You can print the status and the file counts of the batch to see the result of this operation.
        print(file_batch.status)
        print(file_batch.file_counts)

        # Generate unique ID for the contract
        contract_id = str(uuid.uuid4())
        contracts[contract_id] = text

        assistant = client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {
                "vector_store_ids": [vector_store.id]}},
        )
        # Generate AI summary
        summary = generate_summary(temp_file_path, vector_store, assistant)

        return jsonify({
            "contract_id": contract_id,
            "summary": summary
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat_handler():
    """Endpoint for contract-related questions"""
    data = request.get_json()
    if not data or 'contract_id' not in data or 'question' not in data:
        return jsonify({"error": "Missing contract_id or question"}), 400

    contract_id = data['contract_id']
    question = data['question']

    if contract_id not in contracts:
        return jsonify({"error": "Invalid contract ID"}), 404

    try:
        answer = generate_answer(contracts[contract_id], question)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/contracts', methods=['GET'])
def get_contracts():
    """Endpoint to list all uploaded contracts"""
    if not contracts:
        return jsonify({"error": "No contracts uploaded"}), 404
    return jsonify(contracts), 200


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
