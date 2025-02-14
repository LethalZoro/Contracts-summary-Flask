from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from openai import OpenAI
import uuid
import os

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit

# Initialize OpenAI API key from environment variables

# In-memory storage for contract texts (use database in production)
contracts = {}

client = OpenAI(api_key="")


assistant = client.beta.assistants.create(
    name="legal contract analyst",
    instructions="""You are a legal contract analyst. 
                Summarize this construction contract and extract these key sections:
                - Payment Terms
                - Termination Clauses
                - Variations
                - Notifications
                - Retention
                Provide a concise summary followed by clear section details.""",
    tools=[{"type": "file_search"}],
    model="gpt-4o",
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


def generate_summary(assistant, vector_store):
    """Generate contract summary using OpenAI"""
    try:

        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": """Summarize the contract by addressing the user as user.
                        Include key sections such as Payment Terms, Termination Clauses, Variations, Notifications, and Retention.
                        Provide a concise summary followed by clear section details.""",
                    # Attach the new file to the message.
                    # "attachments": [
                    #   { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
                    # ],
                }
            ]
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            tools=[{
                "type": "file_search",
                "file_search": {"max_num_results": 3}
            }]
            # ,max_completion_tokens=1500
        )

        messages = list(client.beta.threads.messages.list(
            thread_id=thread.id, run_id=run.id))
        # print(messages)
        message_content = messages[0].content[0].text
        # annotations = message_content.annotations
        # citations = []
        # for index, annotation in enumerate(annotations):
        #     message_content.value = message_content.value.replace(
        #         annotation.text, f"[{index}]")
        #     if file_citation := getattr(annotation, "file_citation", None):
        #         cited_file = client.files.retrieve(file_citation.file_id)
        #         citations.append(f"[{index}] {cited_file.filename}")

        if run.status == 'completed':
            return message_content.value
            print(message_content.value)
            print("\n".join(citations))
            # print(messages)
        else:
            print(run.status)
            print(run.last_error)
            raise RuntimeError("AI processing failed in summary")

    except Exception as e:
        raise RuntimeError(
            f"AI processing failed before summary error:{e}") from e


def generate_answer(contract_text, question):
    """Generate AI response for chat questions"""
    try:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                    # Attach the new file to the message.
                    # "attachments": [
                    #   { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
                    # ],
                }
            ]
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            tools=[{
                "type": "file_search",
                "file_search": {"max_num_results": 3}
            }],
            instructions=f"""You are a legal contract assistant. 
                Answer questions based strictly on the contract file:
                If information isn't in the contract, say so. 
                If the user asks any question not in the contract, 
                say so explicitly and then you can answer to the best of your knowledge."""
        )
        # ,max_completion_tokens=1500

        messages = list(client.beta.threads.messages.list(
            thread_id=thread.id, run_id=run.id))
        # print(messages)
        message_content = messages[0].content[0].text

        if run.status == 'completed':
            return message_content.value
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
        # Extract text from PDF
        # text = extract_text_from_pdf(file)
        temp_dir = "D:/Coding/Job/Contracts Summary AI alex/tmp/"
        # This will create the directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        temp_path = os.path.join(temp_dir, file.filename)
        print(temp_path)
        file.save(temp_path)
        print("File saved")

        vector_store = client.beta.vector_stores.create(name="legal contract")

        with open(temp_path, "rb") as f:
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=[f]
            )
        # You can print the status and the file counts of the batch to see the result of this operation.
        print(file_batch.status)
        print(file_batch.file_counts)

        client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {
                "vector_store_ids": [vector_store.id]}},
        )

        # print(text)
        # Generate unique ID for the contract
        contract_id = str(uuid.uuid4())

        # Generate AI summary
        summary = generate_summary(assistant, vector_store)

        contracts[contract_id] = {"summary": summary}

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

    app.run(host='0.0.0.0', port=5000, debug=True)
