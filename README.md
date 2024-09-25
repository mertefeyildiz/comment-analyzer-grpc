# Comment Analyzer GRPC Service

This project provides a GRPC-based service for analyzing social media comments using BERT for classification and GPT-Neo for generating conclusions. The service categorizes comments into claims, counterclaims, evidence, and rebuttals, and generates a detailed conclusion based on these categories.

## Features

- **BERT Classification**: Classifies comments as Claims, Counterclaims, Evidence, or Rebuttals.
- **GPT-Neo Conclusion Generation**: Generates a detailed and well-reasoned conclusion based on the classified comments.
- **GRPC Server and Client**: A GRPC server handles requests to analyze comments, and a client sends requests to the server for comment analysis.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mertefeyildiz/comment-analyzer-grpc.git
    ```
2. Navigate to the project directory:
    ```bash
    cd comment-analyzer-grpc
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dependencies

- `grpcio`
- `grpcio-tools`
- `sentence-transformers`
- `transformers`
- `torch`
- `pandas`

## Usage

### Running the GRPC Server

1. Ensure that `topics.csv` and `opinions.csv` are available in the same directory. These files should contain the topics and related opinions for analysis.
2. Run the GRPC server using the following command:
    ```bash
    python grpc_server3.py
    ```
3. The server will start on port 50051 and will be ready to handle requests.

### Running the GRPC Client

1. Make sure the server is running.
2. Use the client script to send a request to the server:
    ```bash
    python grpc_client.py
    ```
3. The client script will send a request to analyze a topic using a given `topic_id` and print the generated conclusion.

### Example Client Code (`grpc_client.py`)

Here's an example of how the client interacts with the server:

```python
import grpc
import grpc_service_pb2
import grpc_service_pb2_grpc

# Connect to the server
channel = grpc.insecure_channel('localhost:50051')
stub = grpc_service_pb2_grpc.CommentAnalyzerStub(channel)

# Send a test topic_id (Example: '007ACE74B050' should exist in your data files)
topic_id = "007ACE74B050"  # Example topic_id
response = stub.AnalyzeComment(grpc_service_pb2.AnalyzeRequest(topic_id=topic_id))
print(f"Conclusion: {response.conclusion}")
