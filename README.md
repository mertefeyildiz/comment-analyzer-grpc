# Comment Analyzer GRPC Service

This project provides a GRPC-based service for analyzing social media comments using BERT for classification and GPT-Neo for generating conclusions. The service categorizes comments into claims, counterclaims, evidence, and rebuttals, and generates a detailed conclusion based on these categories.

## Features

- **BERT Classification**: Classifies comments as Claims, Counterclaims, Evidence, or Rebuttals.
- **GPT-Neo Conclusion Generation**: Generates a detailed and well-reasoned conclusion based on the classified comments.
- **GRPC Server**: Handles requests to analyze comments related to specific topics.

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

1. Ensure that `topics.csv` and `opinions.csv` are available in the same directory. These files should contain the topics and related opinions for analysis.
2. Run the GRPC server:
    ```bash
    python main.py
    ```
3. The server will start on port 50051 and will be ready to handle requests.

## API

### AnalyzeComment

- **Request**: A `topic_id` to identify the topic for analysis.
- **Response**: A generated conclusion based on the arguments classified from related comments.

## Models Used

- **BERT**: Used for classifying comments into categories such as Claims, Counterclaims, Evidence, and Rebuttals.
- **GPT-Neo**: Generates a coherent conclusion based on classified arguments.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
