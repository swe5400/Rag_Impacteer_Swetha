# Simple RAG-Based Document Retriever

This project is a simple Retrieval-Augmented Generation (RAG) style document retriever that can return relevant information from a small collection of predefined documents using vector similarity.

## SentenceTransformer Model Used

*   **Model:** `all-MiniLM-L6-v2`
*   **Why it was chosen:** This model is a small, fast, and balanced model that provides good performance for semantic search. It is a good choice for this project because it is easy to use and does not require a lot of computational resources.

## Chunking Strategy

*   **How the document was chunked:** The documents were chunked into paragraphs based on double newlines.
*   **Why this method was chosen:** This method was chosen because it is simple and effective. It is a good way to split the documents into smaller, more manageable chunks that can be easily processed by the model.

## Similarity Metric

*   **What metric was used:** Cosine similarity was used to measure the similarity between the query and the document chunks.
*   **Why it’s appropriate for semantic retrieval:** Cosine similarity is a good metric for semantic retrieval because it measures the cosine of the angle between two vectors. This means that it is a measure of the orientation of the vectors, not their magnitude. This is important for semantic retrieval because it allows us to find documents that are semantically similar to the query, even if they do not have the same keywords.

## How to Run

1.  **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
3.  **Open the application in your browser:**
    The application will be running at `http://localhost:8501`.
4. **Add documents:**
    Add your `.txt` documents to the `documents` folder.
