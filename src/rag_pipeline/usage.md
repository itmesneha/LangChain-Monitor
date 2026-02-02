
This directory contains the code for the RAG (Retrieval-Augmented Generation) pipeline. The pipeline is designed to generate responses to user queries using a combination of retrieval and generation techniques.

### Directory Structure
- ```form_data_for_collection_1.py```: This script is used to format data for collection 1 in the Milvus database. It takes the summary and metadata of GitHub issue batches and combines them into a structured format for insertion into the database.
- ```form_data_for_collection_2.py```: This script is used to format data for collection 2 in the Milvus database. It takes the summary and metadata of GitHub issue batches and combines them into a structured format for insertion into the database.
generate_response.py: This script is used to generate responses to user queries using the RAG pipeline. It retrieves relevant insights from the Milvus database, combines them with other information, and generates a response using a language model.
- ```set_up_milvus_db.py```: This script is used to set up the Milvus database for the RAG pipeline. It creates the necessary collections and indexes in the database.
- ```test_query.py```: This script is used to test the RAG pipeline's query functionality. It retrieves relevant insights from the Milvus database based on a user query and prints the results.

### How to Use
To use the RAG pipeline, follow these steps:

1. Format the data for collection 1 by running form_data_for_collection_1.py.
2. Format the data for collection 2 by running form_data_for_collection_2.py.
3. Set up the Milvus database by running set_up_milvus_db.py.
4. Test the query functionality by running test_query.py.
5. Generate responses to user queries by running generate_response.py.