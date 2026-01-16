import json
from pymilvus import MilvusClient, DataType


def create_database():
    client = MilvusClient(
        "./Repository_monitoring.db"
    )

    # client.create_database(
    #     db_name="Repository_monitoring",
    #     properties={
    #         "database.max.collections": 5
    #     }
    # )
    return client

'''
batch_id	VARCHAR (PK)	e.g. "langchain_batch_0001"
repo	VARCHAR	"langchain"
batch_number	INT64	1, 2, 3, …
num_issues	INT64	10
Content fields
Field	Type	Notes
issue_digest	VARCHAR	Concatenated issue summaries
business_insights_json	VARCHAR	JSON-stringified list
technical_insights_json	VARCHAR	JSON-stringified list
'''
def create_schema_for_collection_1():
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="batch_id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field(field_name="repo", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="batch_number", datatype=DataType.INT64)
    schema.add_field(field_name="num_issues", datatype=DataType.INT64)
    schema.add_field(field_name="issue_digest", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="business_insights_json", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="technical_insights_json", datatype=DataType.VARCHAR, max_length=65535)
    return schema


def create_collection_1(client, schema):
    client.create_collection(
        collection_name="issue_batches",
        schema=schema,
    )
    res = client.get_load_state(
        collection_name="issue_batches"
    )
    print(res)


def insert_to_collection_1(client, collection_name, data):
    client.insert(
        collection_name=collection_name,
        data=data
    )


def create_schema_for_collection_1():
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="batch_id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field(field_name="repo", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="batch_number", datatype=DataType.INT64)
    schema.add_field(field_name="num_issues", datatype=DataType.INT64)
    schema.add_field(field_name="issue_digest", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="business_insights_json", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="technical_insights_json", datatype=DataType.VARCHAR, max_length=65535)
    return schema



'''
insight_id	VARCHAR (PK)	"langchain_b1_tech_03"
repo	VARCHAR	"langchain"
batch_number	INT64	Links to parent batch
insight_type	VARCHAR	"business" or "technical"
insight_index	INT64	Order within batch (1–N)
insight_text	VARCHAR	The actual insight bullet
Vector field (required)
embedding	FLOAT_VECTOR	Embedding of insight_text
'''
def create_schema_for_collection_2():
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="insight_id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field(field_name="repo", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="batch_number", datatype=DataType.INT64)
    schema.add_field(field_name="insight_type", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="insight_index", datatype=DataType.INT64)
    schema.add_field(field_name="insight_text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="insight_vector", datatype=DataType.VARCHAR, max_length=65535)
    return schema


def insert_to_collection_2(client, collection_name, data):
    client.insert(
        collection_name=collection_name,
        data=data
    )


if __name__ == "__main__":
    client = create_database()
    schema = create_schema_for_collection_1()
    create_collection_1(client, schema)
    collection_1_data_path = "../../data/processed/collection_1.json"
    with open(collection_1_data_path, "r") as f:
        data = json.load(f)
    insert_to_collection_1(client, "issue_batches", data)

    collection_2_data_path = "../../data/processed/collection_2.json"
    with open(collection_2_data_path, "r") as f:
        data = json.load(f)
    insert_to_collection_2(client, "issue_insights", data)


    