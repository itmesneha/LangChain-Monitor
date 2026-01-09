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


if __name__ == "__main__":
    client = create_database()
    schema = create_schema_for_collection_1()
    create_collection_1(client, schema)