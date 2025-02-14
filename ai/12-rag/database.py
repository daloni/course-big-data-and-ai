import pinecone

class Database:
    def __init__(self, api_key, region, index_name, recreate_index):
        self.pc = pinecone.Pinecone(api_key=api_key)
        self.index_name = index_name

        if recreate_index and index_name in [i.name for i in self.pc.list_indexes()]:
            self.delete_index()

        if index_name not in [i.name for i in self.pc.list_indexes()]:
            self.create_index(region=region)

        self.index = self.pc.Index(index_name)

    def delete_index(self):
        self.pc.delete_index(self.index_name)

    def create_index(self, cloud="aws", region="us-east-1"):
        self.pc.create_index(
            name=self.index_name,
            dimension=768,
            metric="cosine",
            spec=pinecone.ServerlessSpec(cloud=cloud, region=region)
        )

    def insert(self, documents):
        self.index.upsert(documents)

    def query(self, vector, top_k=2, include_metadata=True):
        return self.index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)
