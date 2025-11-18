from pymongo import MongoClient

# connect to your Atlas cluster
client = MongoClient("mongodb+srv://bharathbandi13925:Bandi12345@air-quality-cluster.ep5mhed.mongodb.net/?appName=air-quality-cluster")

# select database and collection
db = client["air_quality_db"]
collection = db["historical_data"]

# delete all documents where city == "delhi"
result = collection.delete_many({"city": "Delhi"})

print(f"✅ Deleted {result.deleted_count} documents.")
