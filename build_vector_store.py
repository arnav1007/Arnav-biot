from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import uuid, re

client = OpenAI()

qdrant = QdrantClient(":memory:")  # use ":memory:" for local in-memory db or provide host/url

collection_name = "arnav-chats"

# Create collection
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Load and clean messages
def load_arnav_messages(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    return re.findall(r'\d{1,2}/\d{1,2}/\d{2,4},.*?Arnav:\s*(.+)', raw)

messages = load_arnav_messages("chat.txt")

# Embed messages using OpenAI
batch = []
for i, text in enumerate(messages):
    if not text.strip(): continue
    embedding = client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
    batch.append(PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": text}))

qdrant.upload_points(collection_name=collection_name, points=batch)
print(f"Uploaded {len(batch)} chat messages to Qdrant.")
