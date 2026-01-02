from qdrant_client import QdrantClient, models
import pandas as pd 
import ast
import numpy as np
import os
# 從 Python 建立 Qdrant 的 collection
client = QdrantClient("localhost", port=6333)
collection_name = "tea_disease_collection"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
    )
    print(f"Collection '{collection_name}' is ready.")
else:
    print(f"Collection '{collection_name}' already exists.")

# Collection 本身有三個主要的元素 id, vector, payload,
# 對病害影像資料 
# 將圖片檔名 設定為 id
# 將 resize 成 224 x 224 的圖片以  

# 產生資料用

df = pd.read_csv(r"C:\Users\user\Desktop\謙恩的文件\Tea_Disease_Project\Qdrant_db\ThinResNet18_run4_payloads.csv")
vectors = pd.read_csv(r"C:\Users\user\Desktop\謙恩的文件\Tea_Disease_Project\Qdrant_db\ThinResNet18_run4_vectors.csv")
vectors_array = vectors["0"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ')).to_list()
vectors_array = np.array(vectors_array)  # shape = (num_points, vector_dim)
base_url = "http://localhost:8080/train"
df["local_url"] = df["file_name"].apply(lambda x: f"{base_url}/{x.replace('\\', '/')}")
print(df[["file_name", "local_url"]].head())


print(vectors_array.shape)
print("Vectors shape:", vectors.shape)

points = []

for idx, row in df.iterrows():
    payload = row.to_dict()
    points.append(models.PointStruct(
        id=idx,
        vector=vectors_array[idx].tolist(),
        payload=payload,
    ))

client.upsert(
    collection_name=collection_name,
    points=points,wait=True
)
print(f"{len(points)} points have been added to the collection '{collection_name}'.")

# 刪除用
client.points_api.delete(
    collection_name=collection_name,
    points_selector=models.PointsSelector(
        filter=models.Filter(must=[])
    )
)
print("All points deleted.")

# 查詢用
search_vector = []
results = client.query_points(
    collection_name = collection_name,
    query = search_vector,
    limit = 20,
    with_payload = True,
    with_vectors = False
)
print("\nQuery Results:")
for result in results.points:
    print(f"ID:{result.id}, Score:{result.score:.4f}, Payload:{result.payload}\n")