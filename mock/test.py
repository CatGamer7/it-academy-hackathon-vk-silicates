import random
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, SparseVectorParams, Distance, SparseVector

# ========== НАСТРОЙКИ ==========
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "evaluation"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 1024
NUM_POINTS = 51

# ========== ГЕНЕРАЦИЯ СЛУЧАЙНЫХ ВЕКТОРОВ ==========
def random_dense_vector(dim: int = DENSE_VECTOR_SIZE) -> list[float]:
    """Генерирует случайный плотный вектор (нормализованный для косинусной близости)."""
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

def random_sparse_vector(max_indices: int = 100, dim: int = 30000) -> dict:
    """Генерирует случайный разреженный вектор."""
    num_nonzero = random.randint(5, 50)
    indices = random.sample(range(dim), num_nonzero)
    values = [random.random() for _ in range(num_nonzero)]
    total = sum(values)
    if total > 0:
        values = [v / total for v in values]
    return {"indices": indices, "values": values}

# ========== ГЕНЕРАЦИЯ МЕТАДАННЫХ ==========
def random_metadata(idx: int) -> dict:
    """Генерирует случайные метаданные, соответствующие структуре ChunkMetadata."""
    chat_names = ["general", "backend", "frontend", "devops", "marketing"]
    chat_types = ["group", "channel", "private"]
    participants = ["alice@example.com", "bob@example.com", "charlie@example.com", "diana@example.com"]
    mentions = ["@alice", "@bob", "@charlie"]
    
    return {
        "chat_name": random.choice(chat_names),
        "chat_type": random.choice(chat_types),
        "chat_id": f"chat_{random.randint(100,999)}",
        "chat_sn": f"sn_{random.randint(1000,9999)}",
        "thread_sn": f"thread_{random.randint(1,50)}" if random.random() > 0.7 else None,
        "message_ids": [f"msg_{idx}_{j}" for j in range(random.randint(1, 5))],
        "start": f"2025-01-{random.randint(1,28)}T{random.randint(0,23):02d}:00:00Z",
        "end": f"2025-02-{random.randint(1,28)}T{random.randint(0,23):02d}:00:00Z",
        "participants": random.sample(participants, k=random.randint(1, len(participants))),
        "mentions": random.sample(mentions, k=random.randint(0,2)),
        "contains_forward": random.choice([True, False]),
        "contains_quote": random.choice([True, False]),
        "page_content": f"Это тестовый чанк #{idx}. Случайные слова: {' '.join(random.choices(['код', 'баг', 'фича', 'документ', 'сервер'], k=5))}"
    }

# ========== ОСНОВНАЯ ЛОГИКА ==========
def main():
    # Создаём клиент с отключением проверки совместимости версий
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    
    # Создаём коллекцию, если её нет
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams()
            }
        )
        print(f"Коллекция '{COLLECTION_NAME}' создана.")
    else:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует. Точки будут добавлены.")
    
    # Генерируем и вставляем точки
    points = []
    for i in range(NUM_POINTS):
        dense_vec = random_dense_vector()
        sparse = random_sparse_vector()
        sparse_vec = SparseVector(indices=sparse["indices"], values=sparse["values"])
        metadata = random_metadata(i)
        text = metadata.pop("page_content")   # извлекаем текст для page_content
        
        point = PointStruct(
            id=i,
            vector={
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: sparse_vec
            },
            payload={
                "page_content": text,
                "metadata": metadata
            }
        )
        points.append(point)
        print(f"Сгенерирована точка {i+1}/{NUM_POINTS}")
    
    # Вставка порциями по 10
    batch_size = 10
    for start in range(0, len(points), batch_size):
        batch = points[start:start+batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"Вставлена партия {start//batch_size + 1} (точек {start+1}-{min(start+batch_size, len(points))})")
    
    print(f"Готово! Добавлено {NUM_POINTS} точек в коллекцию '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()