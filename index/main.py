import logging
import os
from functools import lru_cache
from typing import Any
import asyncio
import hashlib
import re
import string

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ваш сервис должен считывать эти переменные из окружения (env), так как проверяющая система управляет ими
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8004"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")


# Модель данных, которую мы предоставляем и рассчитываем получать от вас
class Chat(BaseModel):
    id: str
    name: str
    sn: str
    type: str  # group, channel, private
    is_public: bool | None = None
    members_count: int | None = None
    members: list[dict[str, Any]] | None = None


class Message(BaseModel):
    id: str
    thread_sn: str | None = None
    time: int
    text: str
    sender_id: str
    file_snippets: str
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None
    member_event: dict[str, Any] | None = None
    is_system: bool
    is_hidden: bool
    is_forward: bool
    is_quote: bool


class ChatData(BaseModel):
    chat: Chat
    overlap_messages: list[Message]
    new_messages: list[Message]


class IndexAPIRequest(BaseModel):
    data: ChatData


# dense_content будет передан в dense embedding модель для построения семантического вектора.
# sparse_content будет передан в sparse модель для построения разреженного индекса "по словам".
# Можно оставить dense_content и sparse_content равными page_content,
# а можно формировать для них разные версии текста.
class IndexAPIItem(BaseModel):
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]


class IndexAPIResponse(BaseModel):
    results: list[IndexAPIItem]


class SparseEmbeddingRequest(BaseModel):
    texts: list[str]


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]


app = FastAPI(title="Index Service", version="0.1.0")

# Ваша внутренняя логика построения чанков. Можете делать всё, что посчитаете нужным.
# Текущий код – минимальный пример

CHUNK_SIZE = 1024
OVERLAP_SIZE = 256
SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"

# Важная переманная, которая позволяет вычислять sparse вектор в несколько ядер. Не рекомендуется изменять.
UVICORN_WORKERS=8


RUSSIAN_STOP_WORDS = {
    "и", "в", "во", "не", "что", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его",
    "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот",
    "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам",
    "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо",
    "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз",
    "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой",
    "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были",
    "куда", "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много",
    "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше",
    "чуть", "том", "нельзя", "такой", "ими", "него", "надо", "вон", "кроме", "сегодня", "будь"
}

ENGLISH_STOP_WORDS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "a", "an", "the", "and", "or", "but", "so", "for", "nor",
    "yet", "of", "to", "in", "for", "on", "by", "with", "without", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from", "up", "down", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "that", "then", "these", "those", "too", "very", "just", "but", "do",
    "does", "did", "doing", "have", "has", "had", "having", "be", "am", "are", "is", "was", "were",
    "being", "been", "get", "gets", "got", "getting", "make", "makes", "made", "making", "can",
    "cannot", "could", "will", "would", "should", "may", "might", "must", "shall"
}

# Объединяем в один набор для быстрой проверки
STOP_WORDS = RUSSIAN_STOP_WORDS.union(ENGLISH_STOP_WORDS)
STOP_WORDS.update(['привет', 'здравствуйте', 'hello', 'hi'])

def preprocess_for_sparse_vector(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 2. Токенизация
    # 4. Удаление пунктуации и цифр
    tokens = re.findall(r"[a-zа-яё]+", text, flags=re.IGNORECASE)

    # # 3. Приведение к нижнему регистру (опционально, выключил)
    # if lower:
    #     tokens = [token.lower() for token in tokens]

    # 5. Удаление стоп-слов
    tokens = [token for token in tokens if token.lower() not in STOP_WORDS]

    # 6. Сборка итоговой строки с одиночными пробелами
    cleaned_text = ' '.join(tokens)

    return cleaned_text


def render_message(message: Message) -> str:
    text = ""

    if message.text:
        text += re.sub(r'\s+', ' ', message.text).strip()

    if message.parts:
        parts_text: list[str] = []
        for part in message.parts:
            # parts различаются по своему типу, см. README.md
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text:
                parts_text.append(re.sub(r'\s+', ' ', part_text).strip())
        if parts_text:
            text += " " + " ".join(parts_text)

    return text


def build_chunks(
    overlap_messages: list[Message],
    new_messages: list[Message],
) -> list[IndexAPIItem]:
    result: list[IndexAPIItem] = []

    def build_text_and_ranges(messages: list[Message]) -> tuple[str, list[tuple[int, int, str]]]:
        text_parts: list[str] = []
        message_ranges: list[tuple[int, int, str]] = []
        position = 0

        for index, message in enumerate(messages):
            text = render_message(message)
            if not text:
                continue

            if index > 0 and text_parts:
                text_parts.append("\n")
                position += 1

            start = position
            text_parts.append(text)
            position += len(text)
            message_ranges.append((start, position, message.id))

        return "".join(text_parts), message_ranges

    def slice_tail(
        text: str,
        tail_size: int,
    ) -> str:
        if tail_size <= 0:
            return ""

        tail_start = max(0, len(text) - tail_size)
        return text[tail_start:]

    overlap_text, overlap_message_ranges = build_text_and_ranges(overlap_messages)
    previous_chunk_text = slice_tail(overlap_text, OVERLAP_SIZE)

    new_text, new_message_ranges = build_text_and_ranges(new_messages)

    for start in range(0, len(new_text), CHUNK_SIZE):
        chunk_body = new_text[start : start + CHUNK_SIZE]
        if not chunk_body:
            continue

        chunk_body_ranges = [
            (
                max(message_start, start) - start,
                min(message_end, start + len(chunk_body)) - start,
                message_id,
            )
            for message_start, message_end, message_id in new_message_ranges
            if message_end > start and message_start < start + len(chunk_body)
        ]
        chunk_overlap = previous_chunk_text
        chunk_text = chunk_overlap
        if chunk_text and chunk_body:
            chunk_text += "\n"
        chunk_text += chunk_body

        result.append(
            IndexAPIItem(
                page_content=chunk_text,
                dense_content=chunk_text,
                sparse_content=preprocess_for_sparse_vector(chunk_text),
                message_ids=[message_id for _, _, message_id in chunk_body_ranges],
            )
        )
        previous_chunk_text = slice_tail(chunk_text, OVERLAP_SIZE)

    return result

# Ваш сервис должен имплементировать оба этих метода
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    return IndexAPIResponse(
        results=build_chunks(
            payload.data.overlap_messages,
            payload.data.new_messages,
        )
    )


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding

    # можете делать любой вектор, который будет совместим с вашим поиском в Qdrant
    # помните об ограничении времени выполнения вашей работы в тестирующей системе
    logger.info(
        "Loading sparse model %s from cache %s",
        SPARSE_MODEL_NAME,
        FASTEMBED_CACHE_PATH,
    )
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    vectors: list[dict[str, list[int] | list[float]]] = []

    for item in model.embed(texts):
        vectors.append(
            {
                "indices": item.indices.tolist(),
                "values": item.values.tolist(),
            }
        )

    return vectors


@app.post("/sparse_embedding")
async def sparse_embedding(payload: SparseEmbeddingRequest) -> dict[str, Any]:
    # Проверяющая система вызывает этот endpoint при создании коллекции
    vectors = await asyncio.to_thread(embed_sparse_texts, payload.texts)
    return {"vectors": vectors}

# красивая обработка ошибок
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=UVICORN_WORKERS,
    )


if __name__ == "__main__":
    main()
