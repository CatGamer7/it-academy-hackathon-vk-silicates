import logging
import os
from functools import lru_cache
from typing import Any
import asyncio
import hashlib

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

CHUNK_SIZE = 512
OVERLAP_SIZE = 256
SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"

# Важная переманная, которая позволяет вычислять sparse вектор в несколько ядер. Не рекомендуется изменять.
UVICORN_WORKERS=8

def render_message(message: Message) -> str:
    text = ""

    if message.text:
        text += message.text

    if message.parts:
        parts_text: list[str] = []
        for part in message.parts:
            # parts различаются по своему типу, см. README.md
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text:
                parts_text.append(part_text)
        if parts_text:
            text += " " + "\n".join(parts_text)

    return text


def build_sparse_metadata(
    chat: Chat,
    messages_in_chunk: list[Message],
    max_meta_len: int,
) -> str:
    """Формирует строку метаданных для sparse-вектора с ограничением длины."""
    meta_parts = []
    current_len = 0

    # Название и тип чата
    chat_info = f"[chat: {chat.name}"
    meta_parts.append(chat_info)
    current_len += len(chat_info) + 1  # +1 для пробела

    # Имена участников (первые 5, без email)
    if chat.members:
        names = []
        for m in chat.members[:5]:
            name = str(m.get("name") or m.get("id", ""))
            if name:
                # Проверяем, влезет ли с учётом разделителя
                additional = len(name) + 2  # ", " или начало
                if current_len + additional + len("[people: ]") <= max_meta_len:
                    names.append(name)
                    current_len += additional
                else:
                    break
        if names:
            people_str = f"[people: {', '.join(names)}]"
            # Если не влезает целиком, то не добавляем
            if current_len + len(people_str) <= max_meta_len:
                meta_parts.append(people_str)
                current_len += len(people_str) + 1

    # Ссылки (упрощённо: ищем http в file_snippets)
    links = set()
    for msg in messages_in_chunk:
        if msg.file_snippets:
            import re
            found = re.findall(r'https?://(\S+)', msg.file_snippets)
            links.update(found)
    if links:
        link_list = []
        base_len = len("[links: ]") + current_len
        for link in links:
            additional = len(link) + 2
            if base_len + additional <= max_meta_len:
                link_list.append(link)
                base_len += additional
            else:
                break
        if link_list:
            links_str = f"[links: {' '.join(link_list)}]"
            if current_len + len(links_str) <= max_meta_len:
                meta_parts.append(links_str)

    # Email'ы: из членов чата, отправителей, упоминаний
    emails = set()
    if chat.members:
        for m in chat.members:
            email = m.get("email")
            if email:
                emails.add(str(email))
    for msg in messages_in_chunk:
        if msg.sender_id:
            emails.add(msg.sender_id)
        if msg.mentions:
            emails.update(msg.mentions)

    # Добавляем email'ы, пока не упрёмся в лимит
    if emails:
        email_list = []
        base_len = len("[emails: ]") + current_len
        for email in emails:
            # +2 на пробел и запятую/конец
            additional = len(email) + 2
            if base_len + additional <= max_meta_len:
                email_list.append(email)
                base_len += additional
            else:
                break
        if email_list:
            emails_str = f"[emails: {' '.join(email_list)}]"
            # Проверяем ещё раз полную длину (с учётом уже добавленных частей)
            if current_len + len(emails_str) <= max_meta_len:
                meta_parts.append(emails_str)
                current_len += len(emails_str) + 1

    return " ".join(meta_parts)


def build_chunks(
    chat: Chat,
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

        # Находим сообщения, которые полностью или частично входят в чанк
        chunk_msg_ids = {msg_id for _, _, msg_id in chunk_body_ranges}
        chunk_messages = [msg for msg in new_messages if msg.id in chunk_msg_ids]
        sparse_chunk_text_meta = build_sparse_metadata(chat, chunk_messages, CHUNK_SIZE + OVERLAP_SIZE)
        if chunk_body:
            left_len = CHUNK_SIZE + OVERLAP_SIZE - len(sparse_chunk_text_meta)
            sparse_chunk_text_meta += chunk_body[:left_len]

        result.append(
            IndexAPIItem(
                page_content=chunk_text,
                dense_content=chunk_text,
                sparse_content=sparse_chunk_text_meta,
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
            payload.data.chat,
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
