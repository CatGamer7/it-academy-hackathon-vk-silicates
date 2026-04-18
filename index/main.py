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
META_INFO_SIZE = 256
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
            text += " " + " ".join(parts_text)

    return text


def enrich_chunk_text(
    chunk_text: str,           # оригинальный текст чанка (без метаданных)
    chat: Chat,
    messages_in_chunk: list[Message],  # сообщения, входящие в этот чанк
    message_ranges: list[tuple[int, int, str]],  # (start, end, msg_id) в chunk_text
    max_len: int,
) -> list[tuple[str, list[str]]]:   # возвращает список (обогащённый_текст, [id_сообщений])
    # Формируем строку метаданных
    meta_lines = []
    len_characters_meta_lines = 0

    # Информация из чата
    chat_info = f"[Chat: {chat.name} ({chat.type})]"
    meta_lines.append(chat_info)
    len_characters_meta_lines += len(chat_info)

    members_emails = []
    if chat.members:
        member_names = [str(m.get("name") or m.get("id", "")) for m in chat.members[:5]]
        members_emails = [str(m.get("email", "")) for m in chat.members[:5]]
        members_info = f"[people: {', '.join(member_names)}]"
        meta_lines.append(members_info)
        len_characters_meta_lines += len(members_info)

    # Информация из Message
    all_mentions = set()
    senders_emails = set()
    for msg in messages_in_chunk:
        if msg.mentions:
            all_mentions.update(msg.mentions[:5])
            senders_emails.add(msg.sender_id)

    all_mentions.update(members_emails)
    all_mentions.update(senders_emails)

    mentions_line = "[emails: "
    mentions_emails = []
    len_characters_meta_lines += len(mentions_line) + 1
    if all_mentions:
        for mention in all_mentions:
            if len(mention) + len_characters_meta_lines > META_INFO_SIZE:
                break
            else:
                mentions_emails.append(mention)
                len_characters_meta_lines += len(mention) + 1

    mention_info = f"[emails: {' '.join(mentions_emails)}]"
    meta_lines.append(mention_info)
    # if has_forward:
    #     meta_lines.append("[Есть пересылка]")
    # if has_quote:
    #     meta_lines.append("[Есть цитата]")
    
    meta_str = " ".join(meta_lines) + " "
    total_len = len(meta_str) + len(chunk_text)

    # Если влезает – возвращаем один чанк
    if total_len <= max_len:
        return [(meta_str + chunk_text, [msg_id for _, _, msg_id in message_ranges])]

    # Иначе делим chunk_text на две части по границам сообщений
    # Находим точку раздела – половину длины текста
    split_point = len(chunk_text) // 2
    # Корректируем split_point до ближайшей границы сообщения
    best_split = split_point
    for start, end, _ in message_ranges:
        if start <= split_point <= end:
            # Если точка раздела внутри сообщения, отдаём всё сообщение в первую часть
            best_split = end
            break
        elif start > split_point:
            best_split = start
            break
    else:
        # Если не нашли, оставляем split_point как есть
        best_split = split_point
    
    part1_text = chunk_text[:best_split]
    part2_text = chunk_text[best_split:]
    
    # Распределяем message_ids по частям
    part1_ids = []
    part2_ids = []
    for start, end, msg_id in message_ranges:
        if start < best_split:
            part1_ids.append(msg_id)
        if end > best_split:
            part2_ids.append(msg_id)
        # Если сообщение ровно на границе (end == best_split), включаем его в первую часть
        elif end == best_split:
            part1_ids.append(msg_id)

    # Формируем результат: две части с одними и теми же метаданными
    result = []
    if part1_text.strip():
        result.append((meta_str + part1_text, part1_ids))
    if part2_text.strip():
        result.append((meta_str + part2_text, part2_ids))
    return result


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
                text_parts.append(" ")
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

        # Находим сообщения, которые полностью или частично входят в чанк
        chunk_msg_ids = {msg_id for _, _, msg_id in chunk_body_ranges}
        chunk_messages = [msg for msg in new_messages if msg.id in chunk_msg_ids]

        # Добавляем перекрытие
        chunk_overlap = previous_chunk_text
        # full_chunk_text = chunk_overlap + ("\n" if chunk_overlap and chunk_body else "") + chunk_body        
         # 1. Берём chunk_body.
        # 2. Обогащаем его (может разбиться на несколько под-чанков).
        # 3. К каждому под-чанку добавляем перекрытие (previous_chunk_text) в начало.
        # 4. Обновляем previous_chunk_text для следующей итерации.
        
        # Вызов enrich_chunk_text для chunk_body (без перекрытия)
        enriched_parts = enrich_chunk_text(
            chunk_text=chunk_body,
            chat=chat,
            messages_in_chunk=chunk_messages,
            message_ranges=chunk_body_ranges,
            max_len=CHUNK_SIZE,
        )

        # chunk_overlap = previous_chunk_text
        # chunk_text = chunk_overlap
        # if chunk_text and chunk_body:
        #     chunk_text += "\n"
        # chunk_text += chunk_body

        # Для каждой обогащённой части создаём IndexAPIItem
        for enriched_text, part_msg_ids in enriched_parts:
            # Добавляем перекрытие спереди
            if chunk_overlap:
                final_text = chunk_overlap + " " + enriched_text
            else:
                final_text = enriched_text
            
            result.append(IndexAPIItem(
                page_content=final_text,
                dense_content=final_text,
                sparse_content=final_text,
                message_ids=part_msg_ids,
            ))

            # Обновляем previous_chunk_text для следующего чанка
            # Берём последние OVERLAP_SIZE символов ИТОГОВОГО текста (final_text)
            previous_chunk_text = slice_tail(final_text, OVERLAP_SIZE)

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
