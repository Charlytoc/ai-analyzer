import chromadb
import os
import hashlib
from server.utils.printer import Printer

printer = Printer(name="vector_store")

MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "media")

VECTOR_STORAGE_PATH = os.environ.get(
    "VECTOR_STORAGE_PATH", os.path.join(MEDIA_ROOT, "vector_storage/")
)

CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8004))
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")

if not os.path.exists(VECTOR_STORAGE_PATH):
    os.makedirs(VECTOR_STORAGE_PATH)

ChromaNotInitializedException = Exception("Chroma not yet initialized!")


class Chunk:
    def __init__(self, text: str):
        self.text = text
        self.id = self.generate_id(text)
        self.metadata = {
            "id": self.id,
        }

    def generate_id(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def __repr__(self):
        return f"Chunk(id={self.id[:8]}, words={len(self.text.split())})"


class ChromaManager:
    client = None

    def __init__(self) -> None:
        printer.yellow(
            "🔄 Cliente de Chroma se va a conectar en HOST: ",
            CHROMA_HOST,
            " y en el puerto: ",
            CHROMA_PORT,
        )
        self.client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
        )

    def chunkify(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[Chunk]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk))
            start += chunk_size - chunk_overlap
        return chunks

    def heartbeat(self) -> str:
        if self.client is None:
            raise ChromaNotInitializedException
        return self.client.heartbeat()

    def get_or_create_collection(self, collection_name: str):
        collection = self.client.get_or_create_collection(name=collection_name)
        return collection

    def upsert_chunk(
        self, collection_name: str, chunk_text: str, chunk_id: str, metadata: dict = {}
    ):
        collection = self.get_or_create_collection(collection_name)
        collection.upsert(documents=[chunk_text], ids=[chunk_id], metadatas=[metadata])

    def bulk_upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
    ):
        collection = self.get_or_create_collection(collection_name)
        documents = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        collection.upsert(documents=documents, ids=chunk_ids, metadatas=metadatas)

    def get_results(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 4,
        search_string: str = "",
        # where: dict = {},
    ):
        collection = self.get_or_create_collection(collection_name)

        if search_string:
            return collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where_document={"$contains": search_string},
                # where=where,
            )
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
        )

    def get_collection_or_none(self, collection_name: str):
        try:
            return self.client.get_collection(name=collection_name)
        except Exception as e:
            print(e, "EXCEPTION TRYING TO GET COLLECTION")
            return None

    def delete_collection(self, collection_name: str):
        print("Deleting collection from chroma")
        try:
            self.client.delete_collection(collection_name)
            print("DELETED SUCCESSFULLY")
        except Exception as e:
            print(e, "EXCEPTION TRYING TO DELETE COLLECTION")

    def delete_chunk(self, collection_name: str, chunk_id: str):
        # TODO: This is bad, if the collection doesn't exist, ignore
        collection = self.get_or_create_collection(collection_name)
        collection.delete(ids=[chunk_id])

    def bulk_delete_chunks(self, collection_name: str, chunk_ids: list[str]):
        collection = self.get_collection_or_none(collection_name)
        if collection:
            collection.delete(ids=chunk_ids)


def get_chroma_client():
    if not hasattr(get_chroma_client, "_client"):
        get_chroma_client._client = ChromaManager()
    return get_chroma_client._client
