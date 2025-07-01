import os
import redis
from dotenv import load_dotenv


load_dotenv()


class RedisCache:
    def __init__(self):
        self.client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=True,
        )

    # ------------ Strings ------------
    def exists(self, key: str) -> bool:
        return self.client.exists(key) == 1

    def get(self, key: str) -> str | None:
        return self.client.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.client.set(key, value, ex=ex)

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def flush_all(self):
        self.client.flushall()

    def hset(self, name: str, key: str, value: str) -> None:
        self.client.hset(name, key, value)

    def hget(self, name: str, key: str) -> str | None:
        return self.client.hget(name, key)

    def hdel(self, name: str, key: str) -> None:
        self.client.hdel(name, key)

    def hgetall(self, name: str) -> dict:
        return self.client.hgetall(name)

    def rpush(self, key: str, value: str) -> None:
        self.client.rpush(key, value)

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        return self.client.lrange(key, start, end)

    def lpop(self, key: str) -> str | None:
        return self.client.lpop(key)

    def lset(self, key: str, index: int, value: str) -> None:
        self.client.lset(key, index, value)

    def lrem(self, key: str, count: int, value: str) -> None:
        self.client.lrem(key, count, value)

    def llen(self, key: str) -> int:
        return self.client.llen(key)
