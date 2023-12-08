import aiohttp
from settings import *
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(RETRIES), wait=wait_exponential(multiplier=BACKOFF, min=DELAY), reraise=True, retry_error_callback=logger.error)
async def call_api(
    method: str,
    url: str,
    headers: dict | None = None,
    params: dict | None = None,
    data: dict | aiohttp.FormData | None = None,
    stream: bool = False,
):
    kwargs = {
        "headers": headers,
        "params": params,
    }
    if data:
        if isinstance(data, dict):
            kwargs["json"] = data
        elif isinstance(data, aiohttp.FormData):
            kwargs["data"] = data

    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method=method,
            url=url,
            **kwargs,
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"API call failed with status {resp.status}: {await resp.text()}")
            if stream:
                async for line in resp.content:
                    chunk = line.decode("utf-8").strip()
                    yield chunk
            else:
                if resp.headers["Content-Type"] == "application/json":
                    yield await resp.json()
                else:
                    yield await resp.content.read()