"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class LocalHashEmbeddings:
    """Deterministic offline embeddings for local lab runs."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vec)) or 1.0
        return [value / norm for value in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


def _build_local_llm(temperature: float = 0.0):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class LocalExtractiveChatModel(BaseChatModel):
        temperature: float = 0.0

        @property
        def _llm_type(self) -> str:
            return "local-extractive-rag"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            question = getattr(messages[-1], "content", "") if messages else ""
            system_text = "\n".join(
                getattr(message, "content", str(message))
                for message in messages[:-1]
            )
            context = system_text.split("Context:", 1)[-1] if "Context:" in system_text else system_text
            answer = self._answer_from_context(question, context)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

        @staticmethod
        def _answer_from_context(question: str, context: str) -> str:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", context)
                if len(sentence.strip()) > 20
            ]
            stop_words = {
                "what", "are", "the", "is", "in", "how", "does", "do", "of",
                "and", "to", "a", "an", "why", "with", "for", "used", "explain",
            }
            q_tokens = {
                token
                for token in re.findall(r"[a-zA-Z0-9]+", question.lower())
                if token not in stop_words
            }

            def score(sentence: str) -> int:
                s_tokens = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                return len(q_tokens & s_tokens)

            ranked = sorted(sentences, key=score, reverse=True)
            selected = [sentence for sentence in ranked[:3] if score(sentence) > 0]
            if not selected and ranked:
                selected = ranked[:1]
            if not selected:
                return "I do not know based on the provided context."
            return " ".join(selected)

    return LocalExtractiveChatModel(temperature=temperature)


def _build_gguf_llm(temperature: float = 0.0):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    model_path = (
        Path(__file__).parent.parent.parent
        / "models"
        / "smollm2-360m-instruct-gguf"
        / "smollm2-360m-instruct-q8_0.gguf"
    )

    class GGUFChatModel(BaseChatModel):
        temperature: float = 0.0
        max_tokens: int = 80
        _client: object = PrivateAttr(default=None)

        @property
        def _llm_type(self) -> str:
            return "smollm2-360m-instruct-gguf"

        def _load_client(self):
            if self._client is None:
                from llama_cpp import Llama

                if not model_path.exists():
                    raise FileNotFoundError(
                        f"GGUF model not found at {model_path}. "
                        "Download it from HuggingFaceTB/SmolLM2-360M-Instruct-GGUF first."
                    )
                self._client = Llama(
                    model_path=str(model_path),
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False,
                )
            return self._client

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            client = self._load_client()
            chat_messages = [
                {"role": getattr(message, "type", "user"), "content": getattr(message, "content", str(message))}
                for message in messages
            ]
            for message in chat_messages:
                if message["role"] == "human":
                    message["role"] = "user"
                elif message["role"] == "ai":
                    message["role"] = "assistant"
            output = client.create_chat_completion(
                messages=chat_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )
            content = output["choices"][0]["message"]["content"].strip()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    return GGUFChatModel(temperature=temperature)


def get_llm(provider: str = None, temperature: float = 0.0):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "local":
        return _build_local_llm(temperature=temperature)

    if provider in ("hf_local", "gguf"):
        return _build_gguf_llm(temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter không có API embeddings riêng → dùng local hash embeddings
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    if provider in ("local", "hf_local", "gguf", "openrouter"):
        return LocalHashEmbeddings()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
