from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


# 政策文档模型
class PolicyDocument(BaseModel):
    policy_id: str = Field(min_length=1, max_length=50)
    title: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=50)
    version: str = Field(min_length=1, max_length=20)
    effective_date: date
    status: str = Field(min_length=1, max_length=20)
    applicable_products: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1)
    source: Path


# 政策片段模型
class PolicyChunk(BaseModel):
    chunk_id: str
    policy_id: str
    title: str
    section: str
    content: str
    category: str
    version: str
    effective_date: date
    status: str
    applicable_products: str
    source: Path

    def text_for_embedding(self) -> str:
        return f"政策：{self.title}\n章节：{self.section}\n内容：{self.content}"


class RetrievalEvalCase(BaseModel):
    case_id: str
    query: str
    expected_chunk_ids: list[str]
    require_all: bool = False


class RAGAnswerResult(BaseModel):
    # answer
    # - str
    # - 最少1个字符
    # - 最多2000个字符
    #
    # cited_chunk_ids
    # - list[str]
    # - 默认空列表
    # - 最多5个引用
    answer: str = Field(min_length=1, max_length=2000)
    cited_chunk_ids: list[str] = Field(default_factory=list, max_length=5)

    model_config = ConfigDict(extra="forbid")