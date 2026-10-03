from pathlib import Path

import yaml

from serviceflow.rag.models import PolicyChunk, PolicyDocument
from serviceflow.rag.utils import calculate_content_hash


def load_policy(path: Path) -> PolicyDocument:
    """加载一篇Markdown政策文档。"""
    text = path.read_text(encoding="utf-8")

    if not text.startswith("---\n"):
        raise ValueError(
            f"政策文件缺少YAML元数据：{path}"
        )

    parts = text.split("---", maxsplit=2)

    if len(parts) != 3:
        raise ValueError(
            f"政策文件的YAML元数据没有正确结束：{path}"
        )

    metadata_text = parts[1]
    content = parts[2].strip()

    metadata = yaml.safe_load(metadata_text)

    if not isinstance(metadata, dict):
        raise ValueError(
            f"政策文件YAML元数据必须是对象：{path}"
        )

    return PolicyDocument.model_validate(
        {
            **metadata,
            "content": content,
            "source": path,
        }
    )


# 加载整个目录
def load_policies(directory: Path) -> list[PolicyDocument]:
    paths = sorted(directory.glob("*.md"))

    if not paths:
        raise ValueError(f"政策目录中没有Markdown文件：{directory}")

    return [load_policy(path) for path in paths]


def _build_chunk(
        policy: PolicyDocument,
        section: str,
        lines: list[str],
        index: int,
) -> PolicyChunk:
    """根据政策章节构造一个知识片段。"""
    content = "\n".join(lines).strip()

    if not content:
        raise ValueError(
            f"政策章节没有正文：{policy.source} - {section}"
        )

    return PolicyChunk(
        chunk_id=f"{policy.policy_id}-{index:03d}",
        policy_id=policy.policy_id,
        title=policy.title,
        section=section,
        content=content,
        category=policy.category,
        version=policy.version,
        effective_date=policy.effective_date,
        status=policy.status,
        applicable_products=policy.applicable_products,
        source=policy.source,
    )


# 分类二级标题
def split_policy(policy: PolicyDocument, ) -> list[PolicyChunk]:
    """按照Markdown二级标题切分一篇政策文档。"""

    # 最终结果
    chunks: list[PolicyChunk] = []
    # 当前正在收集的二级标题
    current_section: str | None = None
    # 当前标题下面的正文行
    current_lines: list[str] = []

    for line in policy.content.splitlines():
        if line.startswith("## "):
            if current_section is not None:
                chunk = _build_chunk(
                    policy=policy,
                    section=current_section,
                    lines=current_lines,
                    index=len(chunks) + 1,
                )
                chunks.append(chunk)

            current_section = line.removeprefix("## ").strip()
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    # 遍历结束后，保存最后一个章节
    if current_section is not None:
        chunk = _build_chunk(
            policy=policy,
            section=current_section,
            lines=current_lines,
            index=len(chunks) + 1,
        )
        chunks.append(chunk)

    if not chunks:
        raise ValueError(
            f"政策文档中没有可切分的二级标题：{policy.source}"
        )

    return chunks


def policy_to_document_chunk(
        policy: PolicyDocument,
) -> PolicyChunk:
    # chunk_id: str
    # policy_id: str
    # title: str
    # section: str
    # content: str
    # category: str
    # version: str
    # effective_date: date
    # status: str
    # applicable_products: str
    # source: Path
    return PolicyChunk(
        chunk_id=f"{policy.policy_id}-FULL",
        policy_id=policy.policy_id,
        title=policy.title,
        section="全文",
        content=policy.content,
        category=policy.category,
        version=policy.version,
        effective_date=policy.effective_date,
        status=policy.status,
        applicable_products=policy.applicable_products,
        source=policy.source,
    )


def main() -> None:
    policies = load_policies(
        Path("knowledge/policies")
    )

    section_chunks: list[PolicyChunk] = []
    document_chunks: list[PolicyChunk] = []

    for policy in policies:
        document_chunks.append(policy_to_document_chunk(policy))

        chunks = split_policy(policy)
        section_chunks.extend(chunks)

    print(f"文档数量：{len(policies)}")
    print(f"片段数量：{len(section_chunks)}")
    print("-" * 50)

    for chunk in section_chunks:
        if chunk.policy_id != "POL-RETURN-003":
            continue

        print(f"片段编号：{chunk.chunk_id}")
        print(f"政策标题：{chunk.title}")
        print(f"章节标题：{chunk.section}")
        print(f"正文：{chunk.content}")
        print(f"来源：{chunk.source}")
        print("-" * 50)

        if chunk.section == "质量问题例外":
            embedding_text = chunk.text_for_embedding()
            content_hash = calculate_content_hash(embedding_text)

            print("text_for_embedding: " + embedding_text)
            print("content_hash: " + content_hash)
            print(f"len(content_hash): {len(content_hash)}")

            print("-" * 50)

    print(f"政策数量: {len(policies)}")
    print(f"整篇文档片段: {len(document_chunks)}")
    print(f"章节片段: {len(section_chunks)}")


if __name__ == "__main__":
    main()
