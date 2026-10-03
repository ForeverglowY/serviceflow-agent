from datetime import date
from pathlib import Path

import pytest

from serviceflow.rag.loader import (
    load_policies,
    load_policy,
    policy_to_document_chunk,
    split_policy,
)


POLICY_DIRECTORY = Path("knowledge/policies")


def test_load_all_policies_in_stable_order() -> None:
    policies = load_policies(POLICY_DIRECTORY)

    assert len(policies) == 20
    assert policies[0].policy_id == "POL-RETURN-001"
    assert policies[-1].policy_id == "POL-TRADEIN-001"


def test_load_policy_parses_metadata_and_content() -> None:
    path = POLICY_DIRECTORY / "03-earphone-hygiene-return.md"

    policy = load_policy(path)

    assert policy.policy_id == "POL-RETURN-003"
    assert policy.title == "耳机类商品退换货特别规则"
    assert policy.effective_date == date(2026, 3, 1)
    assert policy.applicable_products == "earphone"
    assert policy.source == path
    assert "## 已拆封商品" in policy.content


def test_split_earphone_policy_by_second_level_heading() -> None:
    policy = load_policy(
        POLICY_DIRECTORY / "03-earphone-hygiene-return.md"
    )

    chunks = split_policy(policy)

    assert [chunk.chunk_id for chunk in chunks] == [
        "POL-RETURN-003-001",
        "POL-RETURN-003-002",
        "POL-RETURN-003-003",
        "POL-RETURN-003-004",
    ]
    assert [chunk.section for chunk in chunks] == [
        "未拆封商品",
        "已拆封商品",
        "质量问题例外",
        "所需信息",
    ]
    assert "不支持七天无理由退货" in chunks[1].content
    assert chunks[1].title == policy.title
    assert chunks[1].source == policy.source


def test_chunk_builds_embedding_text() -> None:
    policy = load_policy(
        POLICY_DIRECTORY / "03-earphone-hygiene-return.md"
    )
    chunk = split_policy(policy)[2]

    assert chunk.text_for_embedding() == (
        "政策：耳机类商品退换货特别规则\n"
        "章节：质量问题例外\n"
        f"内容：{chunk.content}"
    )


def test_all_policies_produce_eighty_chunks() -> None:
    chunks = [
        chunk
        for policy in load_policies(POLICY_DIRECTORY)
        for chunk in split_policy(policy)
    ]

    assert len(chunks) == 80
    assert len({chunk.chunk_id for chunk in chunks}) == 80


def test_policy_to_document_chunk_preserves_document() -> None:
    policy = load_policy(
        POLICY_DIRECTORY / "03-earphone-hygiene-return.md"
    )

    chunk = policy_to_document_chunk(policy)

    assert chunk.chunk_id == "POL-RETURN-003-FULL"
    assert chunk.policy_id == policy.policy_id
    assert chunk.title == policy.title
    assert chunk.section == "全文"
    assert chunk.content == policy.content
    assert chunk.category == policy.category
    assert chunk.version == policy.version
    assert chunk.effective_date == policy.effective_date
    assert chunk.status == policy.status
    assert chunk.applicable_products == policy.applicable_products
    assert chunk.source == policy.source


def test_all_policies_produce_twenty_document_chunks() -> None:
    chunks = [
        policy_to_document_chunk(policy)
        for policy in load_policies(POLICY_DIRECTORY)
    ]

    assert len(chunks) == 20
    assert len({chunk.chunk_id for chunk in chunks}) == 20
    assert all(
        chunk.chunk_id.endswith("-FULL")
        for chunk in chunks
    )


def test_load_policy_rejects_missing_front_matter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.md"
    path.write_text("# 没有元数据的政策", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="政策文件缺少YAML元数据",
    ):
        load_policy(path)


def test_load_policy_rejects_unclosed_front_matter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.md"
    path.write_text(
        "---\npolicy_id: POL-INVALID\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="YAML元数据没有正确结束",
    ):
        load_policy(path)


def test_split_policy_rejects_empty_section(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.md"
    path.write_text(
        """---
policy_id: POL-INVALID
title: 错误政策
category: test
version: "1.0"
effective_date: 2026-01-01
status: active
applicable_products: general
---

# 错误政策

## 空章节

## 下一个章节

这里有正文。
""",
        encoding="utf-8",
    )
    policy = load_policy(path)

    with pytest.raises(
        ValueError,
        match="政策章节没有正文.*空章节",
    ):
        split_policy(policy)
