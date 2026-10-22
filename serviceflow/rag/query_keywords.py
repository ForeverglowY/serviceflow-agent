"""统一中文分词规则，不根据案例编号或预期片段调整关键词。"""
import re

import jieba

STOP_WORDS = frozenset({"可以", "应该", "怎么", "怎样", "什么", "是否", "以后", "已经",
                        "现在", "还是", "一般", "一下", "这个", "那个", "帮我", "请问", "你们", "我的"})


def extract_query_keywords(query: str) -> list[str]:
    tokens = jieba.cut_for_search(query, HMM=False)
    # 去单字、标点、常见疑问词，保留顺序并去重。
    return list(dict.fromkeys(
        token.lower() for token in tokens
        if len(token) >= 2 and token not in STOP_WORDS
        and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9_-]+", token)
    ))
