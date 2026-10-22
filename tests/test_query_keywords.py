from serviceflow.rag.query_keywords import extract_query_keywords


def test_keywords_keep_meaningful_words_and_remove_duplicates_and_questions():
    words = extract_query_keywords("耳机进水，耳机可以免费换新吗？")
    assert "耳机" in words
    assert "进水" in words
    assert "免费" in words
    assert "可以" not in words
    assert len(words) == len(set(words))
    assert all(len(word) >= 2 for word in words)


def test_empty_and_punctuation_have_no_keywords():
    assert extract_query_keywords("   ？！") == []


def test_rules_do_not_depend_on_eval_case_or_expected_ids():
    words = extract_query_keywords("电子发票抬头写错怎样修改？")
    assert "发票" in words
    assert "抬头" in words
    assert "怎样" not in words
