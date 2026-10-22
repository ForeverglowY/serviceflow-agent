from serviceflow.rag.answer_evaluation import REFUSAL, check_rag_answer
from serviceflow.rag.models import RAGAnswerResult


def case(refusal=False):
    return {"case_id":"TEST", "query":"问题", "expect_refusal":refusal, "review_criteria":["依据政策"]}


def test_valid_citation_needs_review_not_automatic_semantic_pass():
    result = RAGAnswerResult(answer="退款已经审批通过", cited_chunk_ids=["P1"])
    row = check_rag_answer(case(), result, ["P1"])
    assert row["status"] == "REVIEW"
    assert "审批通过" in row["risk_signals"]
    assert row["manual_verdict"] is None


def test_unknown_or_missing_citation_fails():
    for ids in [["FAKE"], []]:
        result = RAGAnswerResult(answer="可以检测", cited_chunk_ids=ids)
        assert check_rag_answer(case(), result, ["P1"])["status"] == "FAIL"


def test_refusal_has_no_citations_and_matches_expected():
    result = RAGAnswerResult(answer=REFUSAL, cited_chunk_ids=[])
    assert check_rag_answer(case(True), result, ["P1"])["status"] == "REVIEW"
    assert check_rag_answer(case(False), result, ["P1"])["status"] == "FAIL"
    result.cited_chunk_ids = ["P1"]
    assert check_rag_answer(case(True), result, ["P1"])["status"] == "FAIL"


def test_negated_risk_phrase_is_signal_not_automatic_failure():
    result = RAGAnswerResult(answer="不能承诺免费换新", cited_chunk_ids=["P1"])
    row = check_rag_answer(case(), result, ["P1"])
    assert row["status"] == "REVIEW"
    assert "免费换新" in row["risk_signals"]
