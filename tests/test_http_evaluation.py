from unittest.mock import Mock

from serviceflow.http_evaluation import evaluate_http_scenarios


def test_each_scenario_uses_distinct_thread_and_cancel_reuses_it():
    sent = []
    state = {}

    def post(payload):
        sent.append(payload)
        if "20260721002" in payload["message"]:
            state[payload["thread_id"]] = {"decision":"reject", "status":"collecting"}
        elif payload["message"] == "取消售后":
            state[payload["thread_id"]] = {"status":"canceled"}
        return {"status":200, "body":{"answer":"测试回复", "thread_id":payload["thread_id"]}, "trace_id":"trace"}

    def observe(thread):
        return {"ticket_ids":["old-ticket"], "state":state.get(thread, {})}

    result = evaluate_http_scenarios(post, observe)
    assert result["counts"]["REVIEW"] == 4
    assert len({payload["thread_id"] for payload in sent}) == 4
    assert sent[-1]["thread_id"] == sent[-2]["thread_id"]
    assert all(payload["message"] != "确认" for payload in sent)


def test_unexpected_ticket_stops_without_deleting_anything():
    observe = Mock(side_effect=[{"ticket_ids":[], "state":{}}, {"ticket_ids":["unexpected"], "state":{}}])
    post = Mock(return_value={"status":200, "body":{"answer":"回复", "thread_id":"wrong"}})
    report = evaluate_http_scenarios(post, observe)
    assert report["cases"][0]["status"] == "FAIL"
    assert report["cases"][0]["unexpected_ticket_ids"] == ["unexpected"]
    assert report["not_run"] == 3
    post.assert_called_once()


def test_observation_error_stops_before_request_and_hides_detail():
    post = Mock()
    report = evaluate_http_scenarios(post, Mock(side_effect=RuntimeError("private credentials")))
    post.assert_not_called()
    assert report["counts"]["ERROR"] == 1
    assert "private credentials" not in str(report)
