from types import SimpleNamespace
from unittest.mock import Mock

from openai.types.chat import ChatCompletionMessageFunctionToolCall

from serviceflow.tool_evaluation import evaluate_tool_cases


def response(name="get_order", arguments='{"order_id":"ORDER"}'):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None, tool_calls=[ChatCompletionMessageFunctionToolCall(
            id="call-test", type="function", function={"name": name, "arguments": arguments},
        )],
    ))])


def dataset():
    return [{"case_id":"TEST", "mode":"live", "turns":["查订单ORDER"],
             "expected":{"ticket_created":False, "order_id":"ORDER"}}]


def test_tool_and_parameter_match_passes_without_execution():
    request = Mock(return_value=response())
    report = evaluate_tool_cases(dataset(), {"TEST":{"required":["get_order"], "allowed":["get_order"]}}, request)
    assert report["counts"]["PASS"] == 1
    assert report["cases"][0]["actual_parameters"] == [{"tool":"get_order", "order_id":"ORDER"}]
    request.assert_called_once_with("查订单ORDER")


def test_wrong_order_unknown_tool_and_bad_json_fail():
    for value in [response(arguments='{"order_id":"WRONG"}'), response(name="delete_database"), response(arguments="bad")]:
        report = evaluate_tool_cases(dataset(), {"TEST":{"required":["get_order"], "allowed":["get_order"]}}, Mock(return_value=value))
        assert report["counts"]["FAIL"] == 1


def test_missing_required_tool_and_empty_reply_do_not_pass():
    value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None))])
    for label in [{"required":["get_order"], "allowed":["get_order"]}, {"required":[], "allowed":[]}]:
        report = evaluate_tool_cases(dataset(), {"TEST":label}, Mock(return_value=value))
        assert report["counts"]["FAIL"] == 1


def test_model_error_is_recorded_without_private_message():
    report = evaluate_tool_cases(dataset(), {"TEST":{"required":[], "allowed":[]}}, Mock(side_effect=RuntimeError("private-details")))
    assert report["counts"]["ERROR"] == 1
    assert "private-details" not in str(report)
