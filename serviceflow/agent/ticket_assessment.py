from datetime import date

from serviceflow.llm.service import classify_return_reason
from serviceflow.ticket.rules import TicketEligibilityResult, check_return_eligibility
from serviceflow.ticket.service import load_return_eligibility_input


# “退换货评估”的编排函数
def assess_return_request(order_id: str, issue_description: str, requested_action: str,
                          requested_at: date) -> TicketEligibilityResult:
    # 提取退换货原因
    return_reason_result = classify_return_reason(issue_description=issue_description,
                                                  requested_action=requested_action, )

    # 数据库补齐商品类别、签收日期
    return_eligibility_input = load_return_eligibility_input(order_id=order_id, issue_description=issue_description,
                                                             requested_action=requested_action,
                                                             return_reason=return_reason_result.return_reason,
                                                             requested_at=requested_at)
    # 用确定的规则得出判断
    return check_return_eligibility(return_eligibility_input)
