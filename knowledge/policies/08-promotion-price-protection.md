---
policy_id: POL-PRICE-001
title: 促销商品与价格保护政策
category: price_protection
version: "1.0"
effective_date: 2026-04-01
status: active
applicable_products: general
---

# 促销商品与价格保护政策

> 本文档为 ServiceFlow Agent 项目的模拟政策，仅用于学习和测试。

## 价格保护期限

商品页面明确标注支持价格保护时，用户可以在订单支付后七个自然日内申请。未标注价格保护的商品不自动适用。

## 差价计算

差价以同一用户、同一商品规格、相同购买数量和相同优惠条件下的实际支付金额为基础计算。

## 不适用情形

限时秒杀、赠品变化、会员专享价、优惠券差异、地区补贴和第三方渠道价格不参与价格保护。

## 结果约束

模型可以解释政策，但是否符合价格保护条件以及具体差额必须由Python业务规则和订单数据计算。
