# -*- coding: utf-8 -*-
"""
Alpha158 Enhanced — 在保留原有 handler_pb_turn.Alpha158 特征的基础上“增量扩展”
- 不改动你原始的特征集合；仅在其后追加新因子。
- 兼容 qlib 表达式：仅使用 Ref/Mean/Std/Log/Quantile/Rank/Slope/Rsquare/Resi/Max/Min 等原生算子
- 新增窗口：5/10/20/30/60（rolling 家族）
- 新增主题：换手深化、估值×动量修正、流动性冲击（Amihud）

用法：
from handler_pb_turn_enhanced import Alpha158Enhanced as Alpha158  # 若要直接替换原类名
或在你的配置里将 handler 类改为 Alpha158Enhanced
"""

# 优先相对导入你上传的原始类；若失败再尝试绝对名
try:
    from .handler_pb_turn import Alpha158 as BaseAlpha158
except Exception:
    from handler_pb_turn import Alpha158 as BaseAlpha158  # type: ignore


class Alpha158Enhanced(BaseAlpha158):
    """在原 Alpha158 基础上追加特征，不破坏原始行为"""
    def get_feature_config(self):
        # 先拿到原始 features & names
        orig_fields, orig_names = super().get_feature_config()

        # 组装新增特征
        new_fields, new_names = [], []
        def add(expr, name=None):
            new_fields.append(expr)
            new_names.append(name or expr)

        # ---------- 1) 换手率深化（$turn） ----------
        # 窗口覆盖：5/10/20/30/60；基准相对：20、60；Δ1d 与 Δ5d 平滑
        TO = "$turn"
        to_windows = [5, 10, 20, 30, 60]
        to_bases   = [20, 60]

        # 基础统计
        for w in to_windows:
            add(f"Mean({TO}, {w})", f"TO_MEAN_{w}d")
            add(f"Std({TO}, {w})",  f"TO_STD_{w}d")

        # 相对/标准化到基准
        for b in to_bases:
            for w in to_windows:
                add(f"(Mean({TO}, {w}))/(Mean({TO}, {b})+1e-12)", f"TO_REL_{w}d_over_{b}d")
                add(f"(Mean({TO}, {w})-Mean({TO}, {b}))/(Std({TO}, {b})+1e-12)", f"TO_ZSCORE_{w}d_over_{b}d")
            # 市场均值相对（回退用法，行业中性化可在上游 groupby 实现）
            add(f"{TO}/(Mean({TO}, {b})+1e-12)-1", f"TO_CHG_RATE_over_{b}d")

        # 换手的动量/变化
        add(f"{TO}/(Ref({TO},1)+1e-12) - 1", "TO_DELTA_1d")
        add(f"Mean({TO}/(Ref({TO},1)+1e-12) - 1, 5)", "TO_DELTA_MEAN_5d")

        # 锁仓动量：低换手 × 短期动量（Rank 明确窗口）
        add("Mean($close/Ref($close,1)-1,5) * (1 - Rank($turn, 20))", "LOCK_MOM_5")

        # Δ换手 × 动量（抱团松动捕捉）
        add("($turn/(Ref($turn,1)+1e-12) - 1) * Mean($close/Ref($close,1)-1,5)", "DELTA_TO_MOM_5")

        # ---------- 2) 估值 × 动量修正（pe_ttm / pb） ----------
        # Log/倒数（EP、BM）+ 20/60 平滑；基准用 60；市场均值相对项
        def add_val(raw, base):
            expr_log = f"Log({raw}+1e-12)"
            nm = base
            for w in (20, 60):
                add(f"Mean({expr_log}, {w})", f"LOG_{nm}_MEAN_{w}d")
                add(f"Std({expr_log}, {w})",  f"LOG_{nm}_STD_{w}d")
            # 60 日相对 & ZScore
            add(f"(Mean({expr_log}, 20))/(Mean({expr_log}, 60)+1e-12)", f"LOG_{nm}_REL_20d_over_60d")
            add(f"(Mean({expr_log}, 20)-Mean({expr_log}, 60))/(Std({expr_log}, 60)+1e-12)", f"LOG_{nm}_ZSCORE_20d_over_60d")
            # 市场均值相对的回退实现
            add(f"{expr_log}/(Mean({expr_log}, 60)+1e-12)-1", f"LOG_{nm}_CHG_RATE_over_60d")

        add_val("$pe_ttm", "PE_TTM")
        add_val("$pb", "PB")
        # 倒数（盈利收益率/账面市值比）
        def add_inv(raw, base):
            expr = f"1.0/({raw}+1e-12)"
            for w in (20, 60):
                add(f"Mean({expr}, {w})", f"{base}_MEAN_{w}d")
                add(f"Std({expr}, {w})",  f"{base}_STD_{w}d")
            add(f"(Mean({expr}, 20))/(Mean({expr}, 60)+1e-12)", f"{base}_REL_20d_over_60d")
            add(f"(Mean({expr}, 20)-Mean({expr}, 60))/(Std({expr}, 60)+1e-12)", f"{base}_ZSCORE_20d_over_60d")
            add(f"{expr}/(Mean({expr}, 60)+1e-12)-1", f"{base}_CHG_RATE_over_60d")
        add_inv("$pe_ttm", "EP")
        add_inv("$pb", "BM")

        # 估值×动量的踩踏修正（用市场均值回退）
        add("(1/($pe_ttm+1e-12)) / (Mean(1/($pe_ttm+1e-12),60)+1e-12)", "EP_MKT_REL_60")
        add("(Log($pe_ttm+1e-12) - Mean(Log($pe_ttm+1e-12),60)) * Min(Mean($close/Ref($close,1)-1,5),0)", "EXPN_MOM_RISK_MKT")

        # ---------- 3) 流动性冲击（Amihud proxy） ----------
        # ret_abs / amount 的均值；并构造 Rank 惩罚（指定窗口）
        ret_abs_over_amt_5  = "Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 5)"
        ret_abs_over_amt_20 = "Mean(Abs($close/Ref($close,1)-1)/($amount + 1e-12), 20)"
        add(ret_abs_over_amt_5,  "AMIH_5")
        add(ret_abs_over_amt_20, "AMIH_20")
        add(f"Rank({ret_abs_over_amt_5}, 20)",  "IMPACT_PENALTY_5")
        add(f"Rank({ret_abs_over_amt_20}, 60)", "IMPACT_PENALTY_20")

        # ---------- 4) 扩展 rolling 家族（5/10/20/30/60） ----------
        roll_windows = [5, 10, 20, 30, 60]
        for d in roll_windows:
            # 动量/均值/波动
            add(f"Ref($close, {d})/$close", f"Ref($close, {d})/$close")
            add(f"Mean($close, {d})/$close", f"Mean($close, {d})/$close")
            add(f"Std($close, {d})/$close",  f"Std($close, {d})/$close")
            # 线性拟合
            add(f"Slope($close, {d})/$close", f"Slope($close, {d})/$close")
            add(f"Rsquare($close, {d})",      f"Rsquare($close, {d})")
            add(f"Resi($close, {d})/$close",  f"Resi($close, {d})/$close")
            # 量能
            add(f"Mean($volume, {d})/($volume+1e-12)", f"Mean($volume, {d})/($volume+1e-12)")
            add(f"Std($volume, {d})/($volume+1e-12)",  f"Std($volume, {d})/($volume+1e-12)")

            # 通道/分位：适配中长窗
            if d in (20, 30, 60):
                add(f"Max($high, {d})/$close", f"Max($high, {d})/$close")
                add(f"Min($low, {d})/$close",  f"Min($low, {d})/$close")
                add(f"Quantile($close, {d}, 0.8)/$close", f"Quantile($close, {d}, 0.8)/$close")
                add(f"Quantile($close, {d}, 0.2)/$close", f"Quantile($close, {d}, 0.2)/$close")

            # RSV：常用 10/14/20/30
            if d in (10, 14, 20, 30):
                add(f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)", f"RSV_{d}")

            # RSI 家族（涨跌分解）
            add(f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)",
                f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)")
            add(f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)",
                f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)")
            add(f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)",
                f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)")

            # 上涨天数/下跌天数/差值
            add(f"Mean($close>Ref($close, 1), {d})", f"Mean($close>Ref($close, 1), {d})")
            add(f"Mean($close<Ref($close, 1), {d})", f"Mean($close<Ref($close, 1), {d})")
            add(f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})",
                f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})")

        # 合并并去重：保持“原有在前，新加在后”的顺序
        seen = set()
        out_fields, out_names = [], []
        for f, n in zip(orig_fields + new_fields, orig_names + new_names):
            if f not in seen:
                seen.add(f)
                out_fields.append(f)
                out_names.append(n)

        return out_fields, out_names


# 兼容：若外部仍引用 Alpha158vwap，可提供同名扩展
class Alpha158vwap(Alpha158Enhanced):
    def get_label_config(self):
        return ["Ref($adjclose, -5)/Ref($adjclose, -1) - 1"], ["LABEL0"]
