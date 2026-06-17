import json
from collections import Counter, defaultdict
from pathlib import Path


IN_PATH = Path("/workspace/FIDESlib-GPT/reports/wp4y2q_ppmm_source_sign_sweep_report.json")
OUT_PATH = Path("/workspace/FIDESlib-GPT/reports/wp4y2q_valid_variant_summary_report.json")


def main():
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))

    variants = data.get("variant_results", [])
    num_cases = int(data.get("num_cases", 4))

    completed = []
    partial = []
    failed = []

    error_counter = Counter()
    error_examples = defaultdict(list)

    for vr in variants:
        summary = vr.get("summary", {})
        num_completed = int(summary.get("num_cases_completed", 0))
        num_errors = int(summary.get("num_errors", 0))

        if num_completed == num_cases and num_errors == 0:
            completed.append(vr)
        elif num_completed > 0:
            partial.append(vr)
        else:
            failed.append(vr)

        for err in vr.get("errors", []):
            msg = err.get("error", "")
            # Compact error key.
            key = msg.split("\\n")[0][:240]
            error_counter[key] += 1
            if len(error_examples[key]) < 5:
                error_examples[key].append({
                    "variant_id": vr.get("variant", {}).get("variant_id"),
                    "case": err.get("case"),
                    "error": msg,
                })

    ref_names = [
        "U_matmul_V",
        "U_T_matmul_V",
        "U_matmul_V_T",
        "V_matmul_U",
        "V_T_matmul_U",
        "U_T_matmul_V_T",
    ]

    def best_key(vr):
        s = vr["summary"]
        return (
            -max(s["match_counts"].values()),
            s["total_l1"]["U_matmul_V"],
            s["total_l1"]["U_T_matmul_V"],
            vr["variant"]["variant_id"],
        )

    ranked_completed = sorted(completed, key=best_key)

    solved_completed = []
    for vr in completed:
        for ref in ref_names:
            if vr["summary"]["match_counts"][ref] == num_cases:
                solved_completed.append({
                    "variant_id": vr["variant"]["variant_id"],
                    "reference": ref,
                    "variant": vr["variant"],
                    "summary": vr["summary"],
                })

    report = {
        "experiment": "wp4y2q_valid_variant_summary",
        "purpose": (
            "Re-rank Y2q PP-MM source/sign sweep while excluding variants that completed zero cases. "
            "The original Y2q ranking can be misleading because failed variants have total_l1=0."
        ),
        "input_report": str(IN_PATH),
        "counts": {
            "total_variants": len(variants),
            "completed_all_cases_no_errors": len(completed),
            "partial_completed": len(partial),
            "failed_zero_completed": len(failed),
            "cases_per_variant": num_cases,
        },
        "error_summary": {
            "unique_error_count": len(error_counter),
            "top_errors": [
                {
                    "error_key": key,
                    "count": count,
                    "examples": error_examples[key],
                }
                for key, count in error_counter.most_common(10)
            ],
        },
        "valid_completed_results": {
            "any_completed_variant_solves_U_matmul_V": any(
                x["reference"] == "U_matmul_V" for x in solved_completed
            ),
            "any_completed_variant_solves_any_reference": len(solved_completed) > 0,
            "solved_completed_variants": solved_completed[:20],
            "best_completed_variant": None if not ranked_completed else {
                "variant_id": ranked_completed[0]["variant"]["variant_id"],
                "variant": ranked_completed[0]["variant"],
                "summary": ranked_completed[0]["summary"],
            },
            "top20_completed_variants": [
                {
                    "variant_id": vr["variant"]["variant_id"],
                    "variant": vr["variant"],
                    "summary": vr["summary"],
                }
                for vr in ranked_completed[:20]
            ],
        },
    }

    if len(completed) == 0:
        decision = {
            "value": "y2q_sweep_execution_invalid_no_completed_variants",
            "next_step": "WP4-Y2q-fix2",
            "next_goal": "Fix the execution error before interpreting PP-MM source/sign sweep.",
            "zh": "没有任何 variant 完整跑完，必须先修执行错误，不能进入 symbolic 结论。",
        }
    elif solved_completed:
        decision = {
            "value": "valid_variant_found",
            "next_step": "WP4-Y2r",
            "next_goal": "Promote solved completed variant to larger diagnostics.",
            "zh": "存在完整跑完且匹配参考的 variant，下一步扩大测试。",
        }
    else:
        decision = {
            "value": "valid_variants_completed_but_no_exact_solution",
            "next_step": "WP4-Y2r",
            "next_goal": "Now switch to plaintext symbolic Algorithm 3 simulation using the best completed variant.",
            "zh": "已有有效完成的 variants 但没有 exact 解，此时才进入明文符号模拟。",
        }

    report["global_decision"] = decision

    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2q valid variant summary")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
