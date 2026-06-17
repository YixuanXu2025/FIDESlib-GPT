import json
from pathlib import Path


def nonzero_positions(M):
    if M is None:
        return None
    out = []
    for i, row in enumerate(M):
        if row is None:
            return None
        for j, v in enumerate(row):
            if int(v) != 0:
                out.append([i, j, int(v)])
    return out


def shape_signature(case):
    got = case["final"]["decoded_matrix"]
    ref = case["reference_U_matmul_V"]

    got_nz = nonzero_positions(got)
    ref_nz = nonzero_positions(ref)

    return {
        "name": case["name"],
        "zh": case.get("zh"),
        "reference": ref,
        "decoded": got,
        "diff": case["analysis"].get("diff"),
        "exact": case["analysis"].get("exact"),
        "max_abs_err": case["analysis"].get("max_abs_err"),
        "ref_nonzero_count": None if ref_nz is None else len(ref_nz),
        "got_nonzero_count": None if got_nz is None else len(got_nz),
        "ref_nonzero_positions": ref_nz,
        "got_nonzero_positions": got_nz,
    }


def classify_case(sig):
    if sig["exact"]:
        return "exact"

    ref_count = sig["ref_nonzero_count"]
    got_count = sig["got_nonzero_count"]

    if got_count is None:
        return "decode_or_row_failure"

    if ref_count == 1 and got_count == 1:
        return "single_nonzero_wrong_location_or_value"

    if ref_count == 1 and got_count > 1:
        if got_count >= 12:
            return "one_hot_dense_diffusion"
        return "one_hot_sparse_diffusion"

    if got_count == 0 and ref_count and ref_count > 0:
        return "lost_signal_all_zero"

    return "general_wrong"


def main():
    in_path = Path("/workspace/FIDESlib-GPT/reports/wp4x6_full_paper_pipeline_final_probe_report.json")

    if not in_path.exists():
        raise FileNotFoundError(f"missing X6 report: {in_path}")

    x6 = json.loads(in_path.read_text(encoding="utf-8"))

    cases = []
    for c in x6["cases"]:
        sig = shape_signature(c)
        sig["classification"] = classify_case(sig)
        cases.append(sig)

    classification_counts = {}
    for c in cases:
        classification_counts[c["classification"]] = classification_counts.get(c["classification"], 0) + 1

    report = {
        "experiment": "wp4x7a_analyze_x6_failure_patterns",
        "purpose": (
            "Analyze X6 decoded outputs case-by-case before changing HE code. "
            "This determines whether failures look like transpose/placement errors or dense diffusion."
        ),
        "status": "analysis_only_no_code_change",
        "source_report": str(in_path),
        "x6_summary": x6.get("summary"),
        "classification_counts": classification_counts,
        "cases": cases,
        "interpretation": {
            "one_hot_dense_diffusion": {
                "meaning": "A one-hot reference produces many nonzero decoded entries.",
                "next": "Do not tune final compress/decrypt. Check PP-MM orientation and component/order conventions.",
                "zh": "one-hot 输出扩散，说明不是最终解码问题，而是 Algorithm 3 中间矩阵语义仍未对齐。",
            },
            "single_nonzero_wrong_location_or_value": {
                "meaning": "A one-hot reference remains one-hot but lands at the wrong place/value.",
                "next": "Likely transpose/rotation/placement convention error.",
                "zh": "如果只有一个非零但位置错，则重点调 transpose/placement。",
            },
        },
        "next_step": {
            "name": "WP4-X7b",
            "goal": (
                "Run a controlled convention sweep over PP-MM orientation and component order on the same "
                "one-hot cases, while reusing the paper-shaped pipeline."
            ),
            "zh": "下一步做约定扫描，而不是盲改单个位置。",
        },
    }

    out_path = Path("/workspace/FIDESlib-GPT/reports/wp4x7a_analyze_x6_failure_patterns_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7a analyze X6 failure patterns")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
