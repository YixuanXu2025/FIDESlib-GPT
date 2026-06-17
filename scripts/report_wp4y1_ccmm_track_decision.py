import json
from pathlib import Path


REPORT_DIR = Path("/workspace/FIDESlib-GPT/reports")


def load_report(filename):
    p = REPORT_DIR / filename
    if not p.exists():
        return {
            "exists": False,
            "path": str(p),
            "error": "missing",
        }

    try:
        return {
            "exists": True,
            "path": str(p),
            "data": json.loads(p.read_text(encoding="utf-8")),
        }
    except Exception as e:
        return {
            "exists": True,
            "path": str(p),
            "error": repr(e),
        }


def get_check(report, key_contains):
    if not report.get("exists") or "data" not in report:
        return None

    checks = report["data"].get("checks", {})
    for k, v in checks.items():
        if key_contains in k:
            return v

    return None


def compact_checks(report):
    if not report.get("exists") or "data" not in report:
        return {}

    return report["data"].get("checks", {})


def main():
    sources = {
        "X7d_coeff_row_roundtrip": load_report("wp4x7d_coeff_row_roundtrip_baseline_report.json"),
        "X7e_transparent_coeff_baseline": load_report("wp4x7e_transparent_coeff_row_baseline_report.json"),
        "X7f_logical_n_cmt_oracle": load_report("wp4x7f_logical_n_cmt_oracle_report.json"),
        "X7k_raw_rns_add_line6": load_report("wp4x7k_raw_rns_add_line6_probe_report.json"),
        "X7l_oracle_helper_smoke": load_report("wp4x7l_oracle_helper_smoke_report.json"),
    }

    report = {
        "experiment": "wp4y1_ccmm_track_decision",
        "purpose": (
            "Freeze the current CCMM debugging conclusions and choose the next implementation track. "
            "This separates PCMM, debug/oracle CCMM algebra, and real secure CCMM implementation work."
        ),
        "status": "track_decision_report_no_code_change",
        "source_reports": {
            name: {
                "exists": src.get("exists"),
                "path": src.get("path"),
                "error": src.get("error"),
            }
            for name, src in sources.items()
        },
        "milestones": {
            "PCMM_preserved": {
                "value": True,
                "zh": "原本 PCMM / rt.ccmm_gpu_fused_raw 改名后的路线保留，不作为真实 CCMM。",
            },
            "debug_oracle_algebra_validated": {
                "value": get_check(sources["X7l_oracle_helper_smoke"], "all_cases_equal_U_T_matmul_V") is True,
                "evidence": "X7l all_cases_equal_U_T_matmul_V",
                "zh": "logical-n C-MT oracle + raw RNS PP-MM + raw RNS line6 add 的 debug/oracle 代数链条已稳定。",
            },
            "transparent_coeff_baseline_valid": {
                "value": get_check(sources["X7e_transparent_coeff_baseline"], "transparent_input_rows_decrypt_exact") is True,
                "evidence": "X7e transparent_input_rows_decrypt_exact",
                "zh": "transparent c0=row,c1=0 可以正确表达 coefficient-row 明文语义。",
            },
            "real_coeff_row_encryption_broken": {
                "value": get_check(sources["X7d_coeff_row_roundtrip"], "coeff_after_compress_roundtrip_all_exact") is False,
                "evidence": "X7d coeff_after_compress_roundtrip_all_exact",
                "zh": "真实 encrypt_coeff_row_i64/decrypt_coeff_row_i64 roundtrip 仍不正确。",
            },
            "real_cmt_dimension_contract_broken_for_n4": {
                "value": (
                    get_check(sources["X7e_transparent_coeff_baseline"], "transparent_input_rows_decrypt_exact") is True
                    and get_check(sources["X7e_transparent_coeff_baseline"], "transparent_cmt_outputs_equal_transpose") is False
                    and get_check(sources["X7f_logical_n_cmt_oracle"], "logical_n_cmt_oracle_outputs_equal_transpose") is True
                ),
                "evidence": "X7e real C-MT fails but X7f logical-n oracle passes",
                "zh": "当前真实 C-MT 在 n=4, ring_dim=16384 下不满足 logical-n transpose 语义。",
            },
            "line6_add_requires_raw_rns_add_in_debug_path": {
                "value": get_check(sources["X7k_raw_rns_add_line6"], "raw_full_paper_order_equals_U_T_matmul_V") is True,
                "evidence": "X7j showed add_coeff_ct order dependence; X7k raw RNS add passed",
                "zh": "debug/oracle 路径中 line6 应使用 raw RNS component add，而不是 add_coeff_ct。",
            },
        },
        "tracks": {
            "track_A_debug_oracle": {
                "name": "Debug/oracle CCMM algebra track",
                "status": "available",
                "description": (
                    "Use transparent coefficient rows, logical-n C-MT oracle, raw RNS PP-MM, and raw RNS add "
                    "to validate Algorithm 3 algebra and produce deterministic diagnostics."
                ),
                "security": "not secure / not homomorphic",
                "recommended_use": [
                    "debug Park-2025 Algorithm 3 algebra",
                    "unit-test PP-MM / component formulas",
                    "compare implementation traces against plaintext expectations",
                ],
                "zh": "可用于代数验证，但不能作为最终安全 CCMM。",
            },
            "track_B_real_small_n_ccmm": {
                "name": "Real secure small-n CCMM track",
                "status": "blocked",
                "blockers": [
                    "real coefficient-row encryption semantics are not correct",
                    "current homomorphic C-MT is ring-dimension C-MT, not logical-n C-MT for n=4",
                    "line6 add path must be revisited under real ciphertext semantics",
                ],
                "required_next_steps": [
                    "fix encrypt_coeff_row_i64/decrypt_coeff_row_i64 roundtrip",
                    "design logical-n homomorphic C-MT for n << ring_dim, or block/pad the matrix representation",
                    "then replace debug raw add with secure homomorphic equivalent",
                ],
                "zh": "这是最终想要的真实小矩阵 CCMM，但当前被 coefficient encryption 和 logical-n C-MT 卡住。",
            },
            "track_C_dimension_compatible_benchmark": {
                "name": "Dimension-compatible Park C-MT benchmark track",
                "status": "theoretically aligned but impractical for current toy",
                "description": (
                    "Use matrix dimension equal to the ring dimension so that Park C-MT contract matches directly."
                ),
                "issue": "ring_dim=16384 would imply very large row count and matrix size for a direct test",
                "zh": "理论上最贴论文，但当前直接做 16384×16384 不适合作为短期实验。",
            },
            "track_D_PCMM": {
                "name": "PCMM / plaintext-matrix comparison track",
                "status": "preserved",
                "description": (
                    "Keep the previous GPU fused primitive as PCMM / ciphertext-by-plaintext-matrix comparison path."
                ),
                "zh": "保留原 PCMM，用于后续和 JKSL/BSGS 的速率对比。",
            },
        },
        "recommendation": {
            "short_term": {
                "track": "track_A_debug_oracle + track_D_PCMM",
                "reason": (
                    "Track A gives a stable algebra oracle for CCMM debugging; Track D preserves the PCMM performance path."
                ),
                "zh": "短期保留 PCMM，同时用 oracle CCMM 固定算法语义。",
            },
            "real_ccmm_next": {
                "track": "track_B_real_small_n_ccmm",
                "first_blocker_to_attack": "coefficient-row encryption roundtrip",
                "second_blocker_to_attack": "logical-n homomorphic C-MT",
                "zh": "真实 CCMM 下一步优先修 coefficient-row 加解密，然后再设计 logical-n C-MT。",
            },
        },
        "next_step": {
            "name": "WP4-Y2",
            "goal": (
                "Fix real coefficient-row encryption/decryption semantics. "
                "Target: encrypt_coeff_row_i64(row) -> compress -> decrypt_coeff_row_i64 returns row exactly."
            ),
            "zh": "下一步建议先修真实 coefficient-row 加解密，而不是继续调 Algorithm 3。",
        },
        "raw_source_checks": {
            name: compact_checks(src)
            for name, src in sources.items()
        },
    }

    out_dir = REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y1_ccmm_track_decision_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y1 CCMM track decision report")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
