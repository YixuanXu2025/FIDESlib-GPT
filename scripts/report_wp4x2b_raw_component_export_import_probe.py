import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    cmt_algorithm2_rowwise,
)


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def compact_inspect(info):
    parts = info.get("parts", []) if info else []
    return {
        "valid": info.get("valid") if info else None,
        "openfhe_encoding_type": info.get("openfhe_encoding_type") if info else None,
        "num_parts": info.get("num_parts") if info else None,
        "parts_summary": [
            {
                "part_index": p.get("part_index"),
                "num_towers": p.get("num_towers"),
                "tower0_ring_dim": p.get("towers", [{}])[0].get("ring_dim") if p.get("towers") else None,
                "tower0_coeff_head": p.get("towers", [{}])[0].get("coeff_head") if p.get("towers") else None,
            }
            for p in parts
        ],
    }


def inspect_rt(rt, ct):
    return compact_inspect(rt.inspect_rlwe_components_cpu(ct, coeff_sample=8))


def normalize_export(x):
    # pybind dict/list objects are already JSON-like enough, but this keeps comparisons explicit.
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def row_towers_coeffs(export_obj, row_index):
    row = export_obj["rows"][row_index]
    return [tower["coeffs_u64"] for tower in row["towers"]]


def compare_row_prefix(export_a, row_a, export_b, row_b):
    a = row_towers_coeffs(export_a, row_a)
    b = row_towers_coeffs(export_b, row_b)

    per_tower = []
    ok_all = True

    for t, (ta, tb) in enumerate(zip(a, b)):
        ok = list(ta) == list(tb)
        ok_all = ok_all and ok
        per_tower.append({
            "tower_index": t,
            "equal": ok,
            "a": list(ta),
            "b": list(tb),
        })

    return {
        "equal_all_towers": ok_all,
        "per_tower": per_tower,
    }


def main():
    rng = np.random.default_rng(207700)
    U = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)
    V = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)

    coeff_count = 8

    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        scaling_mod_size=50,
        first_mod_size=60,
        num_large_digits=2,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    report = {
        "experiment": "wp4x2b_raw_component_export_import_probe",
        "purpose": (
            "Verify raw component export/import for DCRTPoly RNS coefficient matrices. "
            "This is the required IO layer before implementing paper PP-MM."
        ),
        "status": "raw_component_export_import_probe_only_no_ppmm",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_i64": (U @ V).tolist(),
        "coeff_count": coeff_count,
        "component_mapping": {
            "part0": "c0 / B",
            "part1": "c1 / A",
            "Algorithm3_sources": {
                "A_U": "C-MT(U).part1",
                "B_U": "C-MT(U).part0",
                "A_V": "V.rowwise.part1",
                "B_V": "V.rowwise.part0",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        U_cmt, U_cmt_trace = cmt_algorithm2_rowwise(rt, U_bundle)

        exports = {}

        exports["A_U"] = safe_call(
            "export_A_U",
            lambda: normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=1, coeff_count=coeff_count)),
        )
        exports["B_U"] = safe_call(
            "export_B_U",
            lambda: normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=0, coeff_count=coeff_count)),
        )
        exports["A_V"] = safe_call(
            "export_A_V",
            lambda: normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=1, coeff_count=coeff_count)),
        )
        exports["B_V"] = safe_call(
            "export_B_V",
            lambda: normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=0, coeff_count=coeff_count)),
        )

        report["exports"] = {
            name: {
                "ok": res["ok"],
                "error": res.get("error"),
                "summary": None if not res["ok"] else {
                    "num_rows": res["value"]["num_rows"],
                    "part": res["value"]["part"],
                    "num_towers": res["value"]["num_towers"],
                    "ring_dim": res["value"]["ring_dim"],
                    "coeff_count": res["value"]["coeff_count"],
                    "moduli_u64": res["value"]["moduli_u64"],
                    "row0_tower0_coeffs_u64": res["value"]["rows"][0]["towers"][0]["coeffs_u64"],
                },
            }
            for name, res in exports.items()
        }

        roundtrip = {
            "source": "A_U row 0",
            "steps": [
                "export C-MT(U).part1 row0",
                "import as one-component ciphertext-like handle",
                "export imported handle part0",
                "compare prefix coefficients tower-by-tower",
            ],
        }

        if exports["A_U"]["ok"]:
            towers_coeffs = row_towers_coeffs(exports["A_U"]["value"], 0)

            imported = safe_call(
                "import_A_U_row0",
                lambda: rt.import_1part_coeff_u64(U_cmt[0], towers_coeffs),
            )

            roundtrip["import_call"] = {
                "ok": imported["ok"],
                "error": imported.get("error"),
            }

            if imported["ok"]:
                roundtrip["imported_inspect"] = safe_call(
                    "inspect_imported",
                    lambda: inspect_rt(rt, imported["value"]),
                )

                reexport = safe_call(
                    "reexport_imported",
                    lambda: normalize_export(
                        rt.export_component_coeff_matrix_u64(
                            [imported["value"]],
                            part=0,
                            coeff_count=coeff_count,
                        )
                    ),
                )

                roundtrip["reexport_call"] = {
                    "ok": reexport["ok"],
                    "error": reexport.get("error"),
                }

                if reexport["ok"]:
                    roundtrip["compare"] = compare_row_prefix(
                        exports["A_U"]["value"],
                        0,
                        reexport["value"],
                        0,
                    )
                    roundtrip["reexport_summary"] = {
                        "num_rows": reexport["value"]["num_rows"],
                        "part": reexport["value"]["part"],
                        "num_towers": reexport["value"]["num_towers"],
                        "ring_dim": reexport["value"]["ring_dim"],
                        "coeff_count": reexport["value"]["coeff_count"],
                        "row0_tower0_coeffs_u64": reexport["value"]["rows"][0]["towers"][0]["coeffs_u64"],
                    }
        else:
            roundtrip["import_call"] = {
                "ok": False,
                "error": "A_U export failed",
            }

        report["roundtrip_A_U_row0"] = roundtrip

        all_exports_ok = all(res["ok"] for res in exports.values())

        imported_1part_ok = (
            roundtrip.get("imported_inspect", {}).get("ok") is True
            and roundtrip["imported_inspect"]["value"].get("num_parts") == 1
        )

        roundtrip_equal = (
            roundtrip.get("compare", {}).get("equal_all_towers") is True
        )

        report["checks"] = {
            "all_four_sources_export_ok（A_U/B_U/A_V/B_V 四个 component source 都能导出）": all_exports_ok,
            "exports_have_4_rows（导出的 component matrix 都有 4 行）": (
                all_exports_ok and all(res["value"]["num_rows"] == 4 for res in exports.values())
            ),
            "exports_have_4_towers（导出的每个 component 都有 4 个 RNS tower）": (
                all_exports_ok and all(res["value"]["num_towers"] == 4 for res in exports.values())
            ),
            "exports_coeff_count_8（每个 tower 导出前 8 个 coefficient）": (
                all_exports_ok and all(res["value"]["coeff_count"] == coeff_count for res in exports.values())
            ),
            "import_A_U_row0_ok（A_U row0 可以 import 成 1-component ciphertext-like handle）": (
                roundtrip.get("import_call", {}).get("ok") is True
            ),
            "imported_handle_is_1part（import 后 handle 只有 1 个 component）": imported_1part_ok,
            "reexport_imported_ok（import 后可以再次 export）": (
                roundtrip.get("reexport_call", {}).get("ok") is True
            ),
            "roundtrip_prefix_equal_all_towers（导出-import-再导出的 prefix coefficient 每个 tower 完全一致）": roundtrip_equal,
            "ready_for_X3_raw_ppmm（已具备 raw component PP-MM 的导出/导入基础）": (
                all_exports_ok and imported_1part_ok and roundtrip_equal
            ),
            "U_cmt_auto_alphas_expected（C-MT(U) Auto alpha 是 [1,3,5,7]）": [
                x["alpha"] for x in U_cmt_trace["auto"]
            ] == [1, 3, 5, 7],
        }

        report["summary"] = {
            "next_step": "WP4-X3",
            "next_step_goal": (
                "Implement CPU toy-size raw RNS PP-MM over exported component matrices: "
                "M00=A_U@A_V, M01=A_U@B_V, M10=B_U@A_V, M11=B_U@B_V, mod each tower modulus."
            ),
            "zh": "如果 X2b 通过，下一步开始真正做 raw coefficient/RNS matrix multiplication，而不是 DCRTPoly polynomial multiplication。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x2b_raw_component_export_import_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X2b raw component export/import probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
