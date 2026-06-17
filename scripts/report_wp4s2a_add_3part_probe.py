import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def compact_inspect(info):
    parts = info.get("parts", []) if info else []
    return {
        "valid": info.get("valid") if info else None,
        "openfhe_level": info.get("openfhe_level") if info else None,
        "openfhe_noise_scale_deg": info.get("openfhe_noise_scale_deg") if info else None,
        "openfhe_scaling_factor": info.get("openfhe_scaling_factor") if info else None,
        "openfhe_slots": info.get("openfhe_slots") if info else None,
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


def shape_ok_3part(val):
    parts = val.get("parts_summary", [])
    return (
        bool(val.get("valid"))
        and val.get("openfhe_encoding_type") == 1
        and val.get("num_parts") == 3
        and all(p.get("num_towers") == 4 for p in parts)
        and all(p.get("tower0_ring_dim") == 16384 for p in parts)
    )


def main():
    rows = [
        np.array([1, -2, 3, -1], dtype=np.int64),
        np.array([2, 1, -1, 0], dtype=np.int64),
        np.array([-1, 0, 2, 3], dtype=np.int64),
        np.array([3, -3, 1, 2], dtype=np.int64),
    ]

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
        "experiment": "wp4s2a_add_3part_probe",
        "purpose": (
            "Verify that no-relin 3-component ciphertexts can be added: "
            "(c0,c1,c2) + (d0,d1,d2) -> 3-component ciphertext."
        ),
        "status": "primitive_probe_only_no_relin_no_rescale_no_full_ccmm",
        "input_rows_i64": [r.tolist() for r in rows],
        "formula": {
            "value": "sum = mul_no_relin(ct0, ct1) + mul_no_relin(ct2, ct3)",
            "zh": "先生成两个三分量乘法结果，再验证三分量密文加法是否保持结构",
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        cts = [rt.encrypt_coeff_row_i64(r.tolist()) for r in rows]

        prod01 = rt.mul_coeff_ct_no_relin(cts[0], cts[1])
        prod23 = rt.mul_coeff_ct_no_relin(cts[2], cts[3])

        report["prod01"] = safe_call("inspect_prod01", lambda: inspect_rt(rt, prod01))
        report["prod23"] = safe_call("inspect_prod23", lambda: inspect_rt(rt, prod23))

        add = safe_call("add_3part", lambda: rt.add_coeff_ct(prod01, prod23))
        report["add_call"] = {"ok": add["ok"], "error": add.get("error")}

        if add["ok"]:
            report["add_output"] = safe_call("inspect_add_output", lambda: inspect_rt(rt, add["value"]))

        prod01_ok = report["prod01"]["ok"] and shape_ok_3part(report["prod01"]["value"])
        prod23_ok = report["prod23"]["ok"] and shape_ok_3part(report["prod23"]["value"])

        add_output_ok = False
        if report.get("add_output", {}).get("ok"):
            add_output_ok = shape_ok_3part(report["add_output"]["value"])

        report["checks"] = {
            "prod01_is_3part（第一个乘法结果是 c0/c1/c2 三分量密文）": prod01_ok,
            "prod23_is_3part（第二个乘法结果是 c0/c1/c2 三分量密文）": prod23_ok,
            "add_3part_call_ok（三分量密文加法调用成功）": report["add_call"]["ok"],
            "add_3part_output_valid（三分量加法输出密文有效）": add_output_ok,
            "add_3part_preserves_three_rlwe_parts（三分量加法输出仍有 c0/c1/c2）": (
                report.get("add_output", {}).get("ok")
                and report["add_output"]["value"].get("num_parts") == 3
            ),
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4s2a_add_3part_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-S2a 3-component ciphertext add probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
