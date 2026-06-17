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
        "num_towers_part0": parts[0].get("num_towers") if parts else None,
        "ring_dim_part0_tower0": parts[0].get("towers", [{}])[0].get("ring_dim") if parts else None,
        "tower0_coeff_head_part0": parts[0].get("towers", [{}])[0].get("coeff_head") if parts else None,
    }


def inspect_rt(rt, ct):
    return compact_inspect(rt.inspect_rlwe_components_cpu(ct, coeff_sample=8))


def main():
    row_a = np.array([1, -2, 3, -1], dtype=np.int64)
    row_b = np.array([2, 1, -1, 0], dtype=np.int64)

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

    exps = [0, 1, 2, 3, 4, 16383, 16384, 16385]

    report = {
        "experiment": "wp4r2b_add_monomial_probe",
        "purpose": "Smoke test coefficient ciphertext add and component-wise monomial multiply X^exp · ct.",
        "status": "primitive_probe_only_no_full_tweak",
        "input_row_a_i64": row_a.tolist(),
        "input_row_b_i64": row_b.tolist(),
        "exps": exps,
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        ct_a = rt.encrypt_coeff_row_i64(row_a.tolist())
        ct_b = rt.encrypt_coeff_row_i64(row_b.tolist())

        report["input_a"] = safe_call("inspect_a", lambda: inspect_rt(rt, ct_a))
        report["input_b"] = safe_call("inspect_b", lambda: inspect_rt(rt, ct_b))

        add_res = safe_call("add_coeff_ct", lambda: rt.add_coeff_ct(ct_a, ct_b))
        report["add_call"] = {"ok": add_res["ok"], "error": add_res.get("error")}
        if add_res["ok"]:
            report["add_output"] = safe_call("inspect_add", lambda: inspect_rt(rt, add_res["value"]))

        mono_cases = []
        for exp in exps:
            case = {"exp": exp}
            mono = safe_call(
                f"monomial_mul_exp_{exp}",
                lambda exp=exp: rt.monomial_mul_coeff_ct(ct_a, exp),
            )
            case["call"] = {"ok": mono["ok"], "error": mono.get("error")}
            if mono["ok"]:
                case["output"] = safe_call(
                    f"inspect_monomial_exp_{exp}",
                    lambda mono=mono: inspect_rt(rt, mono["value"]),
                )
                if case["output"]["ok"]:
                    val = case["output"]["value"]
                    case["checks"] = {
                        "output_valid（输出密文有效）": bool(val.get("valid")),
                        "output_coef_encoding_type_1（输出仍是 coefficient encoding 类型 1）": val.get("openfhe_encoding_type") == 1,
                        "output_has_two_rlwe_parts（输出仍有 c0/c1 两个 RLWE component）": val.get("num_parts") == 2,
                        "output_ring_dim_16384（输出 ring_dim 仍是 16384）": val.get("ring_dim_part0_tower0") == 16384,
                    }
            mono_cases.append(case)

        report["monomial_cases"] = mono_cases

        add_ok = False
        if report.get("add_output", {}).get("ok"):
            val = report["add_output"]["value"]
            add_ok = (
                bool(val.get("valid"))
                and val.get("openfhe_encoding_type") == 1
                and val.get("num_parts") == 2
                and val.get("ring_dim_part0_tower0") == 16384
            )

        report["summary"] = {
            "add_call_ok（coefficient ciphertext 加法调用成功）": report["add_call"]["ok"],
            "add_output_shape_ok（加法输出仍保持 coefficient/c0c1/ring_dim 结构）": add_ok,
            "all_monomial_calls_ok（所有 X^exp·ct 调用成功）": all(c["call"]["ok"] for c in mono_cases),
            "successful_monomial_exps（成功的 exp 列表）": [c["exp"] for c in mono_cases if c["call"]["ok"]],
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4r2b_add_monomial_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-R2b coefficient ciphertext add + monomial multiply probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
