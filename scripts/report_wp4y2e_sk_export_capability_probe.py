import json
from pathlib import Path

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def main():
    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        first_mod_size=60,
        scaling_mod_size=50,
        num_large_digits=2,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    report = {
        "experiment": "wp4y2e_sk_export_capability_probe",
        "purpose": "Check whether Python runtime already exposes secret-key coefficient export needed for controlled SK/RLWE coefficient-row encryption.",
        "status": "capability_probe",
        "required_for_next": {
            "api": "export_secret_key_coeff_u64(coeff_count)",
            "zh": "Y2e controlled RLWE probe 需要导出 secret key 的 RNS coefficient prefix。",
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        candidates = [
            "export_secret_key_coeff_u64",
            "export_sk_coeff_u64",
            "export_secret_key_coeffs_u64",
            "export_private_key_coeff_u64",
            "inspect_secret_key_coeff_u64",
        ]

        attrs = {}
        for name in candidates:
            attrs[name] = hasattr(rt, name)

        ctx_attrs = {}
        ctx = rt.require_context()
        for name in candidates:
            ctx_attrs[name] = hasattr(ctx, name)

        report["python_runtime_attrs"] = attrs
        report["context_attrs"] = ctx_attrs

        calls = {}
        for name in candidates:
            if hasattr(rt, name):
                calls[name] = safe_call(name, lambda name=name: getattr(rt, name)(8))
            elif hasattr(ctx, name):
                calls[name] = safe_call("ctx." + name, lambda name=name: getattr(ctx, name)(8))

        report["calls"] = calls

    report["checks"] = {
        "any_sk_export_api_available（是否已有 secret-key coefficient export API）": any(report["python_runtime_attrs"].values()) or any(report["context_attrs"].values()),
        "any_sk_export_call_ok（是否已有 API 可成功调用）": any(v.get("ok") is True for v in report["calls"].values()),
    }

    report["summary"] = {
        "if_available": "Proceed to Y2f controlled RLWE coefficient-row encryption in Python.",
        "if_missing": "Patch bindings.cpp/runtime.py to expose export_secret_key_coeff_u64(coeff_count).",
        "next_step": "WP4-Y2f",
        "zh": "如果没有 secret-key export API，就先补 pybind；如果已有，就直接构造 controlled RLWE ciphertext。",
    }

    out = Path("/workspace/FIDESlib-GPT/reports/wp4y2e_sk_export_capability_probe_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2e secret-key export capability probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
