import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_decrypt(rt, ct, slots):
    try:
        y = np.array(rt.decrypt(ct, logical_length=slots), dtype=np.float64)
        return y, None
    except Exception as e:
        return None, repr(e)


def main():
    slots = 8
    seed = 205403
    rng = np.random.default_rng(seed)

    x = rng.normal(0.0, 0.25, size=(slots,)).astype(np.float64)

    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        scaling_mod_size=50,
        first_mod_size=60,
        num_large_digits=2,
        batch_size=slots,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    report = {
        "experiment": "wp4e3_gpu_copyback_cpu_debug",
        "slots": slots,
        "seed": seed,
        "x": x.tolist(),
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        ct = rt.encrypt(x.tolist())
        state_in = rt.ciphertext_storage_state(ct)

        y0, e0 = safe_decrypt(rt, ct, slots)

        results = []
        for rev in [0, 1]:
            ct_copy = rt.gpu_copyback_cpu_debug(ct, rev=rev)
            state = rt.ciphertext_storage_state(ct_copy)
            y, e = safe_decrypt(rt, ct_copy, slots)

            item = {
                "rev": rev,
                "state": state,
                "decrypt_error": e,
                "decrypt": None if y is None else y.tolist(),
            }

            if y is not None:
                item["err_to_input"] = float(np.max(np.abs(y - x)))
                if y0 is not None:
                    item["err_to_original_decrypt"] = float(np.max(np.abs(y - y0)))

            results.append(item)

    report["input_state"] = state_in
    report["original_decrypt_error"] = e0
    report["original_decrypt"] = None if y0 is None else y0.tolist()
    if y0 is not None:
        report["original_err_to_input"] = float(np.max(np.abs(y0 - x)))
    report["copyback_results"] = results

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4e3_gpu_copyback_cpu_debug_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("WP4-E3 GPU -> CPU copyback debug")
    print("=" * 100)
    print("input state:")
    print(state_in)
    print(f"original_decrypt_error={e0}")
    if "original_err_to_input" in report:
        print(f"original_err_to_input={report['original_err_to_input']:.6e}")

    for item in results:
        print("-" * 100)
        print(f"rev={item['rev']}")
        print(item["state"])
        print(f"decrypt_error={item['decrypt_error']}")
        if "err_to_input" in item:
            print(f"err_to_input={item['err_to_input']:.6e}")
        if "err_to_original_decrypt" in item:
            print(f"err_to_original_decrypt={item['err_to_original_decrypt']:.6e}")

    print("=" * 100)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
