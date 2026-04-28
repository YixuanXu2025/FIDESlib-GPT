import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def main():
    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        scaling_mod_size=50,
        first_mod_size=60,
        num_large_digits=2,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=False,
        ciphertext_autoload=False,
        with_mult_key=True,
    )

    x = np.array([0.125, -0.25, 0.5, -1.0, 1.5, -2.0, 3.0, -4.0], dtype=np.float64)

    with HERuntime(cfg, rotation_steps=[]) as rt:
        ct = rt.encrypt(x.tolist())

        dec0 = np.array(rt.decrypt(ct, logical_length=len(x)), dtype=np.float64)
        info0 = rt.inspect_rlwe_components_cpu(ct, coeff_sample=4)

        ct2 = rt.roundtrip_rlwe_components_cpu(ct)

        dec1 = np.array(rt.decrypt(ct2, logical_length=len(x)), dtype=np.float64)
        info1 = rt.inspect_rlwe_components_cpu(ct2, coeff_sample=4)

    err_original = float(np.max(np.abs(dec0 - x)))
    err_roundtrip_to_input = float(np.max(np.abs(dec1 - x)))
    err_roundtrip_to_original_dec = float(np.max(np.abs(dec1 - dec0)))

    report = {
        "experiment": "wp4b_component_roundtrip_cpu",
        "input": x.tolist(),
        "decrypt_original": dec0.tolist(),
        "decrypt_roundtrip": dec1.tolist(),
        "err_original": err_original,
        "err_roundtrip_to_input": err_roundtrip_to_input,
        "err_roundtrip_to_original_dec": err_roundtrip_to_original_dec,
        "component_info_original": info0,
        "component_info_roundtrip": info1,
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4b_component_roundtrip_cpu_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("WP4-B component CPU roundtrip result")
    print("=" * 100)
    print(f"err_original={err_original:.6e}")
    print(f"err_roundtrip_to_input={err_roundtrip_to_input:.6e}")
    print(f"err_roundtrip_to_original_dec={err_roundtrip_to_original_dec:.6e}")

    print("-" * 100)
    print("original:")
    print(f"  loaded={info0.get('fides_loaded')} gpu={info0.get('fides_gpu_handle')} parts={info0.get('num_parts')}")
    print(f"  level={info0.get('openfhe_level')} noise_deg={info0.get('openfhe_noise_scale_deg')} slots={info0.get('openfhe_slots')}")

    print("roundtrip:")
    print(f"  loaded={info1.get('fides_loaded')} gpu={info1.get('fides_gpu_handle')} parts={info1.get('num_parts')}")
    print(f"  level={info1.get('openfhe_level')} noise_deg={info1.get('openfhe_noise_scale_deg')} slots={info1.get('openfhe_slots')}")

    for part_idx in range(min(len(info0.get("parts", [])), len(info1.get("parts", [])))):
        p0 = info0["parts"][part_idx]
        p1 = info1["parts"][part_idx]
        h0 = [tw.get("fnv1a64") for tw in p0.get("towers", [])]
        h1 = [tw.get("fnv1a64") for tw in p1.get("towers", [])]
        print(f"part {part_idx} hash_equal={h0 == h1}")

    print("=" * 100)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
