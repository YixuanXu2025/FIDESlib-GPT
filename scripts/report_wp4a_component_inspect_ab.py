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
        num_large_digits=3,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    x = np.array([0.125, -0.25, 0.5, -1.0, 1.5, -2.0, 3.0, -4.0], dtype=np.float64)

    with HERuntime(cfg, rotation_steps=[]) as rt:
        ct = rt.encrypt(x.tolist())
        dec = np.array(rt.decrypt(ct, logical_length=len(x)), dtype=np.float64)
        err = float(np.max(np.abs(dec - x)))

        info = rt.inspect_rlwe_components(ct, coeff_sample=8)

    report = {
        "experiment": "wp4a_component_inspect_ab",
        "input": x.tolist(),
        "decrypt": dec.tolist(),
        "max_abs_err": err,
        "component_info": info,
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4a_component_inspect_ab_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("WP4-A component inspect result")
    print("=" * 100)
    print(f"max_abs_err={err:.6e}")
    print(f"num_parts={info.get('num_parts')}")
    print(f"level={info.get('level')}")
    print(f"noise_scale_deg={info.get('noise_scale_deg')}")
    print(f"scaling_factor={info.get('scaling_factor')}")
    print(f"slots={info.get('slots')}")
    print(f"encoding_type={info.get('encoding_type')}")

    for part in info.get("parts", []):
        print("-" * 100)
        print(f"part_index={part.get('part_index')} label={part.get('label')}")
        print(f"coefficient_format_ok={part.get('coefficient_format_ok')}")
        print(f"num_towers={part.get('num_towers')}")
        towers = part.get("towers", [])
        for tw in towers[:3]:
            print(
                f"  tower={tw.get('tower_index')} "
                f"modulus={tw.get('modulus_u64')} "
                f"ring_dim={tw.get('ring_dim')} "
                f"hash={tw.get('fnv1a64')} "
                f"head={tw.get('coeff_head')}"
            )

    print("=" * 100)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
