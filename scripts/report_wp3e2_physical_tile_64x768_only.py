from pathlib import Path
import json

from hegpt import HEConfig, HERuntime
from report_wp3e2_matrix_role_tile_64x768_he_pcmm import (
    make_matrices,
    run_matrix_physical_tile_case,
)


def main():
    X, W = make_matrices(seed=206768)

    cfg = HEConfig(
        ring_dim=16384,
        multiplicative_depth=2,
        scaling_mod_size=50,
        batch_size=8192,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    with HERuntime(cfg, rotation_steps=()) as rt:
        print("runtime info（运行时信息）:", rt.info())
        report = run_matrix_physical_tile_case(rt, X, W)

    out_path = Path("/workspace/FIDESlib-GPT/reports/wp3e2_physical_tile_64x768_only_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
