import json
import subprocess
from pathlib import Path


def run(cmd):
    p = subprocess.run(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "output": p.stdout[:60000],
    }


def main():
    roots = [
        "/root/fideslib/deps/openfhe-install/include/openfhe",
        "/root/fideslib/deps/openfhe-src/src",
        "/root/fideslib/api",
        "/root/fideslib/src",
        "/workspace/FIDESlib-GPT/bindings",
        "/workspace/FIDESlib-GPT/python",
    ]
    roots = [r for r in roots if Path(r).exists()]
    root_str = " ".join(roots)

    patterns = [
        "GetAllElements",
        "GetElementAtIndex",
        "SetElementAtIndex",
        "GetValues",
        "SetValues",
        "SetValuesToZero",
        "NativeVector",
        "NativeInteger",
        "ConvertToInt",
        "GetModulus",
        "GetParams",
        "DCRTPolyImpl",
        "PolyImpl",
        "NativePoly",
        "SetFormat",
        "COEFFICIENT",
        "EVALUATION",
        "inspect_rlwe_components_cpu",
        "monomial_mul_coeff_ct",
        "component_mul_part_coeff_ct",
    ]

    focused_files = [
        "/root/fideslib/deps/openfhe-install/include/openfhe/core/lattice/hal/default/dcrtpoly.h",
        "/root/fideslib/deps/openfhe-src/src/core/include/lattice/hal/default/dcrtpoly.h",
        "/root/fideslib/deps/openfhe-install/include/openfhe/core/lattice/hal/default/poly.h",
        "/root/fideslib/deps/openfhe-src/src/core/include/lattice/hal/default/poly.h",
        "/root/fideslib/deps/openfhe-install/include/openfhe/core/math/hal/intnat/ubintnat.h",
        "/root/fideslib/deps/openfhe-src/src/core/include/math/hal/intnat/ubintnat.h",
        "/workspace/FIDESlib-GPT/bindings/bindings.cpp",
    ]

    report = {
        "experiment": "wp4x2a_raw_component_api_probe",
        "purpose": (
            "Discover exact OpenFHE/FIDESlib APIs for exporting/importing raw DCRTPoly "
            "RNS coefficient matrices. This is required before implementing Park-2025 PP-MM."
        ),
        "status": "api_probe_only_no_code_change",
        "current_state": {
            "completed": [
                "WP4-X1 corrected Algorithm 3 contract: C-MT(U), not C-MT(V)",
                "W3c proved component-formula pairwise skeleton still gives dense wrong one-hot outputs",
            ],
            "this_step": "find exact raw coefficient read/write methods for DCRTPoly/NativePoly",
        },
        "roots": roots,
        "grep": {},
        "focused_files": {},
        "checks": {
            "probe_only（本脚本只查 API，不修改代码）": True,
            "pcmm_preserved（PCMM backend 保留不动）": True,
            "whole_ct_skeleton_rejected（不再推进整密文乘法 skeleton）": True,
        },
    }

    for pat in patterns:
        report["grep"][pat] = run(
            f"grep -RIn --include='*.h' --include='*.hpp' --include='*.cpp' "
            f"--include='*.cu' '{pat}' {root_str} | head -220"
        )

    for f in focused_files:
        if Path(f).exists():
            report["focused_files"][f] = {
                "head": run(f"sed -n '1,260p' {f}"),
                "relevant_lines": run(
                    f"grep -n -B 10 -A 36 "
                    f"'GetAllElements\\|GetElementAtIndex\\|SetElementAtIndex\\|GetValues\\|SetValues\\|ConvertToInt\\|GetModulus\\|NativeVector\\|DCRTPolyImpl\\|PolyImpl\\|SetFormat' "
                    f"{f} | head -360"
                ),
            }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x2a_raw_component_api_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("WP4-X2a raw component export/import API probe")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
