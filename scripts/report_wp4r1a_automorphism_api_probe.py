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
        "output": p.stdout[:50000],
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
        "EvalAutomorphism",
        "EvalAutomorphismKeyGen",
        "EvalAutomorphismKey",
        "AutomorphismTransform",
        "Automorphism",
        "FindAutomorphismIndex",
        "PrecomputeAutoMap",
        "KeySwitch",
        "KeySwitchGen",
        "EvalAtIndex",
        "EvalRotate",
        "EvalFastRotation",
        "EvalFastRotationPrecompute",
        "EvalSumKeyGen",
        "GetEvalAutomorphismKeyMap",
        "GetEvalAutomorphismKeyMapPtr",
        "InsertEvalAutomorphismKey",
    ]

    report = {
        "experiment": "wp4r1a_automorphism_api_probe",
        "purpose": (
            "Probe OpenFHE/FIDESlib APIs needed for Park-2025 Algorithm 2 C-MT: "
            "Auto(ct; alpha), automorphism key generation, and key-switch support."
        ),
        "status": "api_probe_only_no_algorithm_change",
        "current_state": {
            "completed": [
                "WP4-L/M/N toy Algorithm 2/3 pipeline",
                "WP4-O/P native coefficient ciphertext bridge",
                "WP4-Q1/Q2 real row-wise coefficient ciphertext matrix bundle",
            ],
            "this_step": "discover exact native automorphism/key-switch APIs",
        },
        "roots": roots,
        "grep": {},
        "focused_files": {},
        "checks": {
            "probe_only（本脚本只查 API，不修改代码）": True,
            "pcmm_preserved（PCMM backend component_linear_transform_gpu 保留不动）": True,
        },
    }

    for pat in patterns:
        report["grep"][pat] = run(
            f"grep -RIn --include='*.h' --include='*.hpp' --include='*.cpp' "
            f"--include='*.cu' '{pat}' {root_str} | head -160"
        )

    focused = [
        "/root/fideslib/deps/openfhe-install/include/openfhe/pke/cryptocontext.h",
        "/root/fideslib/deps/openfhe-src/src/pke/include/cryptocontext.h",
        "/root/fideslib/deps/openfhe-src/src/pke/include/schemebase/base-scheme.h",
        "/root/fideslib/deps/openfhe-src/src/pke/include/schemebase/base-leveledshe.h",
        "/root/fideslib/deps/openfhe-src/src/pke/include/schemebase/base-keyswitch.h",
        "/root/fideslib/api/CryptoContext.hpp",
        "/root/fideslib/src/CKKS/Context.cpp",
        "/workspace/FIDESlib-GPT/bindings/bindings.cpp",
    ]

    for f in focused:
        if Path(f).exists():
            report["focused_files"][f] = {
                "automorphism_lines": run(
                    f"grep -n -B 8 -A 24 "
                    f"'EvalAutomorphism\\|Automorphism\\|EvalAtIndex\\|EvalRotate\\|KeySwitch' {f} | head -240"
                )
            }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4r1a_automorphism_api_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("WP4-R1a automorphism/key-switch API probe")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
