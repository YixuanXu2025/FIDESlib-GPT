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
    ]
    roots = [r for r in roots if Path(r).exists()]
    root_str = " ".join(roots)

    patterns = [
        "Clone",
        "CloneEmpty",
        "CloneZero",
        "CiphertextImpl(",
        "SetElements",
        "GetElements()",
        "GetElements",
        "EvalAdd",
        "EvalSub",
        "EvalMult",
        "SetElementAtIndex",
        "GetAllElements",
        "SetValues",
        "GetValues",
        "operator\\[\\]",
        "DCRTPolyImpl",
        "NativePoly",
    ]

    report = {
        "experiment": "wp4r2a_ciphertext_component_mutation_probe",
        "purpose": (
            "Inspect APIs needed for Algorithm 1 Tweak: ciphertext clone, EvalAdd, "
            "mutable ciphertext elements, and DCRTPoly coefficient/tower mutation."
        ),
        "status": "api_probe_only_no_algorithm_change",
        "current_state": {
            "completed": [
                "WP4-R1b native EvalAutomorphism coefficient ciphertext probe passed",
            ],
            "this_step": "discover exact APIs for X^k · ct and ct addition",
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
            f"--include='*.cu' '{pat}' {root_str} | head -180"
        )

    focused = [
        "/root/fideslib/deps/openfhe-install/include/openfhe/pke/ciphertext.h",
        "/root/fideslib/deps/openfhe-src/src/pke/include/ciphertext.h",
        "/root/fideslib/deps/openfhe-install/include/openfhe/core/lattice/hal/default/dcrtpoly.h",
        "/root/fideslib/deps/openfhe-src/src/core/include/lattice/hal/default/dcrtpoly.h",
        "/root/fideslib/deps/openfhe-install/include/openfhe/core/lattice/hal/default/poly.h",
        "/root/fideslib/deps/openfhe-src/src/core/include/lattice/hal/default/poly.h",
        "/root/fideslib/deps/openfhe-install/include/openfhe/pke/cryptocontext.h",
        "/root/fideslib/deps/openfhe-src/src/pke/include/cryptocontext.h",
        "/root/fideslib/api/Ciphertext.hpp",
        "/root/fideslib/api/CryptoContext.hpp",
        "/root/fideslib/src/CKKS/Context.cpp",
        "/workspace/FIDESlib-GPT/bindings/bindings.cpp",
    ]

    for f in focused:
        if Path(f).exists():
            report["focused_files"][f] = {
                "dump_head": run(f"sed -n '1,260p' {f}"),
                "mutation_lines": run(
                    f"grep -n -B 8 -A 32 "
                    f"'Clone\\|CloneEmpty\\|SetElements\\|GetElements\\|EvalAdd\\|SetElementAtIndex\\|GetAllElements\\|SetValues\\|GetValues' {f} | head -260"
                ),
            }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4r2a_ciphertext_component_mutation_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("WP4-R2a ciphertext/component mutation API probe")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
