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
        "output": p.stdout[:30000],
    }


def main():
    roots = [
        "/root/fideslib/include",
        "/root/fideslib/src",
        "/root/fideslib",
        "/workspace/FIDESlib-GPT/bindings",
    ]
    roots = [r for r in roots if Path(r).exists()]
    root_str = " ".join(roots)

    patterns = [
        "class CryptoContextImpl",
        "MakeCKKSPackedPlaintext",
        "MakeCoefPackedPlaintext",
        "CryptoContext<",
        "lbcrypto::CryptoContext",
        "GetCryptoContext",
        "GetContext",
        "Encrypt(",
        "Decrypt(",
        "PlaintextImpl",
        "GetCKKSPackedValue",
        "GetElements",
        "SetElements",
    ]

    report = {
        "experiment": "wp4p2_fideslib_context_api_probe",
        "purpose": (
            "Determine whether FIDESlib's CryptoContextImpl exposes or stores "
            "an underlying OpenFHE CryptoContext suitable for coefficient-packed rows."
        ),
        "status": "probe_only",
        "roots": roots,
        "grep": {},
    }

    for pat in patterns:
        cmd = (
            f"grep -RIn --include='*.h' --include='*.hpp' --include='*.cpp' "
            f"--include='*.cu' '{pat}' {root_str} | head -120"
        )
        report["grep"][pat] = run(cmd)

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4p2_fideslib_context_api_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("WP4-P2 FIDESlib context API probe")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
