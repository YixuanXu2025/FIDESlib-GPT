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
        "output": p.stdout[:40000],
    }


def main():
    files = [
        "/root/fideslib/api/PublicKey.hpp",
        "/root/fideslib/api/PublicKey.cpp",
        "/root/fideslib/api/PrivateKey.hpp",
        "/root/fideslib/api/PrivateKey.cpp",
        "/root/fideslib/api/Ciphertext.hpp",
        "/root/fideslib/api/Ciphertext.cpp",
        "/root/fideslib/api/CryptoContext.hpp",
        "/root/fideslib/api/CryptoContext.cpp",
        "/root/fideslib/api/Definitions.hpp",
    ]

    existing = [f for f in files if Path(f).exists()]
    file_str = " ".join(existing)

    patterns = [
        "class PublicKeyImpl",
        "class PrivateKeyImpl",
        "class CiphertextImpl",
        "PublicKeyImpl::",
        "PrivateKeyImpl::",
        "CiphertextImpl::",
        "PublicKey<",
        "PrivateKey<",
        "Ciphertext<",
        "std::any",
        "cpu",
        "GetCryptoContext",
        "Encrypt(",
        "Decrypt(",
        "CiphertextImpl(",
    ]

    report = {
        "experiment": "wp4p3b_key_ct_wrapper_probe",
        "purpose": "Inspect FIDESlib key/ciphertext wrapper internals for native OpenFHE coefficient-row bridge.",
        "status": "probe_only",
        "existing_files": existing,
        "grep": {},
        "file_dumps": {},
    }

    for pat in patterns:
        report["grep"][pat] = run(
            f"grep -RIn '{pat}' {file_str} | head -160"
        )

    for f in existing:
        report["file_dumps"][f] = run(f"sed -n '1,260p' {f}")

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4p3b_key_ct_wrapper_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("WP4-P3b key/ciphertext wrapper probe")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
