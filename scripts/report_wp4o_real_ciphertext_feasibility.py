import json
import os
import subprocess
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def run_shell(cmd):
    p = subprocess.run(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "output": p.stdout[:20000],
    }


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def simplify_component_inspect(info):
    """
    Keep the report readable. Full inspect output can be very large.
    """
    out = {}
    for k, v in dict(info).items():
        if k != "parts":
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)

    parts_summary = []
    for part in info.get("parts", []):
        p = dict(part)
        towers = p.get("towers", [])
        parts_summary.append({
            "part_index": p.get("part_index"),
            "openfhe_name": p.get("openfhe_name"),
            "fides_raw_name": p.get("fides_raw_name"),
            "format": p.get("format"),
            "num_towers": p.get("num_towers"),
            "tower0_modulus_u64": towers[0].get("modulus_u64") if towers else None,
            "tower0_ring_dim": towers[0].get("ring_dim") if towers else None,
            "tower0_coeff_head": towers[0].get("coeff_head") if towers else None,
            "tower0_fnv1a64": towers[0].get("fnv1a64") if towers else None,
        })

    out["parts_summary"] = parts_summary
    return out


def main():
    slots = 8
    ring_dim = 1 << 14

    report = {
        "experiment": "wp4o_real_ciphertext_feasibility",
        "purpose": (
            "Probe whether the current FIDESlib/OpenFHE environment can support "
            "Park-2025 CC-MM real component bundles and coefficient encoding."
        ),
        "status": "probe_only_no_algorithm_change",
        "important_context": {
            "pcmm_backend_to_keep": "rt.component_linear_transform_gpu(rows, U)",
            "pcmm_backend_semantics": "plaintext matrix times ciphertext rows",
            "ccmm_target": "ciphertext matrix times ciphertext matrix using Park-2025 Algorithm 2/3",
        },
        "environment": {
            "cwd": os.getcwd(),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        },
    }

    # ------------------------------------------------------------------
    # A. Header/API grep probes.
    # ------------------------------------------------------------------
    header_roots = [
        "/root/fideslib/deps/openfhe-install/include",
        "/root/fideslib",
        "/workspace/FIDESlib-GPT",
    ]

    grep_patterns = [
        "MakeCoefPackedPlaintext",
        "CoefPacked",
        "COEF_PACKED_ENCODING",
        "CKKSPacked",
        "MakeCKKSPackedPlaintext",
        "EncodingType",
        "PackedEncoding",
    ]

    grep_results = {}
    for pat in grep_patterns:
        roots = " ".join(str(r) for r in header_roots if Path(r).exists())
        if roots:
            cmd = f"grep -RIn --include='*.h' --include='*.hpp' --include='*.cpp' --include='*.cu' '{pat}' {roots} | head -80"
            grep_results[pat] = run_shell(cmd)
        else:
            grep_results[pat] = {"cmd": "no roots", "returncode": -1, "output": ""}

    report["header_api_grep"] = grep_results

    # ------------------------------------------------------------------
    # B. Current slot-path ciphertext inspect.
    # ------------------------------------------------------------------
    cfg = HEConfig(
        ring_dim=ring_dim,
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

    x = np.arange(slots, dtype=np.float64) - 3.0

    with HERuntime(cfg, rotation_steps=[]) as rt:
        report["runtime_info"] = safe_call("info", lambda: rt.info())

        ct = rt.encrypt(x.tolist())

        report["current_encrypt_roundtrip"] = safe_call(
            "decrypt_current_encrypt",
            lambda: rt.decrypt(ct, logical_length=slots),
        )

        report["current_ciphertext_storage_state"] = safe_call(
            "ciphertext_storage_state",
            lambda: dict(rt.ciphertext_storage_state(ct)),
        )

        inspect = safe_call(
            "inspect_rlwe_components_cpu",
            lambda: rt.inspect_rlwe_components_cpu(ct, coeff_sample=16),
        )

        if inspect["ok"]:
            report["current_ciphertext_component_inspect_summary"] = simplify_component_inspect(inspect["value"])
        else:
            report["current_ciphertext_component_inspect_summary"] = inspect

        # Multiple row-wise ciphertexts, to inspect if they can form A/B component matrices.
        N_probe = 4
        M = np.arange(N_probe * slots, dtype=np.float64).reshape(N_probe, slots) / 10.0
        rows = [rt.encrypt(M[i].tolist()) for i in range(N_probe)]

        row_states = []
        row_component_summaries = []
        for i, row_ct in enumerate(rows):
            row_states.append(safe_call(
                f"row_{i}_state",
                lambda row_ct=row_ct: dict(rt.ciphertext_storage_state(row_ct)),
            ))

            insp = safe_call(
                f"row_{i}_inspect",
                lambda row_ct=row_ct: rt.inspect_rlwe_components_cpu(row_ct, coeff_sample=8),
            )
            if insp["ok"]:
                row_component_summaries.append(simplify_component_inspect(insp["value"]))
            else:
                row_component_summaries.append(insp)

        report["row_bundle_probe"] = {
            "N_probe": N_probe,
            "slots": slots,
            "row_states": row_states,
            "row_component_summaries": row_component_summaries,
        }

    # ------------------------------------------------------------------
    # C. Save.
    # ------------------------------------------------------------------
    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4o_real_ciphertext_feasibility_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("WP4-O real ciphertext feasibility report")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
