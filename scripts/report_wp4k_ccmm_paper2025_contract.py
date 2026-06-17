from hegpt.ccmm_paper2025 import (
    CiphertextMatrixComponentPair,
    make_ccmm_plan,
    ccmm_paper2025,
)


def main():
    left = CiphertextMatrixComponentPair(
        A="A_U_placeholder",
        B="B_U_placeholder",
        shape=(4, 4),
        slots=4,
    )
    right = CiphertextMatrixComponentPair(
        A="A_V_placeholder",
        B="B_V_placeholder",
        shape=(4, 4),
        slots=4,
    )

    plan = make_ccmm_plan(left, right)

    print("=" * 100)
    print("WP4-K Park-2025 CCMM contract")
    print("left_shape:", plan.left_shape)
    print("right_shape:", plan.right_shape)
    print("output_shape:", plan.output_shape)
    print("uses_cmt:", plan.uses_cmt)
    print("uses_ppmm:", plan.uses_ppmm)
    print("backend:", plan.backend)

    print("=" * 100)
    print("calling ccmm_paper2025 should currently fail explicitly")

    try:
        ccmm_paper2025(left, right)
    except NotImplementedError as e:
        print("expected NotImplementedError:")
        print(str(e))
    else:
        raise RuntimeError("ccmm_paper2025 unexpectedly succeeded; this would be misleading.")


if __name__ == "__main__":
    main()
