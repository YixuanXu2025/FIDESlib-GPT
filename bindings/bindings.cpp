#include "fideslib.hpp"
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <any>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <optional>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "/root/fideslib/deps/openfhe-install/include/openfhe/pke/openfhe.h"
#ifdef duration
#undef duration
#endif

#include "CKKS/Ciphertext.cuh"

namespace py = pybind11;
using namespace fideslib;

class CiphertextHandle {
public:
    CiphertextHandle() = default;
    explicit CiphertextHandle(Ciphertext<DCRTPoly> ct) : ct_(std::move(ct)) {}

    bool valid() const {
        return static_cast<bool>(ct_);
    }

    std::size_t level() const {
        require_valid();
        return ct_->GetLevel();
    }

    std::size_t noise_scale_deg() const {
        require_valid();
        return ct_->GetNoiseScaleDeg();
    }

private:
    friend 
class FidesCKKSContext;

    void require_valid() const {
        if (!ct_) {
            throw std::runtime_error("CiphertextHandle is empty");
        }
    

}

    Ciphertext<DCRTPoly> ct_;
};

class FidesCKKSContext {
public:
    FidesCKKSContext() = default;

    ~FidesCKKSContext() {
        try {
            close();
        } catch (...) {
        }
    }

    void init(
        std::uint32_t multiplicative_depth = 2,
        std::uint32_t scaling_mod_size = 50,
        std::uint32_t batch_size = 8,
        std::uint32_t ring_dim = (1u << 14),
        std::int32_t first_mod_size = -1,
        std::int32_t num_large_digits = -1,
        std::vector<int> devices = {0},
        bool plaintext_autoload = true,
        bool ciphertext_autoload = true,
        bool with_mult_key = true,
        std::vector<int> rotation_steps = {}
    ) {
        close();

        CCParams<CryptoContextCKKSRNS> params;
        params.SetMultiplicativeDepth(multiplicative_depth);
        params.SetScalingModSize(scaling_mod_size);
        params.SetBatchSize(batch_size);
        params.SetRingDim(ring_dim);
        params.SetPlaintextAutoload(plaintext_autoload);
        params.SetCiphertextAutoload(ciphertext_autoload);
        params.SetDevices(std::move(devices));

        if (first_mod_size > 0) {
            params.SetFirstModSize(static_cast<std::uint32_t>(first_mod_size));
        }
        if (num_large_digits > 0) {
            params.SetNumLargeDigits(static_cast<std::uint32_t>(num_large_digits));
        }

        cc_ = GenCryptoContext(params);
        cc_->Enable(PKE);
        cc_->Enable(KEYSWITCH);
        cc_->Enable(LEVELEDSHE);

        auto kp = cc_->KeyGen();
        if (!kp.publicKey || !kp.secretKey) {
            throw std::runtime_error("KeyGen failed");
        }

        pk_ = kp.publicKey;
        sk_ = kp.secretKey;

        if (with_mult_key) {
            cc_->EvalMultKeyGen(sk_);
        }

        if (!rotation_steps.empty()) {
            cc_->EvalRotateKeyGen(sk_, rotation_steps);
        }

        cc_->LoadContext(pk_);
    }

    py::dict info() const {
        require_ready();
        py::dict d;
        d["ring_dim"] = cc_->GetRingDimension();
        d["cyclotomic_order"] = cc_->GetCyclotomicOrder();
        return d;
    }

    CiphertextHandle encrypt(const std::vector<double>& x) {
        require_ready();
        auto pt = cc_->MakeCKKSPackedPlaintext(x);
        auto ct = cc_->Encrypt(pt, pk_);
        pt.reset();
        return CiphertextHandle(ct);
    }


    std::vector<std::vector<std::uint64_t>> export_secret_key_coeff_u64(std::size_t coeff_count = 0) {
        require_ready();

        using OpenFHEPrivateKey = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;

        auto* native_sk = std::any_cast<OpenFHEPrivateKey>(&sk_->pimpl);

        if (native_sk == nullptr || !(*native_sk)) {
            throw std::runtime_error(
                "export_secret_key_coeff_u64(): sk_->pimpl is not lbcrypto::PrivateKey<lbcrypto::DCRTPoly>"
            );
        }

        auto sk_poly = (*native_sk)->GetPrivateElement();
        sk_poly.SetFormat(COEFFICIENT);

        const std::size_t num_towers = sk_poly.GetNumOfElements();
        if (num_towers == 0) {
            throw std::runtime_error("export_secret_key_coeff_u64(): secret key has zero RNS towers");
        }

        const std::size_t ring_dim = sk_poly.GetElementAtIndex(0).GetLength();

        if (coeff_count == 0 || coeff_count > ring_dim) {
            coeff_count = ring_dim;
        }

        std::vector<std::vector<std::uint64_t>> out;
        out.reserve(num_towers);

        for (std::size_t t = 0; t < num_towers; ++t) {
            const auto& tower = sk_poly.GetElementAtIndex(t);

            std::vector<std::uint64_t> coeffs;
            coeffs.reserve(coeff_count);

            for (std::size_t i = 0; i < coeff_count; ++i) {
                coeffs.push_back(static_cast<std::uint64_t>(tower[i].ConvertToInt()));
            }

            out.push_back(std::move(coeffs));
        }

        return out;
    }

    CiphertextHandle encrypt_coeff_row_i64(const std::vector<std::int64_t>& coeffs) {
        require_ready();

        using OpenFHEContext = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
        using OpenFHEPublicKey = lbcrypto::PublicKey<lbcrypto::DCRTPoly>;
        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_cc = std::any_cast<OpenFHEContext>(&cc_->cpu);
        auto* native_pk = std::any_cast<OpenFHEPublicKey>(&pk_->pimpl);

        if (native_cc == nullptr || !(*native_cc)) {
            throw std::runtime_error("encrypt_coeff_row_i64(): cc_->cpu is not lbcrypto::CryptoContext<lbcrypto::DCRTPoly>");
        }
        if (native_pk == nullptr || !(*native_pk)) {
            throw std::runtime_error("encrypt_coeff_row_i64(): pk_->pimpl is not lbcrypto::PublicKey<lbcrypto::DCRTPoly>");
        }

        auto pt = (*native_cc)->MakeCoefPackedPlaintext(coeffs);
        OpenFHECiphertext native_ct = (*native_cc)->Encrypt(*native_pk, pt);

        auto out = std::make_shared<CiphertextImpl<DCRTPoly>>(CryptoContext<DCRTPoly>(cc_));
        out->cpu = std::make_any<OpenFHECiphertext>(std::move(native_ct));
        out->loaded = false;
        out->parent_context = cc_;

        return CiphertextHandle(out);
    }


    CiphertextHandle wrap_native_openfhe_ciphertext_for_coeff_ops(
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> native_ct
    ) {
        auto out = std::make_shared<CiphertextImpl<DCRTPoly>>(CryptoContext<DCRTPoly>(cc_));
        out->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(native_ct));
        out->loaded = false;
        out->parent_context = cc_;
        return CiphertextHandle(out);
    }


    CiphertextHandle add_coeff_ct(
        CiphertextHandle& a,
        CiphertextHandle& b
    ) {
        require_ready();
        a.require_valid();
        b.require_valid();

        using OpenFHEContext = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_cc = std::any_cast<OpenFHEContext>(&cc_->cpu);
        auto* native_a = std::any_cast<OpenFHECiphertext>(&a.ct_->cpu);
        auto* native_b = std::any_cast<OpenFHECiphertext>(&b.ct_->cpu);

        if (native_cc == nullptr || !(*native_cc)) {
            throw std::runtime_error("add_coeff_ct(): cc_->cpu is not native OpenFHE context");
        }
        if (native_a == nullptr || !(*native_a) || native_b == nullptr || !(*native_b)) {
            throw std::runtime_error("add_coeff_ct(): input ciphertext cpu is not native OpenFHE ciphertext");
        }

        auto out_native = (*native_cc)->EvalAdd(*native_a, *native_b);
        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }


    CiphertextHandle monomial_mul_coeff_ct(
        CiphertextHandle& h,
        std::uint32_t exp
    ) {
        require_ready();
        h.require_valid();

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_ct = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);
        if (native_ct == nullptr || !(*native_ct)) {
            throw std::runtime_error("monomial_mul_coeff_ct(): ciphertext cpu is not native OpenFHE ciphertext");
        }

        auto out_native = (*native_ct)->Clone();

        auto elems = out_native->GetElements();
        if (elems.empty()) {
            throw std::runtime_error("monomial_mul_coeff_ct(): empty ciphertext elements");
        }

        // Build monomial X^exp modulo X^N + 1 as a DCRTPoly in COEFFICIENT format.
        // For exp >= N, X^(N+t) = -X^t.
        auto params = elems[0].GetParams();
        const std::uint32_t N = elems[0].GetRingDimension();

        std::vector<int64_t> mono(N, 0);
        std::uint32_t e = exp % (2u * N);
        bool neg = false;
        if (e >= N) {
            e -= N;
            neg = true;
        }
        mono[e] = neg ? -1 : 1;

        lbcrypto::DCRTPoly monomial(params, COEFFICIENT, true);
        monomial = mono;
        monomial.SetFormat(EVALUATION);

        for (auto& elem : elems) {
            elem.SetFormat(EVALUATION);
            elem *= monomial;
        }

        out_native->SetElements(std::move(elems));
        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }




    py::dict export_component_coeff_matrix_u64(
        std::vector<CiphertextHandle> rows,
        std::uint32_t part,
        std::uint32_t coeff_count = 0
    ) {
        require_ready();

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        if (rows.empty()) {
            throw std::runtime_error("export_component_coeff_matrix_u64(): rows is empty");
        }

        py::list py_rows;
        std::uint32_t expected_towers = 0;
        std::uint32_t expected_ring_dim = 0;
        py::list moduli;

        for (std::size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
            auto& h = rows[row_idx];
            h.require_valid();

            auto* native_ct = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);
            if (native_ct == nullptr || !(*native_ct)) {
                throw std::runtime_error("export_component_coeff_matrix_u64(): input is not native OpenFHE ciphertext");
            }

            const auto& elems = (*native_ct)->GetElements();
            if (part >= elems.size()) {
                throw std::runtime_error("export_component_coeff_matrix_u64(): part out of range");
            }

            auto poly = elems[part];
            poly.SetFormat(COEFFICIENT);

            const auto& towers = poly.GetAllElements();
            if (towers.empty()) {
                throw std::runtime_error("export_component_coeff_matrix_u64(): component has no RNS towers");
            }

            if (row_idx == 0) {
                expected_towers = static_cast<std::uint32_t>(towers.size());
                expected_ring_dim = static_cast<std::uint32_t>(towers[0].GetValues().GetLength());

                for (std::size_t t = 0; t < towers.size(); ++t) {
                    moduli.append(static_cast<std::uint64_t>(towers[t].GetModulus().ConvertToInt()));
                }

                if (coeff_count == 0) {
                    coeff_count = expected_ring_dim;
                }

                if (coeff_count > expected_ring_dim) {
                    throw std::runtime_error("export_component_coeff_matrix_u64(): coeff_count exceeds ring dimension");
                }
            } else {
                if (towers.size() != expected_towers) {
                    throw std::runtime_error("export_component_coeff_matrix_u64(): inconsistent tower count");
                }
                if (towers[0].GetValues().GetLength() != expected_ring_dim) {
                    throw std::runtime_error("export_component_coeff_matrix_u64(): inconsistent ring dimension");
                }
            }

            py::list tower_list;

            for (std::size_t t = 0; t < towers.size(); ++t) {
                const auto& tower = towers[t];
                const auto& vals = tower.GetValues();

                py::list coeffs;
                for (std::uint32_t k = 0; k < coeff_count; ++k) {
                    coeffs.append(static_cast<std::uint64_t>(vals[k].ConvertToInt()));
                }

                py::dict tower_entry;
                tower_entry["tower_index"] = static_cast<std::uint32_t>(t);
                tower_entry["modulus_u64"] = static_cast<std::uint64_t>(tower.GetModulus().ConvertToInt());
                tower_entry["coeffs_u64"] = coeffs;
                tower_list.append(tower_entry);
            }

            py::dict row_entry;
            row_entry["row_index"] = static_cast<std::uint32_t>(row_idx);
            row_entry["part"] = part;
            row_entry["towers"] = tower_list;
            py_rows.append(row_entry);
        }

        py::dict out;
        out["num_rows"] = static_cast<std::uint32_t>(rows.size());
        out["part"] = part;
        out["num_towers"] = expected_towers;
        out["ring_dim"] = expected_ring_dim;
        out["coeff_count"] = coeff_count;
        out["moduli_u64"] = moduli;
        out["rows"] = py_rows;
        out["format"] = "COEFFICIENT";
        out["zh"] = "rows × towers × coeffs 的 raw RNS coefficient matrix 导出结果";
        return out;
    }


    CiphertextHandle import_1part_coeff_u64(
        CiphertextHandle& template_ct,
        py::list towers_coeffs
    ) {
        require_ready();
        template_ct.require_valid();

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_template = std::any_cast<OpenFHECiphertext>(&template_ct.ct_->cpu);
        if (native_template == nullptr || !(*native_template)) {
            throw std::runtime_error("import_1part_coeff_u64(): template is not native OpenFHE ciphertext");
        }

        const auto& elems = (*native_template)->GetElements();
        if (elems.empty()) {
            throw std::runtime_error("import_1part_coeff_u64(): template has no elements");
        }

        auto poly_out = elems[0];
        poly_out.SetFormat(COEFFICIENT);
        poly_out.SetValuesToZero();

        auto& out_towers = poly_out.GetAllElements();

        if (towers_coeffs.size() > out_towers.size()) {
            throw std::runtime_error("import_1part_coeff_u64(): too many tower coefficient arrays");
        }

        for (std::size_t tower_idx = 0; tower_idx < towers_coeffs.size(); ++tower_idx) {
            py::list coeffs = py::cast<py::list>(towers_coeffs[tower_idx]);

            auto tower = out_towers[tower_idx];
            tower.SetFormat(COEFFICIENT);

            const std::size_t ring_dim = tower.GetValues().GetLength();
            if (coeffs.size() > ring_dim) {
                throw std::runtime_error("import_1part_coeff_u64(): coeff list exceeds ring dimension");
            }

            const auto modulus = tower.GetModulus();
            const std::uint64_t modulus_u64 = static_cast<std::uint64_t>(modulus.ConvertToInt());

            lbcrypto::NativeVector vals(static_cast<usint>(ring_dim), modulus);

            for (std::size_t k = 0; k < ring_dim; ++k) {
                vals[k] = lbcrypto::NativeInteger(0);
            }

            for (std::size_t k = 0; k < coeffs.size(); ++k) {
                std::uint64_t raw = coeffs[k].cast<std::uint64_t>();
                vals[k] = lbcrypto::NativeInteger(raw % modulus_u64);
            }

            tower.SetValues(std::move(vals), COEFFICIENT);
            poly_out.SetElementAtIndex(static_cast<usint>(tower_idx), std::move(tower));
        }

        auto out_native = (*native_template)->Clone();
        out_native->SetElements(std::vector<lbcrypto::DCRTPoly>{std::move(poly_out)});

        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }

    CiphertextHandle component_mul_part_coeff_ct(
        CiphertextHandle& a,
        std::uint32_t part_a,
        CiphertextHandle& b,
        std::uint32_t part_b
    ) {
        require_ready();
        a.require_valid();
        b.require_valid();

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_a = std::any_cast<OpenFHECiphertext>(&a.ct_->cpu);
        auto* native_b = std::any_cast<OpenFHECiphertext>(&b.ct_->cpu);

        if (native_a == nullptr || !(*native_a)) {
            throw std::runtime_error("component_mul_part_coeff_ct(): first input is not native OpenFHE ciphertext");
        }
        if (native_b == nullptr || !(*native_b)) {
            throw std::runtime_error("component_mul_part_coeff_ct(): second input is not native OpenFHE ciphertext");
        }

        const auto& elems_a = (*native_a)->GetElements();
        const auto& elems_b = (*native_b)->GetElements();

        if (part_a >= elems_a.size()) {
            throw std::runtime_error("component_mul_part_coeff_ct(): part_a out of range");
        }
        if (part_b >= elems_b.size()) {
            throw std::runtime_error("component_mul_part_coeff_ct(): part_b out of range");
        }

        auto pa = elems_a[part_a];
        auto pb = elems_b[part_b];

        pa.SetFormat(EVALUATION);
        pb.SetFormat(EVALUATION);

        lbcrypto::DCRTPoly prod = pa * pb;

        auto out_native = (*native_a)->Clone();
        out_native->SetElements(std::vector<lbcrypto::DCRTPoly>{std::move(prod)});

        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }


    CiphertextHandle component_add_1part_coeff_ct(
        CiphertextHandle& a,
        CiphertextHandle& b
    ) {
        require_ready();
        a.require_valid();
        b.require_valid();

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_a = std::any_cast<OpenFHECiphertext>(&a.ct_->cpu);
        auto* native_b = std::any_cast<OpenFHECiphertext>(&b.ct_->cpu);

        if (native_a == nullptr || !(*native_a)) {
            throw std::runtime_error("component_add_1part_coeff_ct(): first input is not native OpenFHE ciphertext");
        }
        if (native_b == nullptr || !(*native_b)) {
            throw std::runtime_error("component_add_1part_coeff_ct(): second input is not native OpenFHE ciphertext");
        }

        const auto& elems_a = (*native_a)->GetElements();
        const auto& elems_b = (*native_b)->GetElements();

        if (elems_a.size() != 1 || elems_b.size() != 1) {
            throw std::runtime_error(
                "component_add_1part_coeff_ct(): expected both inputs to have exactly one component"
            );
        }

        auto ea = elems_a[0];
        auto eb = elems_b[0];

        ea.SetFormat(EVALUATION);
        eb.SetFormat(EVALUATION);

        lbcrypto::DCRTPoly sum = ea + eb;

        auto out_native = (*native_a)->Clone();
        out_native->SetElements(std::vector<lbcrypto::DCRTPoly>{std::move(sum)});

        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }



    CiphertextHandle assemble_2part_from_1parts_coeff_ct(
        CiphertextHandle& c0,
        CiphertextHandle& c1
    ) {
        require_ready();
        c0.require_valid();
        c1.require_valid();

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_c0 = std::any_cast<OpenFHECiphertext>(&c0.ct_->cpu);
        auto* native_c1 = std::any_cast<OpenFHECiphertext>(&c1.ct_->cpu);

        if (native_c0 == nullptr || !(*native_c0)) {
            throw std::runtime_error("assemble_2part_from_1parts_coeff_ct(): c0 is not native OpenFHE ciphertext");
        }
        if (native_c1 == nullptr || !(*native_c1)) {
            throw std::runtime_error("assemble_2part_from_1parts_coeff_ct(): c1 is not native OpenFHE ciphertext");
        }

        const auto& e0s = (*native_c0)->GetElements();
        const auto& e1s = (*native_c1)->GetElements();

        if (e0s.size() != 1 || e1s.size() != 1) {
            throw std::runtime_error(
                "assemble_2part_from_1parts_coeff_ct(): expected both inputs to have exactly one component"
            );
        }

        auto out_native = (*native_c0)->Clone();
        out_native->SetElements(std::vector<lbcrypto::DCRTPoly>{
            e0s[0],
            e1s[0],
        });

        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }

    CiphertextHandle assemble_3part_from_1parts_coeff_ct(
        CiphertextHandle& c0,
        CiphertextHandle& c1,
        CiphertextHandle& c2
    ) {
        require_ready();
        c0.require_valid();
        c1.require_valid();
        c2.require_valid();

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_c0 = std::any_cast<OpenFHECiphertext>(&c0.ct_->cpu);
        auto* native_c1 = std::any_cast<OpenFHECiphertext>(&c1.ct_->cpu);
        auto* native_c2 = std::any_cast<OpenFHECiphertext>(&c2.ct_->cpu);

        if (native_c0 == nullptr || !(*native_c0)) {
            throw std::runtime_error("assemble_3part_from_1parts_coeff_ct(): c0 is not native OpenFHE ciphertext");
        }
        if (native_c1 == nullptr || !(*native_c1)) {
            throw std::runtime_error("assemble_3part_from_1parts_coeff_ct(): c1 is not native OpenFHE ciphertext");
        }
        if (native_c2 == nullptr || !(*native_c2)) {
            throw std::runtime_error("assemble_3part_from_1parts_coeff_ct(): c2 is not native OpenFHE ciphertext");
        }

        const auto& e0s = (*native_c0)->GetElements();
        const auto& e1s = (*native_c1)->GetElements();
        const auto& e2s = (*native_c2)->GetElements();

        if (e0s.size() != 1 || e1s.size() != 1 || e2s.size() != 1) {
            throw std::runtime_error(
                "assemble_3part_from_1parts_coeff_ct(): expected all inputs to have exactly one component"
            );
        }

        auto out_native = (*native_c0)->Clone();
        out_native->SetElements(std::vector<lbcrypto::DCRTPoly>{
            e0s[0],
            e1s[0],
            e2s[0],
        });

        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }

    CiphertextHandle mul_coeff_ct_no_relin(
        CiphertextHandle& a,
        CiphertextHandle& b
    ) {
        require_ready();
        a.require_valid();
        b.require_valid();

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_a = std::any_cast<OpenFHECiphertext>(&a.ct_->cpu);
        auto* native_b = std::any_cast<OpenFHECiphertext>(&b.ct_->cpu);

        if (native_a == nullptr || !(*native_a)) {
            throw std::runtime_error("mul_coeff_ct_no_relin(): first input is not native OpenFHE ciphertext");
        }
        if (native_b == nullptr || !(*native_b)) {
            throw std::runtime_error("mul_coeff_ct_no_relin(): second input is not native OpenFHE ciphertext");
        }

        const auto& elems_a_const = (*native_a)->GetElements();
        const auto& elems_b_const = (*native_b)->GetElements();

        if (elems_a_const.size() != 2 || elems_b_const.size() != 2) {
            throw std::runtime_error(
                "mul_coeff_ct_no_relin(): expected both inputs to have exactly two RLWE components"
            );
        }

        auto a0 = elems_a_const[0];
        auto a1 = elems_a_const[1];
        auto b0 = elems_b_const[0];
        auto b1 = elems_b_const[1];

        // DCRTPoly multiplication is supported in EVALUATION format.
        a0.SetFormat(EVALUATION);
        a1.SetFormat(EVALUATION);
        b0.SetFormat(EVALUATION);
        b1.SetFormat(EVALUATION);

        lbcrypto::DCRTPoly e0 = a0 * b0;
        lbcrypto::DCRTPoly e1 = a0 * b1 + a1 * b0;
        lbcrypto::DCRTPoly e2 = a1 * b1;

        auto out_native = (*native_a)->Clone();
        out_native->SetElements(std::vector<lbcrypto::DCRTPoly>{
            std::move(e0),
            std::move(e1),
            std::move(e2),
        });

        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }


    CiphertextHandle relinearize_coeff_ct(
        CiphertextHandle& h
    ) {
        require_ready();
        h.require_valid();

        using OpenFHEContext = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_cc = std::any_cast<OpenFHEContext>(&cc_->cpu);
        auto* native_ct = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);

        if (native_cc == nullptr || !(*native_cc)) {
            throw std::runtime_error("relinearize_coeff_ct(): cc_->cpu is not native OpenFHE context");
        }
        if (native_ct == nullptr || !(*native_ct)) {
            throw std::runtime_error("relinearize_coeff_ct(): ciphertext cpu is not native OpenFHE ciphertext");
        }

        const auto& elems = (*native_ct)->GetElements();
        if (elems.size() < 3) {
            throw std::runtime_error(
                "relinearize_coeff_ct(): expected a ciphertext with at least three RLWE components"
            );
        }

        auto out_native = (*native_cc)->Relinearize(*native_ct);

        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }


    CiphertextHandle compress_coeff_ct(
        CiphertextHandle& h,
        std::uint32_t towers_left = 1
    ) {
        require_ready();
        h.require_valid();

        using OpenFHEContext = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_cc = std::any_cast<OpenFHEContext>(&cc_->cpu);
        auto* native_ct = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);

        if (native_cc == nullptr || !(*native_cc)) {
            throw std::runtime_error("compress_coeff_ct(): cc_->cpu is not native OpenFHE context");
        }
        if (native_ct == nullptr || !(*native_ct)) {
            throw std::runtime_error("compress_coeff_ct(): ciphertext cpu is not native OpenFHE ciphertext");
        }

        if (towers_left == 0) {
            throw std::runtime_error("compress_coeff_ct(): towers_left must be >= 1");
        }

        auto out_native = (*native_cc)->Compress(*native_ct, towers_left);

        return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
    }

    CiphertextHandle eval_automorphism_coeff_ct(
        CiphertextHandle& h,
        std::uint32_t alpha
    ) {
        require_ready();
        h.require_valid();

        using OpenFHEContext = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
        using OpenFHEPrivateKey = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;
        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_cc = std::any_cast<OpenFHEContext>(&cc_->cpu);
        auto* native_sk = std::any_cast<OpenFHEPrivateKey>(&sk_->pimpl);
        auto* native_ct = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);

        if (native_cc == nullptr || !(*native_cc)) {
            throw std::runtime_error(
                "eval_automorphism_coeff_ct(): cc_->cpu is not lbcrypto::CryptoContext<lbcrypto::DCRTPoly>"
            );
        }
        if (native_sk == nullptr || !(*native_sk)) {
            throw std::runtime_error(
                "eval_automorphism_coeff_ct(): sk_->pimpl is not lbcrypto::PrivateKey<lbcrypto::DCRTPoly>"
            );
        }
        if (native_ct == nullptr || !(*native_ct)) {
            throw std::runtime_error(
                "eval_automorphism_coeff_ct(): ciphertext cpu is not lbcrypto::Ciphertext<lbcrypto::DCRTPoly>"
            );
        }

        if ((alpha % 2u) == 0u) {
            throw std::runtime_error(
                "eval_automorphism_coeff_ct(): alpha must be odd for X -> X^alpha"
            );
        }

        // Auto(ct; 1) is the identity automorphism X -> X.
        // OpenFHE may generate an empty automorphism key map for alpha=1,
        // so bypass EvalAutomorphism and return a cloned ciphertext directly.
        if (alpha == 1u) {
            auto out_native = (*native_ct)->Clone();
            return wrap_native_openfhe_ciphertext_for_coeff_ops(std::move(out_native));
        }

        std::vector<std::uint32_t> index_list = {alpha};

        // Generate automorphism key for this alpha.
        //
        // Important:
        // Some OpenFHE paths may return an empty map while storing the generated
        // evaluation keys in the CryptoContext registry. Therefore:
        //   1. Try the returned map first.
        //   2. If empty/missing alpha, fetch the registered key map by ciphertext keyTag.
        auto eval_keys = (*native_cc)->EvalAutomorphismKeyGen(*native_sk, index_list);

        OpenFHECiphertext out_native;

        if (
            eval_keys != nullptr
            && !eval_keys->empty()
            && eval_keys->find(alpha) != eval_keys->end()
        ) {
            out_native = (*native_cc)->EvalAutomorphism(*native_ct, alpha, *eval_keys);
        } else {
            auto registered_keys = (*native_cc)->GetEvalAutomorphismKeyMap((*native_ct)->GetKeyTag());

            if (
                registered_keys.empty()
                || registered_keys.find(alpha) == registered_keys.end()
            ) {
                throw std::runtime_error(
                    "eval_automorphism_coeff_ct(): empty/missing automorphism key map after EvalAutomorphismKeyGen and registry lookup"
                );
            }

            out_native = (*native_cc)->EvalAutomorphism(*native_ct, alpha, registered_keys);
        }

        auto out = std::make_shared<CiphertextImpl<DCRTPoly>>(CryptoContext<DCRTPoly>(cc_));
        out->cpu = std::make_any<OpenFHECiphertext>(std::move(out_native));
        out->loaded = false;
        out->parent_context = cc_;

        return CiphertextHandle(out);
    }


    std::vector<std::int64_t> decrypt_coeff_row_i64(
        CiphertextHandle& h,
        std::size_t logical_length = 0
    ) {
        require_ready();
        h.require_valid();

        using OpenFHEContext = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
        using OpenFHEPrivateKey = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;
        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_cc = std::any_cast<OpenFHEContext>(&cc_->cpu);
        auto* native_sk = std::any_cast<OpenFHEPrivateKey>(&sk_->pimpl);
        auto* native_ct = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);

        if (native_cc == nullptr || !(*native_cc)) {
            throw std::runtime_error("decrypt_coeff_row_i64(): cc_->cpu is not lbcrypto::CryptoContext<lbcrypto::DCRTPoly>");
        }
        if (native_sk == nullptr || !(*native_sk)) {
            throw std::runtime_error("decrypt_coeff_row_i64(): sk_->pimpl is not lbcrypto::PrivateKey<lbcrypto::DCRTPoly>");
        }
        if (native_ct == nullptr || !(*native_ct)) {
            throw std::runtime_error("decrypt_coeff_row_i64(): ciphertext cpu is not lbcrypto::Ciphertext<lbcrypto::DCRTPoly>");
        }

        lbcrypto::Plaintext pt_dec;
        auto dec_res = (*native_cc)->Decrypt(*native_sk, *native_ct, &pt_dec);
        if (!dec_res.isValid) {
            throw std::runtime_error("decrypt_coeff_row_i64(): native OpenFHE Decrypt failed");
        }

        const auto& vals_ref = pt_dec->GetCoefPackedValue();
        std::vector<std::int64_t> out(vals_ref.begin(), vals_ref.end());

        if (logical_length > 0 && logical_length < out.size()) {
            out.resize(logical_length);
        }

        return out;
    }

    std::vector<std::int64_t> manual_decrypt_coeff_row_i64(
        CiphertextHandle& h,
        std::size_t logical_length = 0
    ) {
        require_ready();
        h.require_valid();

        using OpenFHEPrivateKey = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;
        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_sk = std::any_cast<OpenFHEPrivateKey>(&sk_->pimpl);
        auto* native_ct = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);

        if (native_sk == nullptr || !(*native_sk)) {
            throw std::runtime_error(
                "manual_decrypt_coeff_row_i64(): sk_->pimpl is not lbcrypto::PrivateKey<lbcrypto::DCRTPoly>"
            );
        }
        if (native_ct == nullptr || !(*native_ct)) {
            throw std::runtime_error(
                "manual_decrypt_coeff_row_i64(): ciphertext cpu is not lbcrypto::Ciphertext<lbcrypto::DCRTPoly>"
            );
        }

        const auto& elems = (*native_ct)->GetElements();
        if (elems.size() < 2) {
            throw std::runtime_error(
                "manual_decrypt_coeff_row_i64(): expected at least two ciphertext elements"
            );
        }

        // OpenFHE/FIDESlib convention:
        //   elems[0] = c0
        //   elems[1] = c1
        // RLWE phase:
        //   phase = c0 + c1 * sk
        auto c0 = elems[0];
        auto c1 = elems[1];
        auto sk_poly = (*native_sk)->GetPrivateElement();

        c0.SetFormat(EVALUATION);
        c1.SetFormat(EVALUATION);
        sk_poly.SetFormat(EVALUATION);

        lbcrypto::DCRTPoly phase = c0 + c1 * sk_poly;
        phase.SetFormat(COEFFICIENT);

        auto native_poly = phase.CRTInterpolate();
        const auto& modulus = native_poly.GetModulus();
        auto half = modulus / 2;

        std::size_t n = native_poly.GetLength();
        if (logical_length > 0 && logical_length < n) {
            n = logical_length;
        }

        std::vector<std::int64_t> out;
        out.reserve(n);

        for (std::size_t i = 0; i < n; ++i) {
            auto v = native_poly[i];

            if (v > half) {
                auto signed_abs = modulus - v;
                std::uint64_t u = signed_abs.ConvertToInt();
                out.push_back(-static_cast<std::int64_t>(u));
            } else {
                std::uint64_t u = v.ConvertToInt();
                out.push_back(static_cast<std::int64_t>(u));
            }
        }

        return out;
    }

    std::map<std::string, std::vector<std::int64_t>> manual_decrypt_coeff_row_i64_variants(
        CiphertextHandle& h,
        std::size_t logical_length = 0
    ) {
        require_ready();
        h.require_valid();

        using OpenFHEPrivateKey = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;
        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        auto* native_sk = std::any_cast<OpenFHEPrivateKey>(&sk_->pimpl);
        auto* native_ct = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);

        if (native_sk == nullptr || !(*native_sk)) {
            throw std::runtime_error(
                "manual_decrypt_coeff_row_i64_variants(): sk_->pimpl is not lbcrypto::PrivateKey<lbcrypto::DCRTPoly>"
            );
        }
        if (native_ct == nullptr || !(*native_ct)) {
            throw std::runtime_error(
                "manual_decrypt_coeff_row_i64_variants(): ciphertext cpu is not lbcrypto::Ciphertext<lbcrypto::DCRTPoly>"
            );
        }

        const auto& elems = (*native_ct)->GetElements();
        if (elems.size() < 2) {
            throw std::runtime_error(
                "manual_decrypt_coeff_row_i64_variants(): expected at least two ciphertext elements"
            );
        }

        auto c0 = elems[0];
        auto c1 = elems[1];
        auto sk_poly = (*native_sk)->GetPrivateElement();

        auto to_signed_coeffs = [&](lbcrypto::DCRTPoly poly) {
            poly.SetFormat(COEFFICIENT);
            auto native_poly = poly.CRTInterpolate();

            const auto& modulus = native_poly.GetModulus();
            auto half = modulus / 2;

            std::size_t n = native_poly.GetLength();
            if (logical_length > 0 && logical_length < n) {
                n = logical_length;
            }

            std::vector<std::int64_t> out;
            out.reserve(n);

            for (std::size_t i = 0; i < n; ++i) {
                auto v = native_poly[i];

                if (v > half) {
                    auto signed_abs = modulus - v;
                    std::uint64_t u = signed_abs.ConvertToInt();
                    out.push_back(-static_cast<std::int64_t>(u));
                } else {
                    std::uint64_t u = v.ConvertToInt();
                    out.push_back(static_cast<std::int64_t>(u));
                }
            }

            return out;
        };

        std::map<std::string, std::vector<std::int64_t>> out;

        // Multiplication should normally happen in EVALUATION format.
        auto c0e = c0;
        auto c1e = c1;
        auto ske = sk_poly;
        c0e.SetFormat(EVALUATION);
        c1e.SetFormat(EVALUATION);
        ske.SetFormat(EVALUATION);

        out["eval_c0_plus_c1s"]  = to_signed_coeffs(c0e + c1e * ske);
        out["eval_c0_minus_c1s"] = to_signed_coeffs(c0e - c1e * ske);
        out["eval_c1_plus_c0s"]  = to_signed_coeffs(c1e + c0e * ske);
        out["eval_c1_minus_c0s"] = to_signed_coeffs(c1e - c0e * ske);

        out["eval_neg_c0_minus_c1s"] = to_signed_coeffs(-c0e - c1e * ske);
        out["eval_neg_c0_plus_c1s"]  = to_signed_coeffs(-c0e + c1e * ske);
        out["eval_neg_c1_minus_c0s"] = to_signed_coeffs(-c1e - c0e * ske);
        out["eval_neg_c1_plus_c0s"]  = to_signed_coeffs(-c1e + c0e * ske);

        // Also test coefficient-format multiplication, in case current operator semantics differ.
        auto c0c = c0;
        auto c1c = c1;
        auto skc = sk_poly;
        c0c.SetFormat(COEFFICIENT);
        c1c.SetFormat(COEFFICIENT);
        skc.SetFormat(COEFFICIENT);

        out["coef_c0_plus_c1s"]  = to_signed_coeffs(c0c + c1c * skc);
        out["coef_c0_minus_c1s"] = to_signed_coeffs(c0c - c1c * skc);
        out["coef_c1_plus_c0s"]  = to_signed_coeffs(c1c + c0c * skc);
        out["coef_c1_minus_c0s"] = to_signed_coeffs(c1c - c0c * skc);

        return out;
    }




std::vector<double> decrypt(CiphertextHandle& h, std::size_t logical_length = 0) {
        require_ready();
        h.require_valid();

        Plaintext pt_dec;
        auto dec_res = cc_->Decrypt(h.ct_, sk_, &pt_dec);
        if (!dec_res.isValid) {
            throw std::runtime_error("Decrypt failed");
        }

        if (logical_length > 0) {
            pt_dec->SetLength(logical_length);
        }

        auto out = pt_dec->GetRealPackedValue();
        pt_dec.reset();
        return out;
    }

    CiphertextHandle eval_add_ct(const CiphertextHandle& a, const CiphertextHandle& b) {
        require_ready();
        a.require_valid();
        b.require_valid();

        auto ct_out = cc_->EvalAdd(a.ct_, b.ct_);
        return CiphertextHandle(ct_out);
    }

    CiphertextHandle eval_add_plain_ct(const CiphertextHandle& a, const std::vector<double>& plain) {
        require_ready();
        a.require_valid();

        auto pt = cc_->MakeCKKSPackedPlaintext(plain);
        auto ct_out = cc_->EvalAdd(a.ct_, pt);
        pt.reset();
        return CiphertextHandle(ct_out);
    }

    CiphertextHandle eval_mult_scalar_ct(const CiphertextHandle& a, double scalar) {
        require_ready();
        a.require_valid();

        auto ct_out = cc_->EvalMult(a.ct_, scalar);
        return CiphertextHandle(ct_out);
    }

    CiphertextHandle eval_mult_plain_ct(const CiphertextHandle& a, const std::vector<double>& plain) {
        require_ready();
        a.require_valid();

        auto pt = cc_->MakeCKKSPackedPlaintext(plain);
        auto ct_out = cc_->EvalMult(a.ct_, pt);
        pt.reset();
        return CiphertextHandle(ct_out);
    }

    CiphertextHandle eval_rotate_ct(const CiphertextHandle& a, int steps) {
        require_ready();
        a.require_valid();

        auto ct_out = cc_->EvalRotate(a.ct_, steps);
        return CiphertextHandle(ct_out);
    }





    py::dict inspect_rlwe_components_cpu(const CiphertextHandle& h, std::size_t coeff_sample = 8) {
        require_ready();
        h.require_valid();

        py::dict out;
        out["valid"] = true;
        out["fides_loaded"] = h.ct_->loaded;
        out["fides_gpu_handle"] = h.ct_->gpu;
        out["fides_level"] = h.ct_->GetLevel();
        out["fides_noise_scale_deg"] = h.ct_->GetNoiseScaleDeg();

        if (!h.ct_->cpu.has_value()) {
            throw std::runtime_error("inspect_rlwe_components_cpu(): ciphertext has no CPU object in ct_->cpu. Try ciphertext_autoload=False for this debug probe.");
        }

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
        const OpenFHECiphertext* ct_ptr = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);

        if (ct_ptr == nullptr) {
            throw std::runtime_error("inspect_rlwe_components_cpu(): ct_->cpu is not lbcrypto::Ciphertext<lbcrypto::DCRTPoly>");
        }

        const auto& ct = *ct_ptr;
        if (!ct) {
            throw std::runtime_error("inspect_rlwe_components_cpu(): OpenFHE ciphertext pointer is null");
        }

        const auto& elems = ct->GetElements();

        out["num_parts"] = elems.size();
        out["openfhe_level"] = ct->GetLevel();
        out["openfhe_noise_scale_deg"] = ct->GetNoiseScaleDeg();
        out["openfhe_scaling_factor"] = ct->GetScalingFactor();
        out["openfhe_slots"] = ct->GetSlots();
        out["openfhe_encoding_type"] = static_cast<int>(ct->GetEncodingType());
        out["openfhe_key_tag"] = ct->GetKeyTag();

        py::list parts;

        for (std::size_t part_idx = 0; part_idx < elems.size(); ++part_idx) {
            py::dict part_info;
            part_info["part_index"] = part_idx;

            if (part_idx == 0) {
                part_info["openfhe_name"] = "GetElements()[0]";
                part_info["fides_raw_name"] = "sub_0 / c0";
            } else if (part_idx == 1) {
                part_info["openfhe_name"] = "GetElements()[1]";
                part_info["fides_raw_name"] = "sub_1 / c1";
            } else {
                part_info["openfhe_name"] = std::string("GetElements()[") + std::to_string(part_idx) + "]";
                part_info["fides_raw_name"] = "extra_component";
            }

            lbcrypto::DCRTPoly poly = elems[part_idx];
            part_info["format"] = static_cast<int>(poly.GetFormat());

            auto towers = poly.GetAllElements();
            part_info["num_towers"] = towers.size();

            py::list tower_infos;
            for (std::size_t tower_idx = 0; tower_idx < towers.size(); ++tower_idx) {
                const auto& tower = towers[tower_idx];
                const auto& values = tower.GetValues();

                py::dict tower_info;
                tower_info["tower_index"] = tower_idx;
                tower_info["modulus_u64"] = static_cast<std::uint64_t>(tower.GetModulus().ConvertToInt());
                tower_info["ring_dim"] = values.GetLength();

                std::size_t take = coeff_sample;
                if (take > values.GetLength()) {
                    take = values.GetLength();
                }

                std::uint64_t hash = 1469598103934665603ULL;
                py::list head;
                for (std::size_t i = 0; i < values.GetLength(); ++i) {
                    const std::uint64_t v = static_cast<std::uint64_t>(values[i].ConvertToInt());
                    hash ^= v;
                    hash *= 1099511628211ULL;

                    if (i < take) {
                        head.append(v);
                    }
                }

                tower_info["coeff_head"] = head;
                tower_info["fnv1a64"] = std::to_string(hash);
                tower_infos.append(tower_info);
            }

            part_info["towers"] = tower_infos;
            parts.append(part_info);
        }

        out["parts"] = parts;
        return out;
    }

    py::dict inspect_coeff_row_component_matrix(
        const std::vector<CiphertextHandle>& rows,
        std::size_t coeff_sample = 8
    ) {
        require_ready();

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        py::dict report;
        report["num_rows"] = rows.size();
        report["coeff_sample"] = coeff_sample;

        py::list row_reports;

        std::size_t expected_parts = 0;
        std::size_t expected_towers = 0;
        std::size_t expected_ring_dim = 0;
        bool consistent = true;

        for (std::size_t r = 0; r < rows.size(); ++r) {
            auto h = rows[r];
            h.require_valid();

            auto* native_ct = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);
            if (native_ct == nullptr || !(*native_ct)) {
                throw std::runtime_error(
                    "inspect_coeff_row_component_matrix(): row ciphertext cpu is not lbcrypto::Ciphertext<lbcrypto::DCRTPoly>"
                );
            }

            const auto& elems = (*native_ct)->GetElements();

            py::dict rr;
            rr["row_index"] = r;
            rr["num_parts"] = elems.size();
            rr["level"] = (*native_ct)->GetLevel();
            rr["noise_scale_deg"] = (*native_ct)->GetNoiseScaleDeg();
            rr["scaling_factor"] = (*native_ct)->GetScalingFactor();
            rr["slots"] = (*native_ct)->GetSlots();
            rr["encoding_type"] = static_cast<int>((*native_ct)->GetEncodingType());
            rr["key_tag"] = (*native_ct)->GetKeyTag();

            if (r == 0) {
                expected_parts = elems.size();
            } else if (elems.size() != expected_parts) {
                consistent = false;
            }

            py::list parts;

            for (std::size_t part_idx = 0; part_idx < elems.size(); ++part_idx) {
                auto elem = elems[part_idx];
                elem.SetFormat(COEFFICIENT);

                const auto& towers = elem.GetAllElements();

                if (r == 0 && part_idx == 0) {
                    expected_towers = towers.size();
                    expected_ring_dim = elem.GetLength();
                } else {
                    if (towers.size() != expected_towers || elem.GetLength() != expected_ring_dim) {
                        consistent = false;
                    }
                }

                py::dict pr;
                pr["part_index"] = part_idx;
                pr["num_towers"] = towers.size();
                pr["ring_dim"] = elem.GetLength();

                py::list tower_reports;

                for (std::size_t t = 0; t < towers.size(); ++t) {
                    const auto& tower = towers[t];
                    const auto& vals = tower.GetValues();

                    py::dict tr;
                    tr["tower_index"] = t;
                    tr["modulus_u64"] = tower.GetModulus().ConvertToInt();
                    tr["length"] = tower.GetLength();

                    py::list head;
                    std::size_t n = vals.GetLength();
                    std::size_t take = coeff_sample < n ? coeff_sample : n;

                    for (std::size_t i = 0; i < take; ++i) {
                        head.append(vals[i].ConvertToInt());
                    }

                    tr["coeff_head_u64"] = head;
                    tower_reports.append(tr);
                }

                pr["towers"] = tower_reports;
                parts.append(pr);
            }

            rr["parts"] = parts;
            row_reports.append(rr);
        }

        report["consistent_shape"] = consistent;
        report["expected_parts"] = expected_parts;
        report["expected_towers"] = expected_towers;
        report["expected_ring_dim"] = expected_ring_dim;
        report["rows"] = row_reports;

        // Paper mapping convention for row-wise bundle:
        //   A[row, :] = component part 1 / c1
        //   B[row, :] = component part 0 / c0
        // because OpenFHE ciphertext decrypt convention is c0 + c1*s.
        report["paper_mapping"] = "rowwise: B = GetElements()[0] / c0, A = GetElements()[1] / c1";

        return report;
    }



    CiphertextHandle roundtrip_rlwe_components_cpu(const CiphertextHandle& h) {
        require_ready();
        h.require_valid();

        if (!h.ct_->cpu.has_value()) {
            throw std::runtime_error("roundtrip_rlwe_components_cpu(): ciphertext has no CPU object in ct_->cpu. Use ciphertext_autoload=False for this debug probe.");
        }

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        const OpenFHECiphertext* ct_ptr = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);
        if (ct_ptr == nullptr || !(*ct_ptr)) {
            throw std::runtime_error("roundtrip_rlwe_components_cpu(): ct_->cpu is not a valid lbcrypto::Ciphertext<lbcrypto::DCRTPoly>");
        }

        const auto& ct_src = *ct_ptr;
        const auto& elems_src = ct_src->GetElements();

        if (elems_src.size() < 2) {
            throw std::runtime_error("roundtrip_rlwe_components_cpu(): expected at least two RLWE components");
        }

        // Deep-copy the OpenFHE ciphertext metadata via copy constructor.
        auto ct_dst = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*ct_src);

        // Explicitly reconstruct elements from copied component polynomials.
        std::vector<lbcrypto::DCRTPoly> elems_dst;
        elems_dst.reserve(elems_src.size());

        for (std::size_t i = 0; i < elems_src.size(); ++i) {
            lbcrypto::DCRTPoly poly = elems_src[i];
            elems_dst.push_back(poly);
        }

        ct_dst->SetElements(elems_dst);
        ct_dst->SetLevel(ct_src->GetLevel());
        ct_dst->SetScalingFactor(ct_src->GetScalingFactor());
        ct_dst->SetNoiseScaleDeg(ct_src->GetNoiseScaleDeg());
        ct_dst->SetKeyTag(ct_src->GetKeyTag());
        ct_dst->SetSlots(ct_src->GetSlots());

        // Wrap back into a FIDESlib ciphertext handle.
        // Clone preserves parent_context and FIDESlib-side metadata.
        auto out = h.ct_->Clone();
        out->cpu = ct_dst;
        out->gpu = 0;
        out->loaded = false;
        out->SetLevel(h.ct_->GetLevel());

        return CiphertextHandle(out);
    }


    static std::uint64_t mod_mul_signed_i64(std::int64_t w, std::uint64_t x, std::uint64_t mod) {
        if (mod == 0) {
            throw std::runtime_error("mod_mul_signed_i64(): modulus is zero");
        }

        bool neg = (w < 0);
        std::uint64_t absw = 0;

        if (w == std::numeric_limits<std::int64_t>::min()) {
            absw = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1ULL;
        } else {
            absw = static_cast<std::uint64_t>(neg ? -w : w);
        }

        absw %= mod;
        x %= mod;

        const std::uint64_t prod = static_cast<std::uint64_t>(
            (static_cast<__uint128_t>(absw) * static_cast<__uint128_t>(x)) % static_cast<__uint128_t>(mod)
        );

        if (!neg || prod == 0) {
            return prod;
        }

        return mod - prod;
    }

    static std::uint64_t mod_add_u64(std::uint64_t a, std::uint64_t b, std::uint64_t mod) {
        return static_cast<std::uint64_t>(
            (static_cast<__uint128_t>(a) + static_cast<__uint128_t>(b)) % static_cast<__uint128_t>(mod)
        );
    }

    std::vector<CiphertextHandle> component_int_linear_combination_cpu(
        const std::vector<CiphertextHandle>& rows,
        const std::vector<std::vector<std::int64_t>>& weights
    ) {
        require_ready();

        if (rows.empty()) {
            throw std::runtime_error("component_int_linear_combination_cpu(): rows is empty");
        }
        if (weights.empty()) {
            throw std::runtime_error("component_int_linear_combination_cpu(): weights is empty");
        }

        const std::size_t n_in = rows.size();
        const std::size_t n_out = weights.size();

        for (std::size_t r = 0; r < n_out; ++r) {
            if (weights[r].size() != n_in) {
                throw std::runtime_error("component_int_linear_combination_cpu(): each weight row must have len(rows)");
            }
        }

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        std::vector<const OpenFHECiphertext*> ct_ptrs;
        ct_ptrs.reserve(n_in);

        for (std::size_t i = 0; i < n_in; ++i) {
            rows[i].require_valid();

            if (!rows[i].ct_->cpu.has_value()) {
                throw std::runtime_error("component_int_linear_combination_cpu(): one input row has no CPU object. Use ciphertext_autoload=False.");
            }

            const OpenFHECiphertext* p = std::any_cast<OpenFHECiphertext>(&rows[i].ct_->cpu);
            if (p == nullptr || !(*p)) {
                throw std::runtime_error("component_int_linear_combination_cpu(): ct_->cpu is not a valid OpenFHE ciphertext");
            }

            ct_ptrs.push_back(p);
        }

        const auto& ct0 = *ct_ptrs[0];
        const auto& elems0 = ct0->GetElements();

        if (elems0.size() < 2) {
            throw std::runtime_error("component_int_linear_combination_cpu(): expected at least 2 RLWE components");
        }

        const std::size_t num_parts = elems0.size();

        // Basic compatibility checks.
        for (std::size_t i = 1; i < n_in; ++i) {
            const auto& cti = *ct_ptrs[i];
            const auto& elemsi = cti->GetElements();

            if (elemsi.size() != num_parts) {
                throw std::runtime_error("component_int_linear_combination_cpu(): incompatible num_parts");
            }
            if (cti->GetLevel() != ct0->GetLevel()) {
                throw std::runtime_error("component_int_linear_combination_cpu(): incompatible OpenFHE levels");
            }
            if (cti->GetNoiseScaleDeg() != ct0->GetNoiseScaleDeg()) {
                throw std::runtime_error("component_int_linear_combination_cpu(): incompatible noise scale degree");
            }
            if (cti->GetSlots() != ct0->GetSlots()) {
                throw std::runtime_error("component_int_linear_combination_cpu(): incompatible slots");
            }
        }

        std::vector<CiphertextHandle> outputs;
        outputs.reserve(n_out);

        for (std::size_t r = 0; r < n_out; ++r) {
            // Copy metadata from ct0.
            auto ct_dst = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*ct0);

            std::vector<lbcrypto::DCRTPoly> elems_dst;
            elems_dst.reserve(num_parts);

            for (std::size_t part_idx = 0; part_idx < num_parts; ++part_idx) {
                lbcrypto::DCRTPoly poly_out = elems0[part_idx];

                auto& out_towers = poly_out.GetAllElements();
                const std::size_t num_towers = out_towers.size();

                for (std::size_t tower_idx = 0; tower_idx < num_towers; ++tower_idx) {
                    auto& out_tower = out_towers[tower_idx];
                    auto& out_values = *out_tower.m_values;

                    const std::uint64_t mod = static_cast<std::uint64_t>(out_tower.GetModulus().ConvertToInt());
                    const std::size_t ring_dim = out_values.GetLength();

                    for (std::size_t k = 0; k < ring_dim; ++k) {
                        std::uint64_t acc = 0;

                        for (std::size_t i = 0; i < n_in; ++i) {
                            const std::int64_t w = weights[r][i];
                            if (w == 0) {
                                continue;
                            }

                            const auto& in_poly = (*ct_ptrs[i])->GetElements()[part_idx];
                            const auto& in_tower = in_poly.GetAllElements()[tower_idx];
                            const auto& in_values = in_tower.GetValues();

                            const std::uint64_t x = static_cast<std::uint64_t>(in_values[k].ConvertToInt());
                            const std::uint64_t term = mod_mul_signed_i64(w, x, mod);
                            acc = mod_add_u64(acc, term, mod);
                        }

                        out_values.at(k).SetValue(acc);
                    }
                }

                elems_dst.push_back(poly_out);
            }

            ct_dst->SetElements(elems_dst);
            ct_dst->SetLevel(ct0->GetLevel());
            ct_dst->SetScalingFactor(ct0->GetScalingFactor());
            ct_dst->SetNoiseScaleDeg(ct0->GetNoiseScaleDeg());
            ct_dst->SetKeyTag(ct0->GetKeyTag());
            ct_dst->SetSlots(ct0->GetSlots());

            auto out = rows[0].ct_->Clone();
            out->cpu = ct_dst;
            out->gpu = 0;
            out->loaded = false;
            out->SetLevel(rows[0].ct_->GetLevel());

            outputs.emplace_back(out);
        }

        return outputs;
    }




    py::dict ciphertext_storage_state(const CiphertextHandle& h) {
        h.require_valid();

        py::dict out;
        out["valid"] = true;
        out["loaded"] = h.ct_->loaded;
        out["gpu"] = h.ct_->gpu;
        out["has_cpu"] = h.ct_->cpu.has_value();
        out["level"] = h.ct_->GetLevel();
        out["noise_scale_deg"] = h.ct_->GetNoiseScaleDeg();
        out["original_level"] = h.ct_->original_level;
        return out;
    }






    CiphertextHandle component_linear_wsum_gpu(
        const std::vector<CiphertextHandle>& rows,
        const std::vector<double>& weights
    ) {
        require_ready();

        if (rows.empty()) {
            throw std::runtime_error("component_linear_wsum_gpu(): rows is empty");
        }
        if (weights.size() != rows.size()) {
            throw std::runtime_error("component_linear_wsum_gpu(): weights.size() must equal rows.size()");
        }

        auto parent = rows[0].ct_->parent_context;
        if (!parent) {
            throw std::runtime_error("component_linear_wsum_gpu(): first row has null parent_context");
        }

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        if (!rows[0].ct_->cpu.has_value()) {
            throw std::runtime_error("component_linear_wsum_gpu(): rows[0] has no CPU OpenFHE template ciphertext");
        }

        const OpenFHECiphertext* cpu_template_ptr = std::any_cast<OpenFHECiphertext>(&rows[0].ct_->cpu);
        if (cpu_template_ptr == nullptr || !(*cpu_template_ptr)) {
            throw std::runtime_error("component_linear_wsum_gpu(): rows[0].ct_->cpu is not a valid OpenFHE ciphertext");
        }

        const OpenFHECiphertext& cpu_template = *cpu_template_ptr;

        auto ensure_loaded = [&](const CiphertextHandle& h) {
            h.require_valid();

            if (h.ct_->parent_context != parent) {
                throw std::runtime_error("component_linear_wsum_gpu(): all rows must share the same parent_context");
            }

            if (!h.ct_->loaded || h.ct_->gpu == 0) {
                h.ct_->gpu = parent->CopyDeviceCiphertext(*h.ct_);
                h.ct_->loaded = true;
            }

            auto& opaque = parent->GetDeviceCiphertext(h.ct_->gpu);
            auto gpu_ct = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(opaque);

            if (!gpu_ct) {
                throw std::runtime_error("component_linear_wsum_gpu(): failed to cast device ciphertext handle");
            }

            return gpu_ct;
        };

        std::vector<std::shared_ptr<FIDESlib::CKKS::Ciphertext>> gpu_inputs;
        gpu_inputs.reserve(rows.size());

        for (std::size_t i = 0; i < rows.size(); ++i) {
            gpu_inputs.push_back(ensure_loaded(rows[i]));
        }

        int first_idx = -1;
        std::vector<std::int64_t> int_weights;
        int_weights.reserve(weights.size());

        for (std::size_t i = 0; i < weights.size(); ++i) {
            const double w = weights[i];

            if (!std::isfinite(w)) {
                throw std::runtime_error("component_linear_wsum_gpu(): non-finite weight");
            }

            const double rounded = std::round(w);
            if (std::fabs(w - rounded) > 1e-9) {
                throw std::runtime_error("component_linear_wsum_gpu(): repeated-add/sub path supports integer weights only");
            }

            if (rounded > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
                rounded < static_cast<double>(std::numeric_limits<std::int64_t>::min() + 1)) {
                throw std::runtime_error("component_linear_wsum_gpu(): weight out of int64 range");
            }

            const std::int64_t k = static_cast<std::int64_t>(rounded);
            int_weights.push_back(k);

            if (k != 0 && first_idx < 0) {
                first_idx = static_cast<int>(i);
            }
        }

        if (first_idx < 0) {
            throw std::runtime_error("component_linear_wsum_gpu(): all-zero weights are not supported in this GPU repeated-add/sub path");
        }

        auto first_row = rows[static_cast<std::size_t>(first_idx)].ct_;
        const std::uint32_t out_handle = parent->CopyDeviceCiphertext(*first_row);

        auto out_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(
            parent->GetDeviceCiphertext(out_handle)
        );

        if (!out_gpu) {
            throw std::runtime_error("component_linear_wsum_gpu(): failed to allocate output GPU ciphertext");
        }

        const std::int64_t first_weight = int_weights[static_cast<std::size_t>(first_idx)];
        const std::uint64_t first_abs =
            static_cast<std::uint64_t>(first_weight < 0 ? -first_weight : first_weight);

        if (first_weight < 0) {
            out_gpu->multScalar(-1.0);
        }

        for (std::uint64_t t = 1; t < first_abs; ++t) {
            if (first_weight > 0) {
                out_gpu->add(*gpu_inputs[static_cast<std::size_t>(first_idx)]);
            } else {
                out_gpu->sub(*gpu_inputs[static_cast<std::size_t>(first_idx)]);
            }
        }

        for (std::size_t i = static_cast<std::size_t>(first_idx) + 1; i < rows.size(); ++i) {
            const std::int64_t k = int_weights[i];

            if (k > 0) {
                for (std::int64_t t = 0; t < k; ++t) {
                    out_gpu->add(*gpu_inputs[i]);
                }
            } else if (k < 0) {
                for (std::int64_t t = 0; t < -k; ++t) {
                    out_gpu->sub(*gpu_inputs[i]);
                }
            }
        }

        parent->Synchronize();

        FIDESlib::CKKS::RawCipherText raw_out;
        out_gpu->store(raw_out);

        raw_out.keyid = out_gpu->keyID;
        raw_out.slots = cpu_template->GetSlots();
        raw_out.Noise = cpu_template->GetScalingFactor();
        raw_out.NoiseLevel = out_gpu->NoiseLevel;

        OpenFHECiphertext cpu_out =
            std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*cpu_template);

        FIDESlib::CKKS::GetOpenFHECipherText(cpu_out, raw_out, 1);

        auto out = rows[0].ct_->Clone();
        out->cpu = std::make_any<OpenFHECiphertext>(std::move(cpu_out));
        out->gpu = out_handle;
        out->loaded = true;
        out->parent_context = parent;
        out->original_level = out_gpu->getLevel();
        out->SetLevel(rows[0].ct_->GetLevel());

        return CiphertextHandle(out);
    }



    CiphertextHandle component_linear_wsum_gpu_fused_raw(
        const std::vector<CiphertextHandle>& rows,
        const std::vector<double>& weights
    ) {
        require_ready();

        if (rows.empty()) {
            throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): rows is empty");
        }
        if (weights.size() != rows.size()) {
            throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): weights.size() must equal rows.size()");
        }

        auto parent = rows[0].ct_->parent_context;
        if (!parent) {
            throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): first row has null parent_context");
        }

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        if (!rows[0].ct_->cpu.has_value()) {
            throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): rows[0] has no CPU OpenFHE template ciphertext");
        }

        const OpenFHECiphertext* cpu_template_ptr = std::any_cast<OpenFHECiphertext>(&rows[0].ct_->cpu);
        if (cpu_template_ptr == nullptr || !(*cpu_template_ptr)) {
            throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): rows[0].ct_->cpu is not a valid OpenFHE ciphertext");
        }

        const OpenFHECiphertext& cpu_template = *cpu_template_ptr;

        std::vector<std::shared_ptr<FIDESlib::CKKS::Ciphertext>> gpu_shared;
        std::vector<const FIDESlib::CKKS::RNSPoly*> c0s;
        std::vector<const FIDESlib::CKKS::RNSPoly*> c1s;
        std::vector<std::int64_t> int_weights;

        gpu_shared.reserve(rows.size());
        c0s.reserve(rows.size());
        c1s.reserve(rows.size());
        int_weights.reserve(rows.size());

        for (std::size_t i = 0; i < rows.size(); ++i) {
            rows[i].require_valid();

            if (rows[i].ct_->parent_context != parent) {
                throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): all rows must share the same parent_context");
            }

            if (!rows[i].ct_->loaded || rows[i].ct_->gpu == 0) {
                rows[i].ct_->gpu = parent->CopyDeviceCiphertext(*rows[i].ct_);
                rows[i].ct_->loaded = true;
            }

            auto& opaque = parent->GetDeviceCiphertext(rows[i].ct_->gpu);
            auto gpu_ct = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(opaque);
            if (!gpu_ct) {
                throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): failed to cast device ciphertext handle");
            }

            if (i > 0) {
                if (gpu_ct->keyID != gpu_shared[0]->keyID) {
                    throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): keyID mismatch");
                }
                if (gpu_ct->getLevel() != gpu_shared[0]->getLevel()) {
                    throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): level mismatch");
                }
                if (gpu_ct->NoiseLevel != gpu_shared[0]->NoiseLevel) {
                    throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): NoiseLevel mismatch");
                }
            }

            const double w = weights[i];
            if (!std::isfinite(w)) {
                throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): non-finite weight");
            }

            const double rounded = std::round(w);
            if (std::fabs(w - rounded) > 1e-9) {
                throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): raw fused path supports integer weights only");
            }

            if (rounded > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
                rounded < static_cast<double>(std::numeric_limits<std::int64_t>::min() + 1)) {
                throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): weight out of int64 range");
            }

            gpu_shared.push_back(gpu_ct);
            c0s.push_back(&gpu_ct->c0);
            c1s.push_back(&gpu_ct->c1);
            int_weights.push_back(static_cast<std::int64_t>(rounded));
        }

        const int level = gpu_shared[0]->getLevel();
        const std::size_t active_towers = static_cast<std::size_t>(level + 1);
        const auto& cpu_towers = cpu_template->GetElements()[0].GetAllElements();

        if (cpu_towers.size() < active_towers) {
            throw std::runtime_error("component_linear_wsum_gpu_fused_raw(): CPU template has fewer towers than GPU ciphertext level requires");
        }

        // Important:
        // FIDESlib eval_linear_w_sum_ indexes weights by:
        //   w[primeid] for input 0
        //   w[i * FIDESlib::MAXP + primeid] for input i
        // Therefore weight stride must be MAXP, not active_towers.
        std::vector<std::uint64_t> elem(rows.size() * FIDESlib::MAXP, 0);

        for (std::size_t i = 0; i < rows.size(); ++i) {
            const std::int64_t wi = int_weights[i];

            for (std::size_t primeid = 0; primeid < active_towers; ++primeid) {
                const std::uint64_t mod =
                    static_cast<std::uint64_t>(cpu_towers[primeid].GetModulus().ConvertToInt());

                std::uint64_t residue = 0;
                if (wi >= 0) {
                    residue = static_cast<std::uint64_t>(wi) % mod;
                } else {
                    const std::uint64_t abs_w = static_cast<std::uint64_t>(-wi);
                    const std::uint64_t r = abs_w % mod;
                    residue = (r == 0) ? 0 : (mod - r);
                }

                elem[i * FIDESlib::MAXP + primeid] = residue;
            }
        }

        auto out_gpu = std::make_shared<FIDESlib::CKKS::Ciphertext>(gpu_shared[0]->cc_);

        out_gpu->c0.grow(level);
        out_gpu->c1.grow(level);

        out_gpu->c0.evalLinearWSum(
            static_cast<std::uint32_t>(c0s.size()),
            c0s,
            elem
        );
        out_gpu->c1.evalLinearWSum(
            static_cast<std::uint32_t>(c1s.size()),
            c1s,
            elem
        );

        out_gpu->copyMetadata(*gpu_shared[0]);

        parent->Synchronize();

        FIDESlib::CKKS::RawCipherText raw_out;
        out_gpu->store(raw_out);

        raw_out.keyid = out_gpu->keyID;
        raw_out.slots = cpu_template->GetSlots();
        raw_out.Noise = cpu_template->GetScalingFactor();
        raw_out.NoiseLevel = out_gpu->NoiseLevel;

        OpenFHECiphertext cpu_out =
            std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*cpu_template);

        // WP4-E3 showed REV=1 is correct for GPU -> CPU copyback.
        FIDESlib::CKKS::GetOpenFHECipherText(cpu_out, raw_out, 1);

        auto out = rows[0].ct_->Clone();
        out->cpu = std::make_any<OpenFHECiphertext>(std::move(cpu_out));

        std::shared_ptr<void> out_opaque = std::static_pointer_cast<void>(out_gpu);
        out->gpu = parent->RegisterDeviceCiphertext(std::move(out_opaque));
        out->loaded = true;
        out->parent_context = parent;
        out->original_level = out_gpu->getLevel();
        out->SetLevel(rows[0].ct_->GetLevel());

        return CiphertextHandle(out);
    }




    std::vector<CiphertextHandle> component_linear_matmul_gpu_fused_raw(
        const std::vector<CiphertextHandle>& rows,
        const std::vector<std::vector<double>>& U,
        bool copyback
    ) {
        require_ready();

        if (rows.empty()) {
            throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): rows is empty");
        }
        if (U.empty()) {
            throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): U is empty");
        }

        const std::size_t n_in = rows.size();
        const std::size_t n_out = U.size();

        for (std::size_t r = 0; r < n_out; ++r) {
            if (U[r].size() != n_in) {
                throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): every U row must have length rows.size()");
            }
        }

        auto parent = rows[0].ct_->parent_context;
        if (!parent) {
            throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): first row has null parent_context");
        }

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        // Function-local cache for GPU-only chained CCMM.
        // First CPU-backed call initializes it; later GPU-only inputs can reuse it.
        static std::optional<OpenFHECiphertext> g_gpu_chain_cpu_template;
        static std::vector<std::uint64_t> g_gpu_chain_moduli;
        static std::size_t g_gpu_chain_slots = 0;


        OpenFHECiphertext cpu_template;
        bool have_cpu_template = false;

        // Prefer a real CPU template from any input row.
        for (std::size_t i = 0; i < n_in; ++i) {
            if (!rows[i].ct_) {
                continue;
            }

            if (rows[i].ct_->cpu.has_value()) {
                const OpenFHECiphertext* ptr = std::any_cast<OpenFHECiphertext>(&rows[i].ct_->cpu);
                if (ptr != nullptr && (*ptr)) {
                    cpu_template = *ptr;
                    have_cpu_template = true;
                    break;
                }
            }
        }

        // If all inputs are GPU-only, reuse cached template from an earlier CPU-backed call.
        if (!have_cpu_template && g_gpu_chain_cpu_template.has_value()) {
            cpu_template = *g_gpu_chain_cpu_template;
            have_cpu_template = true;
        }

        if (!have_cpu_template) {
            throw std::runtime_error(
                "component_linear_matmul_gpu_fused_raw(): no CPU template available. "
                "Call once with CPU-backed input before chaining GPU-only outputs."
            );
        }

        // Refresh metadata cache whenever we have a usable template.
        g_gpu_chain_cpu_template = cpu_template;
        g_gpu_chain_slots = cpu_template->GetSlots();

        const auto& cpu_towers_all = cpu_template->GetElements()[0].GetAllElements();
        g_gpu_chain_moduli.clear();
        g_gpu_chain_moduli.reserve(cpu_towers_all.size());

        for (std::size_t j = 0; j < cpu_towers_all.size(); ++j) {
            g_gpu_chain_moduli.push_back(
                static_cast<std::uint64_t>(cpu_towers_all[j].GetModulus().ConvertToInt())
            );
        }

        std::vector<std::shared_ptr<FIDESlib::CKKS::Ciphertext>> gpu_shared;
        std::vector<const FIDESlib::CKKS::RNSPoly*> c0s;
        std::vector<const FIDESlib::CKKS::RNSPoly*> c1s;

        gpu_shared.reserve(n_in);
        c0s.reserve(n_in);
        c1s.reserve(n_in);

        for (std::size_t i = 0; i < n_in; ++i) {
            rows[i].require_valid();

            if (rows[i].ct_->parent_context != parent) {
                throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): all rows must share the same parent_context");
            }

            if (!rows[i].ct_->loaded || rows[i].ct_->gpu == 0) {
                if (!rows[i].ct_->cpu.has_value()) {
                    throw std::runtime_error(
                        "component_linear_matmul_gpu_fused_raw(): input is neither loaded on GPU nor available on CPU"
                    );
                }
                rows[i].ct_->gpu = parent->CopyDeviceCiphertext(*rows[i].ct_);
                rows[i].ct_->loaded = true;
            }

            auto& opaque = parent->GetDeviceCiphertext(rows[i].ct_->gpu);
            auto gpu_ct = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(opaque);

            if (!gpu_ct) {
                throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): failed to cast device ciphertext handle");
            }

            if (i > 0) {
                if (gpu_ct->keyID != gpu_shared[0]->keyID) {
                    throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): keyID mismatch");
                }
                if (gpu_ct->getLevel() != gpu_shared[0]->getLevel()) {
                    throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): level mismatch");
                }
                if (gpu_ct->NoiseLevel != gpu_shared[0]->NoiseLevel) {
                    throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): NoiseLevel mismatch");
                }
            }

            gpu_shared.push_back(gpu_ct);
            c0s.push_back(&gpu_ct->c0);
            c1s.push_back(&gpu_ct->c1);
        }

        const int level = gpu_shared[0]->getLevel();
        const std::size_t active_towers = static_cast<std::size_t>(level + 1);

        if (g_gpu_chain_moduli.size() < active_towers) {
            throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): cached modulus list is too short for active level");
        }

        auto build_elem = [&](const std::vector<double>& weights) {
            std::vector<std::uint64_t> elem(n_in * FIDESlib::MAXP, 0);

            for (std::size_t i = 0; i < n_in; ++i) {
                const double w = weights[i];

                if (!std::isfinite(w)) {
                    throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): non-finite weight");
                }

                const double rounded = std::round(w);
                if (std::fabs(w - rounded) > 1e-9) {
                    throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): fused raw path supports integer weights only");
                }

                if (rounded > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
                    rounded < static_cast<double>(std::numeric_limits<std::int64_t>::min() + 1)) {
                    throw std::runtime_error("component_linear_matmul_gpu_fused_raw(): weight out of int64 range");
                }

                const std::int64_t wi = static_cast<std::int64_t>(rounded);

                for (std::size_t primeid = 0; primeid < active_towers; ++primeid) {
                    const std::uint64_t mod = g_gpu_chain_moduli[primeid];

                    std::uint64_t residue = 0;
                    if (wi >= 0) {
                        residue = static_cast<std::uint64_t>(wi) % mod;
                    } else {
                        const std::uint64_t abs_w = static_cast<std::uint64_t>(-wi);
                        const std::uint64_t r = abs_w % mod;
                        residue = (r == 0) ? 0 : (mod - r);
                    }

                    elem[i * FIDESlib::MAXP + primeid] = residue;
                }
            }

            return elem;
        };

        std::vector<std::shared_ptr<FIDESlib::CKKS::Ciphertext>> out_gpus;
        out_gpus.reserve(n_out);

        for (std::size_t r = 0; r < n_out; ++r) {
            auto out_gpu = std::make_shared<FIDESlib::CKKS::Ciphertext>(gpu_shared[0]->cc_);

            out_gpu->c0.grow(level);
            out_gpu->c1.grow(level);

            std::vector<std::uint64_t> elem = build_elem(U[r]);

            out_gpu->c0.evalLinearWSum(
                static_cast<std::uint32_t>(c0s.size()),
                c0s,
                elem
            );

            out_gpu->c1.evalLinearWSum(
                static_cast<std::uint32_t>(c1s.size()),
                c1s,
                elem
            );

            out_gpu->copyMetadata(*gpu_shared[0]);
            out_gpus.push_back(out_gpu);
        }

        parent->Synchronize();

        std::vector<CiphertextHandle> outputs;
        outputs.reserve(n_out);

        for (std::size_t r = 0; r < n_out; ++r) {
            auto& out_gpu = out_gpus[r];
            auto out = rows[0].ct_->Clone();

            if (copyback) {
                FIDESlib::CKKS::RawCipherText raw_out;
                out_gpu->store(raw_out);

                raw_out.keyid = out_gpu->keyID;
                raw_out.slots = g_gpu_chain_slots;
                raw_out.Noise = cpu_template->GetScalingFactor();
                raw_out.NoiseLevel = out_gpu->NoiseLevel;

                OpenFHECiphertext cpu_out =
                    std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*cpu_template);

                FIDESlib::CKKS::GetOpenFHECipherText(cpu_out, raw_out, 1);

                out->cpu = std::make_any<OpenFHECiphertext>(std::move(cpu_out));
            } else {
                // GPU-only intermediate:
                // keep CPU-side OpenFHE metadata/template so the next layer can
                // std::any_cast<OpenFHECiphertext>(&ct_->cpu) and recover slots,
                // scale, level, key tag, and modulus structure.
                //
                // Numerical ciphertext data is NOT copied back here; it remains
                // in out_gpu and is registered below via out->gpu.
                OpenFHECiphertext cpu_meta =
                    std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*cpu_template);

                cpu_meta->SetLevel(cpu_template->GetLevel());
                cpu_meta->SetScalingFactor(cpu_template->GetScalingFactor());
                cpu_meta->SetNoiseScaleDeg(cpu_template->GetNoiseScaleDeg());
                cpu_meta->SetKeyTag(cpu_template->GetKeyTag());
                cpu_meta->SetSlots(cpu_template->GetSlots());

                out->cpu = std::make_any<OpenFHECiphertext>(std::move(cpu_meta));
            }

            std::shared_ptr<void> out_opaque = std::static_pointer_cast<void>(out_gpu);
            out->gpu = parent->RegisterDeviceCiphertext(std::move(out_opaque));
            out->loaded = true;
            out->parent_context = parent;
            out->original_level = out_gpu->getLevel();
            out->SetLevel(rows[0].ct_->GetLevel());

            outputs.emplace_back(out);
        }

        return outputs;
    }


    CiphertextHandle gpu_copyback_cpu_debug(const CiphertextHandle& h, int rev) {
        require_ready();
        h.require_valid();

        if (!h.ct_->parent_context) {
            throw std::runtime_error("gpu_copyback_cpu_debug(): null parent_context");
        }

        auto parent = h.ct_->parent_context;

        if (!h.ct_->loaded || h.ct_->gpu == 0) {
            h.ct_->gpu = parent->CopyDeviceCiphertext(*h.ct_);
            h.ct_->loaded = true;
        }

        if (!h.ct_->cpu.has_value()) {
            throw std::runtime_error("gpu_copyback_cpu_debug(): input has no CPU OpenFHE template");
        }

        using OpenFHECiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

        const OpenFHECiphertext* cpu_template_ptr = std::any_cast<OpenFHECiphertext>(&h.ct_->cpu);
        if (cpu_template_ptr == nullptr || !(*cpu_template_ptr)) {
            throw std::runtime_error("gpu_copyback_cpu_debug(): input CPU object is not valid OpenFHE ciphertext");
        }

        const OpenFHECiphertext& cpu_template = *cpu_template_ptr;

        auto& opaque = parent->GetDeviceCiphertext(h.ct_->gpu);
        auto gpu_ct = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(opaque);
        if (!gpu_ct) {
            throw std::runtime_error("gpu_copyback_cpu_debug(): failed to cast GPU ciphertext");
        }

        parent->Synchronize();

        FIDESlib::CKKS::RawCipherText raw;
        gpu_ct->store(raw);

        raw.keyid = gpu_ct->keyID;
        raw.slots = cpu_template->GetSlots();
        raw.Noise = cpu_template->GetScalingFactor();
        raw.NoiseLevel = gpu_ct->NoiseLevel;

        OpenFHECiphertext cpu_out =
            std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*cpu_template);

        FIDESlib::CKKS::GetOpenFHECipherText(cpu_out, raw, rev);

        auto out = h.ct_->Clone();

        out->cpu = std::make_any<OpenFHECiphertext>(std::move(cpu_out));

        // CPU-only copyback debug object.
        // Do not reuse h.ct_->gpu here. Clone() may have copied the original handle,
        // and sharing/evicting that handle can corrupt the registry.
        out->gpu = 0;
        out->loaded = false;
        out->parent_context = parent;
        out->original_level = gpu_ct->getLevel();
        out->SetLevel(h.ct_->GetLevel());

        return CiphertextHandle(out);
    }

    CiphertextHandle materialize_gpu_ciphertext(const CiphertextHandle& h) {
        // Formal API for GPU -> CPU/OpenFHE materialization.
        // REV=1 was validated in WP4-E/WP4-I as the correct GPU->CPU component order.
        return gpu_copyback_cpu_debug(h, 1);
    }

    std::vector<CiphertextHandle> materialize_gpu_ciphertexts(
        const std::vector<CiphertextHandle>& hs
    ) {
        std::vector<CiphertextHandle> out;
        out.reserve(hs.size());

        for (const auto& h : hs) {
            out.emplace_back(materialize_gpu_ciphertext(h));
        }

        return out;
    }


    std::vector<double> roundtrip(const std::vector<double>& x) {
        auto ct = encrypt(x);
        return decrypt(ct, x.size());
    }

    std::vector<double> eval_add(const std::vector<double>& x, const std::vector<double>& y) {
        require_ready();

        if (x.size() != y.size()) {
            throw std::runtime_error("eval_add(): x and y must have the same length");
        }

        auto ct_x = encrypt(x);
        auto ct_y = encrypt(y);
        auto ct_sum = eval_add_ct(ct_x, ct_y);
        return decrypt(ct_sum, x.size());
    }

    std::vector<double> eval_mult_scalar(const std::vector<double>& x, double scalar) {
        require_ready();

        auto ct_x = encrypt(x);
        auto ct_mul = eval_mult_scalar_ct(ct_x, scalar);
        return decrypt(ct_mul, x.size());
    }

    void close() {
        if (sk_) sk_.reset();
        if (pk_) pk_.reset();
        if (cc_) cc_.reset();
        (void)cudaDeviceSynchronize();
    }

private:
    void require_ready() const {
        if (!cc_ || !pk_ || !sk_) {
            throw std::runtime_error("FidesCKKSContext is not initialized");
        }
    }

    CryptoContext<DCRTPoly> cc_;
    PublicKey<DCRTPoly> pk_;
    PrivateKey<DCRTPoly> sk_;
};

PYBIND11_MODULE(_fideslib, m) {
    m.doc() = "pybind11 bindings for FIDESlib";

    py::class_<CiphertextHandle>(m, "CiphertextHandle")
        .def(py::init<>())
        .def("valid", &CiphertextHandle::valid)
        .def("level", &CiphertextHandle::level)
        .def("noise_scale_deg", &CiphertextHandle::noise_scale_deg);

    py::class_<FidesCKKSContext>(m, "FidesCKKSContext")
        .def(py::init<>())

        .def(
            "init",
            &FidesCKKSContext::init,
            py::arg("multiplicative_depth") = 2,
            py::arg("scaling_mod_size") = 50,
            py::arg("batch_size") = 8,
            py::arg("ring_dim") = (1u << 14),
            py::arg("first_mod_size") = -1,
            py::arg("num_large_digits") = -1,
            py::arg("devices") = std::vector<int>{0},
            py::arg("plaintext_autoload") = true,
            py::arg("ciphertext_autoload") = true,
            py::arg("with_mult_key") = true,
            py::arg("rotation_steps") = std::vector<int>{}
        )

        .def("info", &FidesCKKSContext::info)

        .def("encrypt", &FidesCKKSContext::encrypt, py::arg("x"))
        .def("decrypt", &FidesCKKSContext::decrypt,
             py::arg("ciphertext"),
             py::arg("logical_length") = 0)
        .def("eval_add_ct", &FidesCKKSContext::eval_add_ct,
             py::arg("a"), py::arg("b"))
        .def("eval_add_plain_ct", &FidesCKKSContext::eval_add_plain_ct,
             py::arg("a"), py::arg("plain"))
        .def("eval_mult_scalar_ct", &FidesCKKSContext::eval_mult_scalar_ct,
             py::arg("a"), py::arg("scalar"))
        .def("eval_mult_plain_ct", &FidesCKKSContext::eval_mult_plain_ct,
             py::arg("a"), py::arg("plain"))
        .def("eval_rotate_ct", &FidesCKKSContext::eval_rotate_ct,
             py::arg("a"), py::arg("steps"))

        .def("inspect_rlwe_components_cpu", &FidesCKKSContext::inspect_rlwe_components_cpu,
             py::arg("ciphertext"), py::arg("coeff_sample") = 8)
        .def("roundtrip_rlwe_components_cpu", &FidesCKKSContext::roundtrip_rlwe_components_cpu,
             py::arg("ciphertext"))
        .def("component_int_linear_combination_cpu", &FidesCKKSContext::component_int_linear_combination_cpu,
             py::arg("rows"), py::arg("weights"))
        .def("ciphertext_storage_state", &FidesCKKSContext::ciphertext_storage_state,
             py::arg("ciphertext"))
        .def("component_linear_wsum_gpu", &FidesCKKSContext::component_linear_wsum_gpu,
             py::arg("rows"), py::arg("weights"))
        .def("gpu_copyback_cpu_debug", &FidesCKKSContext::gpu_copyback_cpu_debug,
             py::arg("ciphertext"), py::arg("rev") = 0)
        .def("component_linear_wsum_gpu_fused_raw", &FidesCKKSContext::component_linear_wsum_gpu_fused_raw,
             py::arg("rows"), py::arg("weights"))
        .def("materialize_gpu_ciphertext", &FidesCKKSContext::materialize_gpu_ciphertext,
             py::arg("ciphertext"))
        .def("materialize_gpu_ciphertexts", &FidesCKKSContext::materialize_gpu_ciphertexts,
             py::arg("ciphertexts"))
        .def("component_linear_matmul_gpu_fused_raw", &FidesCKKSContext::component_linear_matmul_gpu_fused_raw,
             py::arg("rows"), py::arg("U"), py::arg("copyback") = true)
        .def("export_secret_key_coeff_u64", &FidesCKKSContext::export_secret_key_coeff_u64, py::arg("coeff_count") = 0)
        .def("encrypt_coeff_row_i64", &FidesCKKSContext::encrypt_coeff_row_i64,
             py::arg("coeffs"))
        .def("decrypt_coeff_row_i64", &FidesCKKSContext::decrypt_coeff_row_i64,
             py::arg("ciphertext"), py::arg("logical_length") = 0)
        .def("manual_decrypt_coeff_row_i64", &FidesCKKSContext::manual_decrypt_coeff_row_i64,
             py::arg("ciphertext"), py::arg("logical_length") = 0)
        .def("manual_decrypt_coeff_row_i64_variants", &FidesCKKSContext::manual_decrypt_coeff_row_i64_variants,
             py::arg("ciphertext"), py::arg("logical_length") = 0)
        .def("inspect_coeff_row_component_matrix", &FidesCKKSContext::inspect_coeff_row_component_matrix,
             py::arg("rows"), py::arg("coeff_sample") = 8)
        .def("eval_automorphism_coeff_ct", &FidesCKKSContext::eval_automorphism_coeff_ct,
             py::arg("ciphertext"), py::arg("alpha"))
        .def("add_coeff_ct", &FidesCKKSContext::add_coeff_ct,
             py::arg("a"), py::arg("b"))
        .def("mul_coeff_ct_no_relin", &FidesCKKSContext::mul_coeff_ct_no_relin,
             py::arg("a"), py::arg("b"))
        .def("component_mul_part_coeff_ct", &FidesCKKSContext::component_mul_part_coeff_ct,
             py::arg("a"), py::arg("part_a"), py::arg("b"), py::arg("part_b"))
        .def("export_component_coeff_matrix_u64", &FidesCKKSContext::export_component_coeff_matrix_u64,
             py::arg("rows"), py::arg("part"), py::arg("coeff_count") = 0)
        .def("import_1part_coeff_u64", &FidesCKKSContext::import_1part_coeff_u64,
             py::arg("template_ct"), py::arg("towers_coeffs"))
        .def("component_add_1part_coeff_ct", &FidesCKKSContext::component_add_1part_coeff_ct,
             py::arg("a"), py::arg("b"))
        .def("assemble_3part_from_1parts_coeff_ct", &FidesCKKSContext::assemble_3part_from_1parts_coeff_ct,
             py::arg("c0"), py::arg("c1"), py::arg("c2"))
        .def("assemble_2part_from_1parts_coeff_ct", &FidesCKKSContext::assemble_2part_from_1parts_coeff_ct,
             py::arg("c0"), py::arg("c1"))
        .def("relinearize_coeff_ct", &FidesCKKSContext::relinearize_coeff_ct,
             py::arg("ciphertext"))
        .def("compress_coeff_ct", &FidesCKKSContext::compress_coeff_ct,
             py::arg("ciphertext"), py::arg("towers_left") = 1)
        .def("monomial_mul_coeff_ct", &FidesCKKSContext::monomial_mul_coeff_ct,
             py::arg("ciphertext"), py::arg("exp"))
        .def("roundtrip", &FidesCKKSContext::roundtrip)
        .def("eval_add", &FidesCKKSContext::eval_add)
        .def("eval_mult_scalar", &FidesCKKSContext::eval_mult_scalar)

        .def("close", &FidesCKKSContext::close);
}
