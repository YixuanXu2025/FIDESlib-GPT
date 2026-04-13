#include "fideslib.hpp"
#include "cpp/native_helpers.hpp"

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace fideslib;
using namespace lbcrypto;

namespace {

ScalingTechnique parse_scaling_technique(const std::string& s) {
    if (s == "FIXEDMANUAL") return FIXEDMANUAL;
    if (s == "FIXEDAUTO") return FIXEDAUTO;
    if (s == "FLEXIBLEAUTO") return FLEXIBLEAUTO;
    if (s == "FLEXIBLEAUTOEXT") return FLEXIBLEAUTOEXT;
    if (s == "NORESCALE") return NORESCALE;
    throw std::runtime_error("Unknown scaling_technique: " + s);
}

KeySwitchTechnique parse_key_switch_technique(const std::string& s) {
    if (s == "BV") return BV;
    if (s == "HYBRID") return HYBRID;
    throw std::runtime_error("Unknown key_switch_technique: " + s);
}

SecurityLevel parse_security_level(const std::string& s) {
    if (s == "HEStd_NotSet") return HEStd_NotSet;
    if (s == "HEStd_128_classic") return HEStd_128_classic;
    if (s == "HEStd_192_classic") return HEStd_192_classic;
    if (s == "HEStd_256_classic") return HEStd_256_classic;
    if (s == "HEStd_128_quantum") return HEStd_128_quantum;
    if (s == "HEStd_192_quantum") return HEStd_192_quantum;
    if (s == "HEStd_256_quantum") return HEStd_256_quantum;
    throw std::runtime_error("Unknown security_level: " + s);
}

SecretKeyDist parse_secret_key_dist(const std::string& s) {
    if (s == "GAUSSIAN") return GAUSSIAN;
    if (s == "UNIFORM_TERNARY") return UNIFORM_TERNARY;
    if (s == "SPARSE_TERNARY") return SPARSE_TERNARY;
    throw std::runtime_error("Unknown secret_key_dist: " + s);
}

}  // namespace

class CiphertextHandle {
public:
    CiphertextHandle() = default;

    explicit CiphertextHandle(Ciphertext<DCRTPoly> ct)
        : ct_(std::move(ct)) {}

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
    friend class FidesCKKSContext;

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
        const std::string& scaling_technique = "",
        const std::string& key_switch_technique = "",
        const std::string& security_level = "",
        const std::string& secret_key_dist = "",
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
        if (!scaling_technique.empty()) {
            params.SetScalingTechnique(parse_scaling_technique(scaling_technique));
        }
        if (!key_switch_technique.empty()) {
            params.SetKeySwitchTechnique(parse_key_switch_technique(key_switch_technique));
        }
        if (!security_level.empty()) {
            params.SetSecurityLevel(parse_security_level(security_level));
        }
        if (!secret_key_dist.empty()) {
            params.SetSecretKeyDist(parse_secret_key_dist(secret_key_dist));
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

        // 必须在 LoadContext 之前
        if (with_mult_key) {
            cc_->EvalMultKeyGen(sk_);
        }

        // 旋转 key 也必须在 LoadContext 之前
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

    // ------------------------------------------------------------------
    // 基础句柄式接口
    // ------------------------------------------------------------------

    CiphertextHandle encrypt(const std::vector<double>& x) {
        require_ready();
        auto pt = cc_->MakeCKKSPackedPlaintext(x);
        auto ct = cc_->Encrypt(pt, pk_);
        pt.reset();
        return CiphertextHandle(ct);
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

    CiphertextHandle eval_rotate_ct(const CiphertextHandle& a, std::int32_t steps) {
        require_ready();
        a.require_valid();
        auto ct_out = cc_->EvalRotate(a.ct_, steps);
        return CiphertextHandle(ct_out);
    }

    // ------------------------------------------------------------------
    // block / PCMM / CCMM scaffold 接口
    // 说明：
    //   这一步先把 Python <-> C++ <-> helper 的接口形状固定下来。
    //   真正的 bert-tiny 协议迁移，后续再把 stub/reference 替换成真实实现。
    // ------------------------------------------------------------------

    py::dict block_layout_summary(
        std::uint32_t rows,
        std::uint32_t cols,
        std::uint32_t ring_dim = (1u << 14),
        std::uint32_t block_rows = 128,
        std::uint32_t block_cols = 128
    ) const {
        auto s = hegpt::summarize_block_layout(rows, cols, ring_dim, block_rows, block_cols);

        py::dict d;
        d["rows"] = s.rows;
        d["cols"] = s.cols;
        d["ring_dim"] = s.ring_dim;
        d["slots_per_ct"] = s.slots_per_ct;
        d["block_rows"] = s.block_rows;
        d["block_cols"] = s.block_cols;
        d["logical_block_elems"] = s.logical_block_elems;
        d["act_tile_rows"] = s.act_tile_rows;
        d["act_tile_cols"] = s.act_tile_cols;
        d["weight_tile_rows"] = s.weight_tile_rows;
        d["weight_tile_cols"] = s.weight_tile_cols;
        d["tiles_per_logical_block"] = s.tiles_per_logical_block;
        return d;
    }

    std::vector<double> extract_block_row_major(
        const std::vector<double>& matrix_flat,
        std::uint32_t rows,
        std::uint32_t cols,
        std::uint32_t row0,
        std::uint32_t col0,
        std::uint32_t block_rows = 128,
        std::uint32_t block_cols = 128,
        bool zero_pad = true
    ) const {
        return hegpt::extract_block_row_major(
            matrix_flat, rows, cols, row0, col0, block_rows, block_cols, zero_pad
        );
    }

    std::vector<std::vector<double>> split_activation_block_tiles(
        const std::vector<double>& block_flat,
        std::uint32_t block_rows = 128,
        std::uint32_t block_cols = 128,
        std::uint32_t tile_cols = 64
    ) const {
        return hegpt::split_activation_block_tiles(
            block_flat, block_rows, block_cols, tile_cols
        );
    }

    std::vector<std::vector<double>> split_weight_block_tiles(
        const std::vector<double>& block_flat,
        std::uint32_t block_rows = 128,
        std::uint32_t block_cols = 128,
        std::uint32_t tile_rows = 64
    ) const {
        return hegpt::split_weight_block_tiles(
            block_flat, block_rows, block_cols, tile_rows
        );
    }

    std::vector<double> pcmm_square_plain_reference(
        const std::vector<double>& x_block_flat,
        const std::vector<double>& w_block_flat,
        std::uint32_t block_rows = 128,
        std::uint32_t block_cols = 128,
        std::uint32_t split_dim = 64
    ) const {
        return hegpt::pcmm_square_plain_reference(
            x_block_flat, w_block_flat, block_rows, block_cols, split_dim
        );
    }

    std::vector<double> ccmm_square_plain_reference(
        const std::vector<double>& a_block_flat,
        const std::vector<double>& b_block_flat,
        std::uint32_t block_rows = 128,
        std::uint32_t block_cols = 128
    ) const {
        return hegpt::ccmm_square_plain_reference(
            a_block_flat, b_block_flat, block_rows, block_cols
        );
    }

    // 预留：后续真正接 bert-tiny 的 PCMMSquare / CCMMSquare 协议实现
    CiphertextHandle eval_pcmm_square_ct_plain_stub(
        const CiphertextHandle& x_block_ct,
        const std::vector<double>& w_block_flat,
        std::uint32_t block_rows = 128,
        std::uint32_t block_cols = 128
    ) {
        require_ready();
        x_block_ct.require_valid();
        (void)w_block_flat;
        (void)block_rows;
        (void)block_cols;
        throw std::runtime_error(
            "eval_pcmm_square_ct_plain_stub: protocol port not implemented yet"
        );
    }

    CiphertextHandle eval_ccmm_square_ct_ct_stub(
        const CiphertextHandle& a_block_ct,
        const CiphertextHandle& b_block_ct,
        std::uint32_t block_rows = 128,
        std::uint32_t block_cols = 128
    ) {
        require_ready();
        a_block_ct.require_valid();
        b_block_ct.require_valid();
        (void)block_rows;
        (void)block_cols;
        throw std::runtime_error(
            "eval_ccmm_square_ct_ct_stub: protocol port not implemented yet"
        );
    }

    // ------------------------------------------------------------------
    // 旧接口（保留兼容）
    // ------------------------------------------------------------------

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
    m.doc() = "pybind11 bindings for FIDESlib + hegpt native helper scaffolds";

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
            py::arg("scaling_technique") = "",
            py::arg("key_switch_technique") = "",
            py::arg("security_level") = "",
            py::arg("secret_key_dist") = "",
            py::arg("devices") = std::vector<int>{0},
            py::arg("plaintext_autoload") = true,
            py::arg("ciphertext_autoload") = true,
            py::arg("with_mult_key") = true,
            py::arg("rotation_steps") = std::vector<int>{}
        )

        .def("info", &FidesCKKSContext::info)

        // 基础句柄式接口
        .def("encrypt", &FidesCKKSContext::encrypt, py::arg("x"))
        .def("decrypt", &FidesCKKSContext::decrypt, py::arg("ciphertext"), py::arg("logical_length") = 0)
        .def("eval_add_ct", &FidesCKKSContext::eval_add_ct, py::arg("a"), py::arg("b"))
        .def("eval_add_plain_ct", &FidesCKKSContext::eval_add_plain_ct, py::arg("a"), py::arg("plain"))
        .def("eval_mult_scalar_ct", &FidesCKKSContext::eval_mult_scalar_ct, py::arg("a"), py::arg("scalar"))
        .def("eval_mult_plain_ct", &FidesCKKSContext::eval_mult_plain_ct, py::arg("a"), py::arg("plain"))
        .def("eval_rotate_ct", &FidesCKKSContext::eval_rotate_ct, py::arg("a"), py::arg("steps"))

        // block / PCMM / CCMM scaffold
        .def(
            "block_layout_summary",
            &FidesCKKSContext::block_layout_summary,
            py::arg("rows"),
            py::arg("cols"),
            py::arg("ring_dim") = (1u << 14),
            py::arg("block_rows") = 128,
            py::arg("block_cols") = 128
        )
        .def(
            "extract_block_row_major",
            &FidesCKKSContext::extract_block_row_major,
            py::arg("matrix_flat"),
            py::arg("rows"),
            py::arg("cols"),
            py::arg("row0"),
            py::arg("col0"),
            py::arg("block_rows") = 128,
            py::arg("block_cols") = 128,
            py::arg("zero_pad") = true
        )
        .def(
            "split_activation_block_tiles",
            &FidesCKKSContext::split_activation_block_tiles,
            py::arg("block_flat"),
            py::arg("block_rows") = 128,
            py::arg("block_cols") = 128,
            py::arg("tile_cols") = 64
        )
        .def(
            "split_weight_block_tiles",
            &FidesCKKSContext::split_weight_block_tiles,
            py::arg("block_flat"),
            py::arg("block_rows") = 128,
            py::arg("block_cols") = 128,
            py::arg("tile_rows") = 64
        )
        .def(
            "pcmm_square_plain_reference",
            &FidesCKKSContext::pcmm_square_plain_reference,
            py::arg("x_block_flat"),
            py::arg("w_block_flat"),
            py::arg("block_rows") = 128,
            py::arg("block_cols") = 128,
            py::arg("split_dim") = 64
        )
        .def(
            "ccmm_square_plain_reference",
            &FidesCKKSContext::ccmm_square_plain_reference,
            py::arg("a_block_flat"),
            py::arg("b_block_flat"),
            py::arg("block_rows") = 128,
            py::arg("block_cols") = 128
        )
        .def(
            "eval_pcmm_square_ct_plain_stub",
            &FidesCKKSContext::eval_pcmm_square_ct_plain_stub,
            py::arg("x_block_ct"),
            py::arg("w_block_flat"),
            py::arg("block_rows") = 128,
            py::arg("block_cols") = 128
        )
        .def(
            "eval_ccmm_square_ct_ct_stub",
            &FidesCKKSContext::eval_ccmm_square_ct_ct_stub,
            py::arg("a_block_ct"),
            py::arg("b_block_ct"),
            py::arg("block_rows") = 128,
            py::arg("block_cols") = 128
        )

        // 旧接口（保留兼容）
        .def("roundtrip", &FidesCKKSContext::roundtrip)
        .def("eval_add", &FidesCKKSContext::eval_add)
        .def("eval_mult_scalar", &FidesCKKSContext::eval_mult_scalar)
        .def("close", &FidesCKKSContext::close);
}
