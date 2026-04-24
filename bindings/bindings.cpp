#include "fideslib.hpp"
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

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

        .def("roundtrip", &FidesCKKSContext::roundtrip)
        .def("eval_add", &FidesCKKSContext::eval_add)
        .def("eval_mult_scalar", &FidesCKKSContext::eval_mult_scalar)

        .def("close", &FidesCKKSContext::close);
}
