#include "fideslib.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

struct TestResult {
    std::string name;
    bool ok;
    double max_error;
    long long op_us;
};

static void print_vec(const std::string& name, const std::vector<double>& v) {
    std::cout << name;
    for (double x : v) std::cout << x << " ";
    std::cout << "\n";
}

static double max_abs_error(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
    }
    return e;
}

struct ContextBundle {
    fideslib::CryptoContext<fideslib::DCRTPoly> cc;
    fideslib::KeyPair<fideslib::DCRTPoly> kp;
};

int main() {
    using namespace fideslib;
    std::cout << std::fixed << std::setprecision(10);

    auto make_context = [](std::vector<int> devices, bool need_mult_key) {
        CCParams<CryptoContextCKKSRNS> params;
        params.SetMultiplicativeDepth(2);
        params.SetScalingModSize(50);
        params.SetBatchSize(8);
        params.SetRingDim(1 << 14);
        params.SetPlaintextAutoload(true);
        params.SetCiphertextAutoload(true);
        params.SetDevices(std::move(devices));

        auto cc = GenCryptoContext(params);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);

        auto kp = cc->KeyGen();
        if (!kp.publicKey || !kp.secretKey) {
            throw std::runtime_error("KeyGen failed");
        }

        if (need_mult_key) {
            cc->EvalMultKeyGen(kp.secretKey);
        }

        cc->LoadContext(kp.publicKey);

        return ContextBundle{std::move(cc), std::move(kp)};
    };

    std::vector<TestResult> results;

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 0.5, -1.25, 8.0, 9.5};
    std::vector<double> y = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double scalar = 2.5;

    auto bundle1 = make_context({0}, true);
    auto& cc1 = bundle1.cc;
    auto& kp1 = bundle1.kp;

    // Test 1: 单卡 encrypt/decrypt
    {
        auto pt = cc1->MakeCKKSPackedPlaintext(x);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto ct = cc1->Encrypt(pt, kp1.publicKey);
        Plaintext pt_dec;
        auto dec_res = cc1->Decrypt(ct, kp1.secretKey, &pt_dec);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (!dec_res.isValid) {
            std::cerr << "[FAIL] single_gpu_encrypt_decrypt: decrypt invalid\n";
            return 1;
        }

        pt_dec->SetLength(x.size());
        auto out = pt_dec->GetRealPackedValue();
        double err = max_abs_error(x, out);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        print_vec("single roundtrip input:  ", x);
        print_vec("single roundtrip output: ", out);
        std::cout << "single roundtrip max_error: " << err << "\n";
        std::cout << "single roundtrip total_us:  " << us << "\n\n";

        results.push_back({"single_gpu_encrypt_decrypt", dec_res.isValid, err, us});

        pt_dec.reset();
        ct.reset();
        pt.reset();
    }

    // Test 2: 单卡 EvalAdd
    {
        std::vector<double> ref(x.size());
        for (size_t i = 0; i < x.size(); ++i) ref[i] = x[i] + y[i];

        auto pt_x = cc1->MakeCKKSPackedPlaintext(x);
        auto pt_y = cc1->MakeCKKSPackedPlaintext(y);
        auto ct_x = cc1->Encrypt(pt_x, kp1.publicKey);
        auto ct_y = cc1->Encrypt(pt_y, kp1.publicKey);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto ct_sum = cc1->EvalAdd(ct_x, ct_y);
        auto t1 = std::chrono::high_resolution_clock::now();

        Plaintext pt_dec;
        auto dec_res = cc1->Decrypt(ct_sum, kp1.secretKey, &pt_dec);
        if (!dec_res.isValid) {
            std::cerr << "[FAIL] eval_add: decrypt invalid\n";
            return 1;
        }

        pt_dec->SetLength(ref.size());
        auto out = pt_dec->GetRealPackedValue();
        double err = max_abs_error(ref, out);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        print_vec("add ref:    ", ref);
        print_vec("add output: ", out);
        std::cout << "add max_error: " << err << "\n";
        std::cout << "add op_us:     " << us << "\n\n";

        results.push_back({"single_gpu_eval_add", dec_res.isValid, err, us});

        pt_dec.reset();
        ct_sum.reset();
        ct_y.reset();
        ct_x.reset();
        pt_y.reset();
        pt_x.reset();
    }

    // Test 3: 单卡 EvalMult(scalar)
    {
        std::vector<double> ref(x.size());
        for (size_t i = 0; i < x.size(); ++i) ref[i] = x[i] * scalar;

        auto pt_x = cc1->MakeCKKSPackedPlaintext(x);
        auto ct_x = cc1->Encrypt(pt_x, kp1.publicKey);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto ct_mul = cc1->EvalMult(ct_x, scalar);
        auto t1 = std::chrono::high_resolution_clock::now();

        Plaintext pt_dec;
        auto dec_res = cc1->Decrypt(ct_mul, kp1.secretKey, &pt_dec);
        if (!dec_res.isValid) {
            std::cerr << "[FAIL] eval_mult_scalar: decrypt invalid\n";
            return 1;
        }

        pt_dec->SetLength(ref.size());
        auto out = pt_dec->GetRealPackedValue();
        double err = max_abs_error(ref, out);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        print_vec("mul ref:    ", ref);
        print_vec("mul output: ", out);
        std::cout << "mul max_error: " << err << "\n";
        std::cout << "mul op_us:     " << us << "\n\n";

        results.push_back({"single_gpu_eval_mult_scalar", dec_res.isValid, err, us});

        pt_dec.reset();
        ct_mul.reset();
        ct_x.reset();
        pt_x.reset();
    }

    kp1.secretKey.reset();
    kp1.publicKey.reset();
    cc1.reset();
    cudaDeviceSynchronize();

    auto bundle2 = make_context({0, 1}, false);
    auto& cc2 = bundle2.cc;
    auto& kp2 = bundle2.kp;

    // Test 4: 多卡 encrypt/decrypt
    {
        auto pt = cc2->MakeCKKSPackedPlaintext(x);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto ct = cc2->Encrypt(pt, kp2.publicKey);
        Plaintext pt_dec;
        auto dec_res = cc2->Decrypt(ct, kp2.secretKey, &pt_dec);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (!dec_res.isValid) {
            std::cerr << "[FAIL] multi_gpu_encrypt_decrypt: decrypt invalid\n";
            return 1;
        }

        pt_dec->SetLength(x.size());
        auto out = pt_dec->GetRealPackedValue();
        double err = max_abs_error(x, out);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        print_vec("multi roundtrip input:  ", x);
        print_vec("multi roundtrip output: ", out);
        std::cout << "multi roundtrip max_error: " << err << "\n";
        std::cout << "multi roundtrip total_us:  " << us << "\n\n";

        results.push_back({"multi_gpu_encrypt_decrypt", dec_res.isValid, err, us});

        pt_dec.reset();
        ct.reset();
        pt.reset();
    }

    kp2.secretKey.reset();
    kp2.publicKey.reset();
    cc2.reset();
    cudaDeviceSynchronize();

    std::cout << "================ SUMMARY ================\n";
    for (const auto& r : results) {
        std::cout << r.name
                  << " | ok=" << std::boolalpha << r.ok
                  << " | max_error=" << r.max_error
                  << " | op_us=" << r.op_us
                  << "\n";
    }
    std::cout << "=========================================\n";

    return 0;
}
