#include "native_helpers.hpp"

#include <stdexcept>

namespace hegpt {

namespace {

// 行主序矩阵乘：A(m,k) @ B(k,n) -> C(m,n)
std::vector<double> matmul_row_major(
    const std::vector<double>& a,
    const std::vector<double>& b,
    std::uint32_t m,
    std::uint32_t k,
    std::uint32_t n
) {
    if (a.size() != static_cast<std::size_t>(m) * k) {
        throw std::runtime_error("matmul_row_major: a.size mismatch");
    }
    if (b.size() != static_cast<std::size_t>(k) * n) {
        throw std::runtime_error("matmul_row_major: b.size mismatch");
    }

    std::vector<double> c(static_cast<std::size_t>(m) * n, 0.0);

    for (std::uint32_t i = 0; i < m; ++i) {
        for (std::uint32_t p = 0; p < k; ++p) {
            const double a_ip = a[static_cast<std::size_t>(i) * k + p];
            for (std::uint32_t j = 0; j < n; ++j) {
                c[static_cast<std::size_t>(i) * n + j] +=
                    a_ip * b[static_cast<std::size_t>(p) * n + j];
            }
        }
    }
    return c;
}

std::vector<double> add_same_shape(
    const std::vector<double>& x,
    const std::vector<double>& y
) {
    if (x.size() != y.size()) {
        throw std::runtime_error("add_same_shape: size mismatch");
    }
    std::vector<double> out(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] + y[i];
    }
    return out;
}

}  // namespace

std::uint32_t ceil_div_u32(std::uint32_t a, std::uint32_t b) {
    if (b == 0) {
        throw std::runtime_error("ceil_div_u32: divisor must be non-zero");
    }
    return (a + b - 1) / b;
}

BlockLayoutSummary summarize_block_layout(
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t ring_dim,
    std::uint32_t block_rows,
    std::uint32_t block_cols
) {
    if (block_rows == 0 || block_cols == 0) {
        throw std::runtime_error("summarize_block_layout: invalid block shape");
    }

    const std::uint32_t slots_per_ct = ring_dim / 2;
    const std::uint32_t logical_block_elems = block_rows * block_cols;

    // 第一版固定为角色化 tile：
    //   activation tile = 128x64
    //   weight tile     = 64x128
    const std::uint32_t act_tile_rows = block_rows;
    const std::uint32_t act_tile_cols = 64;
    const std::uint32_t weight_tile_rows = 64;
    const std::uint32_t weight_tile_cols = block_cols;

    if (act_tile_rows * act_tile_cols > slots_per_ct) {
        throw std::runtime_error("activation tile exceeds slots_per_ct");
    }
    if (weight_tile_rows * weight_tile_cols > slots_per_ct) {
        throw std::runtime_error("weight tile exceeds slots_per_ct");
    }
    if (block_cols % act_tile_cols != 0) {
        throw std::runtime_error("block_cols must be divisible by act_tile_cols");
    }
    if (block_rows % weight_tile_rows != 0) {
        throw std::runtime_error("block_rows must be divisible by weight_tile_rows");
    }

    BlockLayoutSummary s{};
    s.rows = rows;
    s.cols = cols;
    s.ring_dim = ring_dim;
    s.slots_per_ct = slots_per_ct;
    s.block_rows = block_rows;
    s.block_cols = block_cols;
    s.logical_block_elems = logical_block_elems;
    s.act_tile_rows = act_tile_rows;
    s.act_tile_cols = act_tile_cols;
    s.weight_tile_rows = weight_tile_rows;
    s.weight_tile_cols = weight_tile_cols;
    s.tiles_per_logical_block = block_cols / act_tile_cols;  // 这里等于 2
    return s;
}

std::vector<double> extract_block_row_major(
    const std::vector<double>& matrix_flat,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t row0,
    std::uint32_t col0,
    std::uint32_t block_rows,
    std::uint32_t block_cols,
    bool zero_pad
) {
    if (matrix_flat.size() != static_cast<std::size_t>(rows) * cols) {
        throw std::runtime_error("extract_block_row_major: matrix_flat size mismatch");
    }

    std::vector<double> block(static_cast<std::size_t>(block_rows) * block_cols, 0.0);

    for (std::uint32_t r = 0; r < block_rows; ++r) {
        for (std::uint32_t c = 0; c < block_cols; ++c) {
            const std::uint32_t rr = row0 + r;
            const std::uint32_t cc = col0 + c;

            if (rr < rows && cc < cols) {
                block[static_cast<std::size_t>(r) * block_cols + c] =
                    matrix_flat[static_cast<std::size_t>(rr) * cols + cc];
            } else if (!zero_pad) {
                throw std::runtime_error("extract_block_row_major: out-of-range without zero_pad");
            }
        }
    }

    return block;
}

std::vector<std::vector<double>> split_activation_block_tiles(
    const std::vector<double>& block_flat,
    std::uint32_t block_rows,
    std::uint32_t block_cols,
    std::uint32_t tile_cols
) {
    if (block_flat.size() != static_cast<std::size_t>(block_rows) * block_cols) {
        throw std::runtime_error("split_activation_block_tiles: block size mismatch");
    }
    if (tile_cols == 0 || (block_cols % tile_cols) != 0) {
        throw std::runtime_error("split_activation_block_tiles: invalid tile_cols");
    }

    const std::uint32_t num_tiles = block_cols / tile_cols;
    std::vector<std::vector<double>> out;
    out.reserve(num_tiles);

    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        const std::uint32_t c0 = t * tile_cols;
        std::vector<double> tile(static_cast<std::size_t>(block_rows) * tile_cols, 0.0);

        for (std::uint32_t r = 0; r < block_rows; ++r) {
            for (std::uint32_t c = 0; c < tile_cols; ++c) {
                tile[static_cast<std::size_t>(r) * tile_cols + c] =
                    block_flat[static_cast<std::size_t>(r) * block_cols + (c0 + c)];
            }
        }
        out.push_back(std::move(tile));
    }
    return out;
}

std::vector<std::vector<double>> split_weight_block_tiles(
    const std::vector<double>& block_flat,
    std::uint32_t block_rows,
    std::uint32_t block_cols,
    std::uint32_t tile_rows
) {
    if (block_flat.size() != static_cast<std::size_t>(block_rows) * block_cols) {
        throw std::runtime_error("split_weight_block_tiles: block size mismatch");
    }
    if (tile_rows == 0 || (block_rows % tile_rows) != 0) {
        throw std::runtime_error("split_weight_block_tiles: invalid tile_rows");
    }

    const std::uint32_t num_tiles = block_rows / tile_rows;
    std::vector<std::vector<double>> out;
    out.reserve(num_tiles);

    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        const std::uint32_t r0 = t * tile_rows;
        std::vector<double> tile(static_cast<std::size_t>(tile_rows) * block_cols, 0.0);

        for (std::uint32_t r = 0; r < tile_rows; ++r) {
            for (std::uint32_t c = 0; c < block_cols; ++c) {
                tile[static_cast<std::size_t>(r) * block_cols + c] =
                    block_flat[static_cast<std::size_t>(r0 + r) * block_cols + c];
            }
        }
        out.push_back(std::move(tile));
    }
    return out;
}

std::vector<double> pcmm_square_plain_reference(
    const std::vector<double>& x_block_flat,
    const std::vector<double>& w_block_flat,
    std::uint32_t block_rows,
    std::uint32_t block_cols,
    std::uint32_t split_dim
) {
    if (split_dim == 0 || block_cols % split_dim != 0 || block_rows % split_dim != 0) {
        throw std::runtime_error("pcmm_square_plain_reference: invalid split_dim");
    }

    const auto x_tiles = split_activation_block_tiles(
        x_block_flat, block_rows, block_cols, split_dim
    );
    const auto w_tiles = split_weight_block_tiles(
        w_block_flat, block_rows, block_cols, split_dim
    );

    if (x_tiles.size() != w_tiles.size()) {
        throw std::runtime_error("pcmm_square_plain_reference: tile count mismatch");
    }

    std::vector<double> out(static_cast<std::size_t>(block_rows) * block_cols, 0.0);

    for (std::size_t i = 0; i < x_tiles.size(); ++i) {
        const auto partial = matmul_row_major(
            x_tiles[i],     // block_rows x split_dim
            w_tiles[i],     // split_dim x block_cols
            block_rows,
            split_dim,
            block_cols
        );
        out = add_same_shape(out, partial);
    }

    return out;
}

std::vector<double> ccmm_square_plain_reference(
    const std::vector<double>& a_block_flat,
    const std::vector<double>& b_block_flat,
    std::uint32_t block_rows,
    std::uint32_t block_cols
) {
    return matmul_row_major(
        a_block_flat,
        b_block_flat,
        block_rows,
        block_cols,
        block_cols
    );
}

// ----------------------------------------------------------------------
// 协议入口壳子
// 当前仍然只做“接口固定”，不做真正协议迁移。
// 下一步真正开始从 bert-tiny 抽 PCMMSquare / CCMMSquare 时，只改这里。
// ----------------------------------------------------------------------

Ciphertext<DCRTPoly> pcmm_square_ct_plain_protocol(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& x_block_ct,
    const std::vector<double>& w_block_flat,
    std::uint32_t block_rows,
    std::uint32_t block_cols,
    std::uint32_t split_dim
) {
    (void)cc;
    (void)x_block_ct;
    (void)w_block_flat;
    (void)block_rows;
    (void)block_cols;
    (void)split_dim;

    throw std::runtime_error(
        "pcmm_square_ct_plain_protocol: bert-tiny PCMMSquare port not implemented yet"
    );
}

Ciphertext<DCRTPoly> ccmm_square_ct_ct_protocol(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& a_block_ct,
    const Ciphertext<DCRTPoly>& b_block_ct,
    std::uint32_t block_rows,
    std::uint32_t block_cols
) {
    (void)cc;
    (void)a_block_ct;
    (void)b_block_ct;
    (void)block_rows;
    (void)block_cols;

    throw std::runtime_error(
        "ccmm_square_ct_ct_protocol: bert-tiny CCMMSquare port not implemented yet"
    );
}

}  // namespace hegpt
