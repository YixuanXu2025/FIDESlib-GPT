#pragma once

#include "fideslib.hpp"

#include <cstdint>
#include <vector>

namespace hegpt {

using namespace fideslib;

// ----------------------------------------------------------------------
// block / tile 布局摘要
// ----------------------------------------------------------------------
struct BlockLayoutSummary {
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t ring_dim;
    std::uint32_t slots_per_ct;
    std::uint32_t block_rows;
    std::uint32_t block_cols;
    std::uint32_t logical_block_elems;

    // 角色化 tile：
    //   activation tile = block_rows x act_tile_cols
    //   weight tile     = weight_tile_rows x block_cols
    std::uint32_t act_tile_rows;
    std::uint32_t act_tile_cols;
    std::uint32_t weight_tile_rows;
    std::uint32_t weight_tile_cols;

    // 一个 logical block 会拆成多少个 role-specific tiles
    std::uint32_t tiles_per_logical_block;
};

std::uint32_t ceil_div_u32(std::uint32_t a, std::uint32_t b);

BlockLayoutSummary summarize_block_layout(
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t ring_dim = 16384,
    std::uint32_t block_rows = 128,
    std::uint32_t block_cols = 128
);

// ----------------------------------------------------------------------
// 纯明文参考：block / tile 工具
// ----------------------------------------------------------------------
std::vector<double> extract_block_row_major(
    const std::vector<double>& matrix_flat,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t row0,
    std::uint32_t col0,
    std::uint32_t block_rows = 128,
    std::uint32_t block_cols = 128,
    bool zero_pad = true
);

std::vector<std::vector<double>> split_activation_block_tiles(
    const std::vector<double>& block_flat,
    std::uint32_t block_rows = 128,
    std::uint32_t block_cols = 128,
    std::uint32_t tile_cols = 64
);

std::vector<std::vector<double>> split_weight_block_tiles(
    const std::vector<double>& block_flat,
    std::uint32_t block_rows = 128,
    std::uint32_t block_cols = 128,
    std::uint32_t tile_rows = 64
);

std::vector<double> pcmm_square_plain_reference(
    const std::vector<double>& x_block_flat,
    const std::vector<double>& w_block_flat,
    std::uint32_t block_rows = 128,
    std::uint32_t block_cols = 128,
    std::uint32_t split_dim = 64
);

std::vector<double> ccmm_square_plain_reference(
    const std::vector<double>& a_block_flat,
    const std::vector<double>& b_block_flat,
    std::uint32_t block_rows = 128,
    std::uint32_t block_cols = 128
);

// ----------------------------------------------------------------------
// 协议入口壳子（下一步真正迁移 bert-tiny 的 PCMMSquare / CCMMSquare 时填充）
//
// 这一层的作用：
//   - 先把“helper API 形状”固定下来
//   - bindings 以后只转调这层
//   - 真协议实现落在 native_helpers.cpp，不污染 bindings.cpp
// ----------------------------------------------------------------------

// ct-pt：单个 logical block 的 PCMM 风格乘法
Ciphertext<DCRTPoly> pcmm_square_ct_plain_protocol(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& x_block_ct,
    const std::vector<double>& w_block_flat,
    std::uint32_t block_rows = 128,
    std::uint32_t block_cols = 128,
    std::uint32_t split_dim = 64
);

// ct-ct：单个 logical block 的 CCMM 风格乘法
Ciphertext<DCRTPoly> ccmm_square_ct_ct_protocol(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& a_block_ct,
    const Ciphertext<DCRTPoly>& b_block_ct,
    std::uint32_t block_rows = 128,
    std::uint32_t block_cols = 128
);

}  // namespace hegpt
