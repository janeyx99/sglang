/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <torch/csrc/stable/tensor_struct.h>

#include <cstdint>
#include <optional>

void causal_conv1d_update(
    const torch::stable::Tensor& x,
    const torch::stable::Tensor& conv_state,
    const torch::stable::Tensor& weight,
    const std::optional<torch::stable::Tensor>& bias_,
    bool silu_activation,
    const std::optional<torch::stable::Tensor>& cache_seqlens_,
    const std::optional<torch::stable::Tensor>& conv_state_indices_,
    int64_t pad_slot_id);

void causal_conv1d_fwd(
    const torch::stable::Tensor& x,
    const torch::stable::Tensor& weight,
    const std::optional<torch::stable::Tensor>& bias_,
    const std::optional<torch::stable::Tensor>& conv_states,
    const std::optional<torch::stable::Tensor>& query_start_loc,
    const std::optional<torch::stable::Tensor>& cache_indices,
    const std::optional<torch::stable::Tensor>& has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id);
