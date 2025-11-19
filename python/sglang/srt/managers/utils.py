from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput] = None
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None
    next_token_ids: Optional[torch.Tensor] = None
    num_accepted_tokens: Optional[int] = None
    can_run_cuda_graph: bool = False

    # For output processing
    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None

    # For overlap scheduling
    copy_done: Optional[torch.cuda.Event] = None
    delay_sample_func: Optional[callable] = None
    future_indices: Optional[FutureIndices] = None

    # FIXME(lsyin): maybe move to a better place?
    # sync path: forward stream -> output processor
    accept_lens: Optional[torch.Tensor] = None
    allocate_lens: Optional[torch.Tensor] = None

    # relay path: forward stream -> next step forward
    next_draft_input: Optional[EagleDraftInput] = None

    def copy_to_cpu(self, return_logprob: bool = False):
        """Copy tensors to CPU in overlap scheduling.
        Only the tensors which are needed for processing results are copied,
        e.g., next_token_ids, logits outputs
        """
        if return_logprob:
            if self.logits_output.next_token_logits is not None:
                self.logits_output.next_token_logits = (
                    self.logits_output.next_token_logits.to("cpu", non_blocking=True)
                )
            if self.logits_output.input_token_logprobs is not None:
                self.logits_output.input_token_logprobs = (
                    self.logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                )
        if self.logits_output.hidden_states is not None:
            self.logits_output.hidden_states = self.logits_output.hidden_states.to(
                "cpu", non_blocking=True
            )
        self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)

        if self.accept_lens is not None:
            self.accept_lens = self.accept_lens.to("cpu", non_blocking=True)

        if self.allocate_lens is not None:
            self.allocate_lens = self.allocate_lens.to("cpu", non_blocking=True)

        self.copy_done.record()

    @classmethod
    def from_pp_proxy(
        cls, logits_output, next_pp_outputs: PPProxyTensors, can_run_cuda_graph
    ):
        # TODO(lsyin): refactor PP and avoid using dict
        proxy_dict = next_pp_outputs.tensors
        return cls(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=next_pp_outputs["next_token_ids"],
            extend_input_len_per_req=proxy_dict.get("extend_input_len_per_req", None),
            extend_logprob_start_len_per_req=proxy_dict.get(
                "extend_logprob_start_len_per_req", None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
        )


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:

    logits_output = result.logits_output
    assert logits_output is not None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
        "next_token_logprobs": result.logits_output.next_token_logprobs,
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,
        "input_token_logprobs": result.logits_output.input_token_logprobs,
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,
    }


def get_logprob_from_pp_outputs(
    next_pp_outputs: PPProxyTensors,
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:
    logits_output = LogitsProcessorOutput(
        # Do not send logits and hidden states because they are large
        next_token_logits=None,
        hidden_states=None,
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"
        ],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req


def calculate_dynamic_chunk_size(
    current_seq_len: int,
    target_chunk_time: float,
    quadratic_coeff_a: float,
    linear_coeff_b: float,
    min_chunk_size: int = 1,
    max_chunk_size: Optional[int] = None,
) -> int:
    """
    Calculate the next chunk size to achieve consistent chunk time in PP mode.
    
    Formula: x = (-(2aL+b) + sqrt((2aL+b)^2 + 4aT)) / (2a)
    where:
    - L: current sequence length
    - T: target chunk time
    - a: quadratic coefficient (from attention complexity O(n^2))
    - b: linear coefficient
    
    Args:
        current_seq_len: Current sequence length (L)
        target_chunk_time: Target time for each chunk (T)
        quadratic_coeff_a: Quadratic coefficient (a)
        linear_coeff_b: Linear coefficient (b)
        min_chunk_size: Minimum chunk size to return
        max_chunk_size: Maximum chunk size to return (None means no limit)
    
    Returns:
        Calculated chunk size (x)
    """
    if quadratic_coeff_a <= 0:
        # Fallback to linear if quadratic coefficient is invalid
        if linear_coeff_b > 0:
            chunk_size = int(target_chunk_time / linear_coeff_b)
        else:
            # Default fallback
            chunk_size = 8192
    else:
        # Calculate using quadratic formula: x = (-(2aL+b) + sqrt((2aL+b)^2 + 4aT)) / (2a)
        two_a_L_plus_b = 2 * quadratic_coeff_a * current_seq_len + linear_coeff_b
        discriminant = two_a_L_plus_b * two_a_L_plus_b + 4 * quadratic_coeff_a * target_chunk_time
        
        if discriminant < 0:
            # Fallback if discriminant is negative
            chunk_size = 8192
        else:
            sqrt_discriminant = discriminant ** 0.5
            chunk_size = int((-two_a_L_plus_b + sqrt_discriminant) / (2 * quadratic_coeff_a))
    
    # Apply bounds
    chunk_size = max(chunk_size, min_chunk_size)
    if max_chunk_size is not None:
        chunk_size = min(chunk_size, max_chunk_size)
    
    return chunk_size


def fit_quadratic_coefficients(
    seq_lens: List[int],
    chunk_times: List[float],
) -> Tuple[float, float]:
    """
    Fit quadratic coefficients from sequence lengths and chunk times.
    
    Model: time = a * seq_len^2 + b * seq_len + c
    We solve for a and b using least squares.
    
    Args:
        seq_lens: List of sequence lengths (L)
        chunk_times: List of corresponding chunk execution times (T)
    
    Returns:
        Tuple of (quadratic_coeff_a, linear_coeff_b)
    """
    if len(seq_lens) < 3:
        # Need at least 3 points to fit quadratic model reliably
        logger.warning(
            f"Not enough data points for fitting ({len(seq_lens)} < 3). "
            "Need at least 3 samples with different sequence lengths."
        )
        return (0.0, 0.0)
    
    # Convert to numpy arrays for easier computation
    try:
        import numpy as np
    except ImportError:
        logger.warning(
            "numpy not available, cannot fit quadratic coefficients. "
            "Please install numpy or set coefficients manually."
        )
        return (0.0, 0.0)
    
    L = np.array(seq_lens, dtype=np.float64)
    T = np.array(chunk_times, dtype=np.float64)
    
    # Build design matrix for quadratic model: T = a*L^2 + b*L + c
    # We'll solve for [a, b, c] using least squares
    # X = [L^2, L, 1]
    X = np.column_stack([L * L, L, np.ones(len(L))])
    
    # Solve: X @ [a, b, c]^T = T
    try:
        # Use least squares to solve
        coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
        
        if len(coeffs) >= 2:
            a = float(coeffs[0])  # quadratic coefficient
            b = float(coeffs[1])  # linear coefficient
            c = float(coeffs[2]) if len(coeffs) > 2 else 0.0  # constant term
            
            # Ensure quadratic coefficient is non-negative
            # (time should increase quadratically with seq_len due to attention)
            if a < 0:
                logger.warning(
                    f"Fitted quadratic coefficient is negative ({a:.2e}), "
                    "setting to 0.0. This may indicate insufficient or noisy data."
                )
                a = 0.0
            
            # Log fitting quality if residuals are available
            if len(residuals) > 0:
                mse = float(residuals[0] / len(L)) if len(L) > 0 else 0.0
                logger.info(
                    f"Quadratic fit: a={a:.2e}, b={b:.2e}, c={c:.2e}, "
                    f"MSE={mse:.2e}"
                )
            
            return (a, b)
        else:
            logger.warning("Failed to fit coefficients, insufficient data")
            return (0.0, 0.0)
    except np.linalg.LinAlgError as e:
        logger.warning(f"Failed to fit quadratic coefficients: {e}")
        return (0.0, 0.0)
