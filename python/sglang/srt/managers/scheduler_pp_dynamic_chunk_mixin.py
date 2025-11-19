from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import (
    calculate_dynamic_chunk_size,
    fit_quadratic_coefficients,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class SchedulerPPDynamicChunkMixin:
    """Mixin for PP mode dynamic chunk size adjustment."""

    def init_pp_dynamic_chunk_size(self: "Scheduler"):
        """Initialize PP dynamic chunk size adjustment."""
        from sglang.srt.utils import get_bool_env_var, get_int_env_var

        self.enable_dynamic_chunk_size = (
            self.pp_size > 1
            and self.chunked_prefill_size is not None
            and get_bool_env_var("SGLANG_ENABLE_PP_DYNAMIC_CHUNK_SIZE", "false")
        )
        
        if self.enable_dynamic_chunk_size:
            # Base chunk size for warmup (use chunked_prefill_size)
            self.base_chunk_size = self.chunked_prefill_size
            # Target chunk time (will be measured during warmup)
            self.target_chunk_time: Optional[float] = None
            # Quadratic and linear coefficients (will be auto-fitted during warmup)
            self.quadratic_coeff_a = 0.0
            self.linear_coeff_b = 0.0
            # Warmup state - collect data for fitting
            # Data collection happens during warmup phase (via warmup requests)
            self.warmup_chunk_times: List[float] = []
            self.warmup_seq_lens: List[int] = []  # Sequence lengths at start of chunk
            self.warmup_chunk_sizes: List[int] = []  # Actual chunk sizes processed
            self.warmup_complete = False
            self.warmup_samples_needed = get_int_env_var(
                "SGLANG_PP_WARMUP_SAMPLES", 5
            )  # Need at least 3-5 samples for fitting
            # Store original chunked_prefill_size for fallback
            self.original_chunked_prefill_size = self.chunked_prefill_size
        else:
            self.target_chunk_time = None
            self.quadratic_coeff_a = 0.0
            self.linear_coeff_b = 0.0

    def _is_warmup_request(self: "Scheduler", batch: ScheduleBatch) -> bool:
        """Check if the batch contains warmup requests."""
        return any(
            req.rid is not None and req.rid.startswith("PP_WARMUP_")
            for req in batch.reqs
        )

    def _extract_chunk_info(
        self: "Scheduler", batch: ScheduleBatch
    ) -> tuple[int, int, int]:
        """
        Extract chunk information from batch.
        
        Returns:
            (chunk_size, seq_len_at_start, current_seq_len)
        """
        req = batch.reqs[0]
        if hasattr(req, "extend_input_len") and req.extend_input_len > 0:
            chunk_size = req.extend_input_len
            seq_len_at_start = len(req.prefix_indices)
            current_seq_len = seq_len_at_start + chunk_size
        else:
            # Non-chunked prefill
            current_seq_len = req.seqlen
            chunk_size = current_seq_len
            seq_len_at_start = 0
        
        return chunk_size, seq_len_at_start, current_seq_len

    def _collect_warmup_sample(
        self: "Scheduler",
        chunk_time: float,
        chunk_size: int,
        seq_len_at_start: int,
    ) -> bool:
        """
        Collect a warmup sample and check if warmup is complete.
        
        Returns:
            True if warmup is complete, False otherwise
        """
        self.warmup_chunk_times.append(chunk_time)
        self.warmup_seq_lens.append(seq_len_at_start)
        self.warmup_chunk_sizes.append(chunk_size)
        
        logger.debug(
            f"[PP Dynamic Chunk] Warmup sample {len(self.warmup_chunk_times)}: "
            f"seq_len={seq_len_at_start}, chunk_size={chunk_size}, "
            f"time={chunk_time:.4f}s"
        )
        
        # Check if we have enough samples for fitting
        if len(self.warmup_chunk_times) >= self.warmup_samples_needed:
            self._complete_warmup()
            return True
        return False

    def _complete_warmup(self: "Scheduler"):
        """Complete warmup phase by fitting coefficients."""
        # Calculate average chunk time for base chunk size as target
        base_chunk_indices = [
            i
            for i, size in enumerate(self.warmup_chunk_sizes)
            if size == self.base_chunk_size
        ]
        if base_chunk_indices:
            base_chunk_times = [
                self.warmup_chunk_times[i] for i in base_chunk_indices
            ]
            self.target_chunk_time = sum(base_chunk_times) / len(base_chunk_times)
        else:
            # Fallback: use average of all samples
            self.target_chunk_time = sum(self.warmup_chunk_times) / len(
                self.warmup_chunk_times
            )
        
        # Auto-fit coefficients from warmup data
        # Use sequence lengths at start + chunk sizes for fitting
        # The effective sequence length for attention is seq_len_at_start + chunk_size
        effective_seq_lens = [
            seq_len + chunk_size
            for seq_len, chunk_size in zip(
                self.warmup_seq_lens, self.warmup_chunk_sizes
            )
        ]
        
        fitted_a, fitted_b = fit_quadratic_coefficients(
            effective_seq_lens, self.warmup_chunk_times
        )
        
        if fitted_a > 0 or fitted_b > 0:
            self.quadratic_coeff_a = fitted_a
            self.linear_coeff_b = fitted_b
            logger.info(
                f"[PP Dynamic Chunk] Coefficients fitted: "
                f"a={fitted_a:.2e}, b={fitted_b:.2e} "
                f"(from {len(self.warmup_chunk_times)} samples)"
            )
        else:
            logger.warning(
                "[PP Dynamic Chunk] Failed to fit coefficients, "
                "falling back to time-based heuristic"
            )
        
        self.warmup_complete = True
        logger.info(
            f"[PP Dynamic Chunk] Warmup complete. "
            f"Target chunk time: {self.target_chunk_time:.4f}s, "
            f"Base chunk size: {self.base_chunk_size}, "
            f"Coefficients: a={self.quadratic_coeff_a:.2e}, "
            f"b={self.linear_coeff_b:.2e}"
        )

    def record_chunk_execution_time(
        self: "Scheduler",
        batch: ScheduleBatch,
        chunk_start_time: Optional[float],
    ):
        """Record chunk execution time for dynamic chunk size adjustment.
        
        This method is only called during warmup phase to collect timing data
        for fitting quadratic coefficients.
        """
        if not (
            self.enable_dynamic_chunk_size
            and batch.forward_mode == ForwardMode.EXTEND
            and chunk_start_time is not None
            and batch.reqs
        ):
            return
        
        # Only record during warmup phase
        if self.warmup_complete:
            return
        
        # Check if this is a warmup request
        is_warmup_request = self._is_warmup_request(batch)
        if not is_warmup_request:
            return
        
        chunk_end_time = time.perf_counter()
        chunk_time = chunk_end_time - chunk_start_time
        
        # Extract chunk information
        chunk_size, seq_len_at_start, current_seq_len = self._extract_chunk_info(batch)
        
        # Collect warmup sample for fitting
        self._collect_warmup_sample(chunk_time, chunk_size, seq_len_at_start)

    def get_dynamic_chunk_size(self: "Scheduler") -> Optional[int]:
        """
        Calculate dynamic chunk size based on current state.
        
        Returns:
            Dynamic chunk size to use, or None to use original chunked_prefill_size
        """
        if not self.enable_dynamic_chunk_size:
            return None
        
        if not self.warmup_complete:
            # During warmup, use base chunk size
            return self.base_chunk_size
        
        if self.target_chunk_time is None or self.chunked_req is None:
            return None
        
        # Calculate dynamic chunk size based on current sequence length
        # Get the current sequence length of the chunked request
        current_seq_len = self.chunked_req.seqlen
        
        # Coefficients must be fitted during warmup phase
        # If coefficients are not available, return None to use original chunk size
        if not (self.quadratic_coeff_a > 0 or self.linear_coeff_b > 0):
            return None
        
        calculated_chunk_size = calculate_dynamic_chunk_size(
            current_seq_len=current_seq_len,
            target_chunk_time=self.target_chunk_time,
            quadratic_coeff_a=self.quadratic_coeff_a,
            linear_coeff_b=self.linear_coeff_b,
            min_chunk_size=self.page_size,  # At least one page
            max_chunk_size=self.original_chunked_prefill_size,
        )
        
        # Align to page size
        if calculated_chunk_size > 0:
            dynamic_chunk_size = (
                calculated_chunk_size // self.page_size
            ) * self.page_size
        else:
            dynamic_chunk_size = self.page_size
        
        return dynamic_chunk_size

