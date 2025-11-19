from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("pp_chunk_tuning")
async def pp_chunk_tuning(
    disaggregation_mode: str, tokenizer_manager: "TokenizerManager"
):
    """
    Warmup function for PP mode chunk size coefficient tuning.
    Sends requests with different sequence lengths to collect timing data
    for fitting quadratic coefficients.
    """
    from sglang.srt.server_args import get_global_server_args

    server_args = get_global_server_args()
    
    # Only run if PP mode is enabled and dynamic chunk size is enabled
    if server_args.pp_size <= 1:
        logger.info("PP chunk tuning warmup skipped: PP mode not enabled")
        return
    
    if not server_args.chunked_prefill_size or server_args.chunked_prefill_size <= 0:
        logger.info("PP chunk tuning warmup skipped: chunked prefill not enabled")
        return
    
    # Check if dynamic chunk size is enabled
    import os
    if not os.environ.get("SGLANG_ENABLE_PP_DYNAMIC_CHUNK_SIZE", "false").lower() == "true":
        logger.info("PP chunk tuning warmup skipped: dynamic chunk size not enabled")
        return
    
    logger.info("Starting PP chunk tuning warmup...")
    
    try:
        # Base chunk size for warmup (use chunked_prefill_size from server args)
        base_chunk_size = server_args.chunked_prefill_size
        if base_chunk_size is None or base_chunk_size <= 0:
            logger.warning(
                "PP chunk tuning warmup: chunked_prefill_size not set, "
                "cannot determine base chunk size"
            )
            return
        
        warmup_samples = int(os.environ.get("SGLANG_PP_WARMUP_SAMPLES", "5"))
        
        # Generate requests with different sequence lengths
        # We'll use multiples of base_chunk_size to get different sequence lengths
        chunk_sizes_to_test = []
        
        # Test with base chunk size multiple times (for averaging)
        for _ in range(max(2, warmup_samples // 3)):
            chunk_sizes_to_test.append(base_chunk_size)
        
        # Test with different chunk sizes to get more data points
        if warmup_samples > 3:
            # Add some smaller and larger chunks
            chunk_sizes_to_test.append(base_chunk_size // 2)
            if warmup_samples > 4:
                chunk_sizes_to_test.append(base_chunk_size * 2)
        
        # Ensure we have enough samples
        while len(chunk_sizes_to_test) < warmup_samples:
            chunk_sizes_to_test.append(base_chunk_size)
        
        logger.info(
            f"PP chunk tuning warmup: testing {len(chunk_sizes_to_test)} requests "
            f"with chunk sizes: {chunk_sizes_to_test}"
        )
        
        for i, chunk_size in enumerate(chunk_sizes_to_test):
            # Create input with approximately chunk_size tokens
            # We'll use a simple pattern to generate tokens
            input_ids = list(range(chunk_size))
            
            # Use special rid prefix to mark warmup requests
            warmup_rid = f"PP_WARMUP_{i}_{chunk_size}"
            
            generate_req_input = GenerateReqInput(
                rid=warmup_rid,
                input_ids=input_ids,
                sampling_params={
                    "max_new_tokens": 0,  # Prefill only for timing measurement
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
            )
            
            if disaggregation_mode != "null":
                generate_req_input.bootstrap_room = 0
                generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST
            
            try:
                logger.info(
                    f"PP chunk tuning warmup [{i+1}/{len(chunk_sizes_to_test)}]: "
                    f"sending request with {chunk_size} tokens"
                )
                # Send the request and wait for completion
                async for _ in tokenizer_manager.generate_request(generate_req_input, None):
                    pass
            except Exception as e:
                logger.warning(
                    f"PP chunk tuning warmup request {i+1} failed: {e}. Continuing..."
                )
        
        logger.info(
            f"PP chunk tuning warmup completed. "
            f"Scheduler will fit coefficients from collected timing data."
        )
    finally:
        pass  # No cleanup needed when using rid prefix
