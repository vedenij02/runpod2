#!/usr/bin/env python3
"""
RunPod Serverless Handler for PoC v2 using vLLM.

This handler communicates with vLLM's PoC endpoints via HTTP.
vLLM is started by startup.sh before this handler runs.
"""

import os
import time
import logging
import requests
from typing import Generator, Any, Dict, List, Optional

import runpod

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)-16s:%(lineno)-4d %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# vLLM server settings
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
K_DIM = int(os.getenv("K_DIM", "12"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "256"))

# HTTP timeouts
REQUEST_TIMEOUT = 300  # 5 minutes for compute


# =============================================================================
# vLLM Client
# =============================================================================

class VLLMPoCClient:
    """Client for vLLM PoC v2 endpoints."""

    def __init__(self, base_url: str = VLLM_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if vLLM is healthy."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict:
        """Get PoC status."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/v1/pow/status",
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "detail": resp.text}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def init_generate(
        self,
        block_hash: str,
        block_height: int,
        public_key: str,
        node_id: int,
        node_count: int,
        batch_size: int = 32,
        callback_url: Optional[str] = None,
        group_id: int = 0,
        n_groups: int = 1,
    ) -> Dict:
        """
        Start continuous artifact generation.

        Results are sent to callback_url if provided.
        """
        payload = {
            "block_hash": block_hash,
            "block_height": block_height,
            "public_key": public_key,
            "node_id": node_id,
            "node_count": node_count,
            "batch_size": batch_size,
            "group_id": group_id,
            "n_groups": n_groups,
            "params": {
                "model": MODEL_NAME,
                "seq_len": SEQ_LEN,
                "k_dim": K_DIM,
            },
        }
        if callback_url:
            payload["url"] = callback_url

        logger.info(f"init_generate: {payload}")

        resp = self.session.post(
            f"{self.base_url}/api/v1/pow/init/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"init_generate failed: {resp.status_code} {resp.text}")

        return resp.json()

    def generate(
        self,
        block_hash: str,
        block_height: int,
        public_key: str,
        node_id: int,
        node_count: int,
        nonces: List[int],
        batch_size: int = 32,
        wait: bool = True,
    ) -> Dict:
        """
        Generate artifacts for specific nonces.

        Args:
            wait: If True, wait for results synchronously
        """
        payload = {
            "block_hash": block_hash,
            "block_height": block_height,
            "public_key": public_key,
            "node_id": node_id,
            "node_count": node_count,
            "nonces": nonces,
            "batch_size": batch_size,
            "wait": wait,
            "params": {
                "model": MODEL_NAME,
                "seq_len": SEQ_LEN,
                "k_dim": K_DIM,
            },
        }

        logger.info(f"generate: nonces={len(nonces)}, wait={wait}")

        resp = self.session.post(
            f"{self.base_url}/api/v1/pow/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"generate failed: {resp.status_code} {resp.text}")

        return resp.json()

    def stop(self) -> Dict:
        """Stop artifact generation."""
        resp = self.session.post(
            f"{self.base_url}/api/v1/pow/stop",
            json={},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"stop failed: {resp.status_code} {resp.text}")
        return resp.json() if resp.status_code == 200 else {"status": "error"}


# Global client
vllm_client = VLLMPoCClient()


# =============================================================================
# Handler Modes
# =============================================================================

def single_handler_v2(input_data: Dict) -> Generator[Dict, None, None]:
    """
    Single mode for testing - generate artifacts for specific nonces.

    Input:
        block_hash: str
        block_height: int
        public_key: str
        batch_size: int (default 32)
        start_nonce: int (default 0)
        max_batches: int (default 10)
    """
    block_hash = input_data["block_hash"]
    block_height = input_data["block_height"]
    public_key = input_data["public_key"]
    batch_size = input_data.get("batch_size", 32)
    start_nonce = input_data.get("start_nonce", 0)
    max_batches = input_data.get("max_batches", 10)

    logger.info(f"SINGLE V2 MODE: block_hash={block_hash[:16]}..., public_key={public_key[:16]}...")
    logger.info(f"  batch_size={batch_size}, start_nonce={start_nonce}, max_batches={max_batches}")

    # Check vLLM health
    if not vllm_client.health_check():
        yield {"error": "vLLM server not healthy", "error_type": "ServerError", "fatal": True}
        return

    yield {"status": "vllm_ready", "poc_version": "v2"}

    # Generate artifacts batch by batch
    all_artifacts = []
    total_nonces = batch_size * max_batches
    start_time = time.time()

    for batch_idx in range(max_batches):
        batch_start = start_nonce + batch_idx * batch_size
        nonces = list(range(batch_start, batch_start + batch_size))

        yield {
            "status": "computing",
            "batch": batch_idx + 1,
            "total_batches": max_batches,
            "nonces_range": f"{nonces[0]}-{nonces[-1]}",
        }

        try:
            result = vllm_client.generate(
                block_hash=block_hash,
                block_height=block_height,
                public_key=public_key,
                node_id=0,
                node_count=1,
                nonces=nonces,
                batch_size=batch_size,
                wait=True,
            )

            if result.get("status") == "completed":
                artifacts = result.get("artifacts", [])
                all_artifacts.extend(artifacts)
                logger.info(f"Batch {batch_idx + 1}: got {len(artifacts)} artifacts")
            else:
                logger.warning(f"Batch {batch_idx + 1} status: {result.get('status')}")

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} error: {e}")
            yield {"status": "batch_error", "batch": batch_idx + 1, "error": str(e)}

    elapsed = time.time() - start_time

    # Final result
    encoding = {
        "dtype": "f16",
        "k_dim": K_DIM,
        "endian": "le",
    }

    yield {
        "status": "completed",
        "artifacts": all_artifacts,
        "encoding": encoding,
        "stats": {
            "total_artifacts": len(all_artifacts),
            "elapsed_seconds": int(elapsed),
            "artifacts_per_second": round(len(all_artifacts) / elapsed, 2) if elapsed > 0 else 0,
        },
    }


def pooled_handler_v2(input_data: Dict) -> Generator[Dict, None, None]:
    """
    Pooled mode for orchestrator - continuous artifact generation with callbacks.

    Input:
        orchestrator_url: str - URL to send results
        block_hash: str
        block_height: int
        public_key: str
        node_id: int
        node_count: int
        batch_size: int (default 32)
    """
    orchestrator_url = input_data.get("orchestrator_url")
    if not orchestrator_url:
        yield {"error": "orchestrator_url required", "error_type": "ValidationError", "fatal": True}
        return

    block_hash = input_data["block_hash"]
    block_height = input_data["block_height"]
    public_key = input_data["public_key"]
    node_id = input_data.get("node_id", 0)
    node_count = input_data.get("node_count", 1)
    batch_size = input_data.get("batch_size", 32)

    logger.info(f"POOLED V2 MODE: orchestrator={orchestrator_url}")
    logger.info(f"  block_hash={block_hash[:16]}..., public_key={public_key[:16]}...")
    logger.info(f"  node_id={node_id}, node_count={node_count}, batch_size={batch_size}")

    # Check vLLM health
    if not vllm_client.health_check():
        yield {"error": "vLLM server not healthy", "error_type": "ServerError", "fatal": True}
        return

    yield {"status": "vllm_ready", "poc_version": "v2"}

    # Start continuous generation
    try:
        result = vllm_client.init_generate(
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            node_id=node_id,
            node_count=node_count,
            batch_size=batch_size,
            callback_url=orchestrator_url,
        )

        yield {
            "status": "generating",
            "poc_version": "v2",
            "init_result": result,
        }

    except Exception as e:
        logger.error(f"init_generate error: {e}")
        yield {"error": str(e), "error_type": "InitError", "fatal": True}
        return

    # Monitor generation (vLLM sends results to orchestrator via callbacks)
    start_time = time.time()
    last_status_time = 0

    while True:
        elapsed = time.time() - start_time

        # Send status every 30 seconds
        if elapsed - last_status_time >= 30:
            status = vllm_client.get_status()
            yield {
                "status": "progress",
                "elapsed_seconds": int(elapsed),
                "vllm_status": status,
            }
            last_status_time = elapsed

            # Check if generation stopped
            if status.get("status") in ("IDLE", "STOPPED", "ERROR"):
                logger.info(f"Generation ended: {status.get('status')}")
                break

        time.sleep(5)

    # Final status
    final_status = vllm_client.get_status()
    yield {
        "status": "completed",
        "elapsed_seconds": int(time.time() - start_time),
        "final_status": final_status,
    }


# =============================================================================
# Main Handler
# =============================================================================

def handler(job: Dict) -> Generator[Dict, None, None]:
    """
    Main RunPod handler.

    Modes:
        - single_v2: Generate artifacts for testing (returns artifacts directly)
        - pooled_v2: Continuous generation with orchestrator callbacks
    """
    input_data = job.get("input", {})
    mode = input_data.get("mode", "single_v2")

    logger.info(f"Job received: mode={mode}")

    try:
        if mode == "single_v2":
            yield from single_handler_v2(input_data)

        elif mode == "pooled_v2":
            yield from pooled_handler_v2(input_data)

        else:
            yield {
                "error": f"Unknown mode: {mode}",
                "error_type": "ValidationError",
                "supported_modes": ["single_v2", "pooled_v2"],
            }

    except Exception as e:
        logger.exception(f"Handler error: {e}")
        yield {
            "error": str(e),
            "error_type": type(e).__name__,
            "fatal": True,
        }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RunPod2 vLLM Handler Starting")
    logger.info("=" * 60)
    logger.info(f"vLLM URL: {VLLM_BASE_URL}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"K_DIM: {K_DIM}, SEQ_LEN: {SEQ_LEN}")

    # Wait for vLLM to be ready
    logger.info("Checking vLLM health...")
    max_wait = 60
    start = time.time()

    while time.time() - start < max_wait:
        if vllm_client.health_check():
            logger.info("vLLM is healthy!")
            break
        time.sleep(2)
    else:
        logger.error(f"vLLM not ready after {max_wait}s")
        # Continue anyway - vLLM might become ready later

    # Start RunPod serverless worker
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
