# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import logging
import json
import time
from typing import List, Optional, Tuple
import os

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PromptClient:
    """Simplified client for vLLM server interactions focused on benchmarking needs."""
    
    def __init__(
        self, 
        model_name: str,
        host: str = "127.0.0.1",
        port: int = 8000
    ):
        self.model_name = model_name
        self.host = host
        self.port = port
        
        # Build headers using the same pattern as original vLLM code
        auth_token = self._get_authorization()
        self.headers = {"Authorization": f"Bearer {auth_token}"}
        
        self.completions_url = self._get_api_completions_url()
        self.health_url = self._get_api_health_url()
        self.server_ready = False

    def _get_authorization(self) -> str:
        """Get authorization token following vLLM's original pattern."""
        # Check AUTHORIZATION environment variable (for custom auth tokens)
        if env_auth := os.environ.get("AUTHORIZATION"):
            return env_auth
        
        # Fallback to OPENAI_API_KEY (matching original vLLM pattern)
        return os.environ.get("OPENAI_API_KEY", "")

    def _get_api_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _get_api_base_url_nov1(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _get_api_completions_url(self) -> str:
        return f"{self._get_api_base_url()}/completions"

    def _get_api_health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def _get_api_tokenize_url(self) -> str:
        """Get tokenize API URL."""
        return f"{self._get_api_base_url_nov1()}/tokenize"

    def _get_api_detokenize_url(self) -> str:
        """Get detokenize API URL."""
        return f"{self._get_api_base_url_nov1()}/detokenize"

    def get_health(self) -> requests.Response:
        return requests.get(self.health_url, headers=self.headers)

    def wait_for_healthy(self, timeout: float = 1200.0, interval: int = 10) -> bool:
        timeout = float(timeout)
        if self.server_ready:
            return True

        start_time = time.time()
        total_time_waited = 0

        while time.time() - start_time < timeout:
            req_time = time.time()
            try:
                response = requests.get(
                    self.health_url, headers=self.headers, timeout=interval
                )
                if response.status_code == 200:
                    startup_time = time.time() - start_time
                    logger.info(
                        f"vLLM service is healthy. startup_time:= {startup_time} seconds"
                    )
                    self.server_ready = True
                    return True
                else:
                    logger.warning(f"Health check failed: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Health check failed: {e}")

            total_time_waited = time.time() - start_time
            sleep_interval = max(2 - (time.time() - req_time), 0)
            logger.info(
                f"Service not ready after {total_time_waited:.2f} seconds, "
                f"waiting {sleep_interval:.2f} seconds before polling ..."
            )
            time.sleep(sleep_interval)

        logger.error(f"Service did not become healthy within {timeout} seconds")
        return False

    def entokenize(self, prompt: str, model: Optional[str] = None) -> dict:
        """
        Tokenize text using server-side tokenization.

        Args:
            prompt: The text to tokenize
            model: Model to use for tokenization (defaults to instance model_name)

        Returns:
            Dictionary with tokenization result including tokens
        """
        model_name = model or self.model_name

        url = self._get_api_tokenize_url()

        payload = {
            "model": model_name,
            "prompt": prompt
        }

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Error: {response.status_code}, {response.text}"
            logger.error(f"Server-side tokenization failed: {error_msg}")
            return {"error": error_msg}

    def detokenize(self, tokens: List[int], model: Optional[str] = None) -> dict:
        """
        Detokenize tokens using server-side detokenization.

        Args:
            tokens: The token IDs to detokenize
            model: Model to use for detokenization (defaults to instance model_name)

        Returns:
            Dictionary with detokenization result including prompt
        """
        model_name = model or self.model_name

        url = self._get_api_detokenize_url()

        payload = {
            "model": model_name,
            "tokens": tokens
        }

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Error: {response.status_code}, {response.text}"
            logger.error(f"Server-side detokenization failed: {error_msg}")
            return {"error": error_msg}

    def capture_traces(self, context_lens: List[Tuple[int, int]], timeout: float = 1200.0) -> None:
        """
        Simplified trace capture for benchmarking purposes.
        
        Args:
            context_lens: List of (input_len, output_len) tuples
            timeout: Timeout in seconds
        """
        logger.info("Trace capture requested but not implemented in simplified client")
        logger.info(f"Would capture traces for context lengths: {context_lens}")
        # In a real implementation, this would make requests to warm up the server
        # with the specified input/output length combinations 