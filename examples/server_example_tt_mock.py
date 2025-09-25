# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import runpy
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm import ModelRegistry


class MockTTDevice:
    """Mock TT device that mimics the interface of real TT devices."""
    
    def __init__(self, num_devices: int = 1):
        self._num_devices = num_devices
    
    def get_num_devices(self) -> int:
        return self._num_devices


class MockTTModel(nn.Module):
    """Mock TT model that produces pseudo-random tokens efficiently."""
    
    def __init__(self, config: PretrainedConfig, device: Any, max_num_seqs: int, 
                 max_seq_len: int, tt_data_parallel: int = 1):
        super().__init__()
        self.config = config
        self.device = device
        self.max_num_seqs = max_num_seqs
        self.max_seq_len = max_seq_len
        self.tt_data_parallel = tt_data_parallel
        
        # Model dimensions
        self.vocab_size = getattr(config, 'vocab_size', 32000)
        self.hidden_size = getattr(config, 'hidden_size', 4096)
        self.num_attention_heads = getattr(config, 'num_attention_heads', 32)
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', 
                                          getattr(config, 'num_attention_heads', 32))
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_layers = getattr(config, 'num_hidden_layers', 32)
        
        # For encoder-decoder models
        self.max_cross_attn_tokens = getattr(config, 'max_position_embeddings', 2048)
        
        # Random seed for consistent pseudo-random output
        self.generator = torch.Generator()
        self.generator.manual_seed(42)
        
        print(f"MockTTModel initialized: vocab_size={self.vocab_size}, "
              f"hidden_size={self.hidden_size}, num_layers={self.num_layers}")
    
    @classmethod
    def initialize_vllm_model(cls, config: PretrainedConfig, device: Any, 
                             max_num_seqs: int, max_seq_len: int, 
                             tt_data_parallel: int = 1) -> 'MockTTModel':
        """Initialize the mock TT model for vLLM."""
        return cls(config, device, max_num_seqs, max_seq_len, tt_data_parallel)
    
    def allocate_kv_cache(self, kv_cache_shape: Tuple[int, ...], 
                         dtype: torch.dtype, num_layers: int) -> List[List[torch.Tensor]]:
        """Allocate KV cache tensors."""
        kv_cache = []
        for _ in range(num_layers):
            # Create K and V cache tensors
            cache_k = torch.zeros(kv_cache_shape, dtype=dtype, device="cpu")
            cache_v = torch.zeros(kv_cache_shape, dtype=dtype, device="cpu")
            kv_cache.append([cache_k, cache_v])
        return kv_cache
    
    def prefill_forward(self, tokens: torch.Tensor, page_table: torch.Tensor,
                       kv_cache: List[List[torch.Tensor]], prompt_lens: List[int],
                       empty_slots: Optional[List[int]] = None,
                       cross_page_table: Optional[torch.Tensor] = None,
                       sampling_params: Optional[Any] = None,
                       **kwargs) -> torch.Tensor:
        """Mock prefill forward pass."""
        batch_size = tokens.shape[0]
        max_seq_len = tokens.shape[1] if len(tokens.shape) > 1 else 1
        
        # Simulate some processing time
        time.sleep(0.001)  # 1ms delay to simulate computation
        
        if sampling_params is not None:
            # Return token IDs directly (sampling on device)
            return torch.randint(0, self.vocab_size, (batch_size,), 
                               dtype=torch.long, generator=self.generator)
        else:
            # Return logits for host-side sampling
            return torch.randn(batch_size, max_seq_len, self.vocab_size, 
                             generator=self.generator)
    
    def decode_forward(self, tokens: torch.Tensor, page_table: torch.Tensor,
                      kv_cache: List[List[torch.Tensor]], start_pos: torch.Tensor,
                      enable_trace: bool = True, read_from_device: bool = False,
                      cross_page_table: Optional[torch.Tensor] = None,
                      sampling_params: Optional[Any] = None,
                      prefill_cross_attention_masks: Optional[List] = None,
                      prefill_full_text_row_masked_out_mask: Optional[List] = None,
                      decode_cross_attention_masks: Optional[List] = None,
                      decode_full_text_row_masked_out_mask: Optional[List] = None,
                      rot_mats_all_users: Optional[List] = None,
                      **kwargs) -> torch.Tensor:
        """Mock decode forward pass."""
        batch_size = tokens.shape[0]
        
        # Simulate some processing time
        time.sleep(0.001)  # 1ms delay to simulate computation
        
        if sampling_params is not None:
            # Return token IDs directly (sampling on device)
            return torch.randint(0, self.vocab_size, (batch_size,), 
                               dtype=torch.long, generator=self.generator)
        else:
            # Return logits for host-side sampling (last token only for decode)
            return torch.randn(batch_size, 1, self.vocab_size, 
                             generator=self.generator)
    
    def read_decode_output(self, tt_out: torch.Tensor, 
                          async_read: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Mock reading decode output from device."""
        if async_read:
            # Return tensor and mock read event
            mock_event = MockReadEvent()
            return tt_out, [mock_event]
        else:
            return tt_out
    
    def process_decode_output_host(self, tt_out: torch.Tensor, 
                                  is_tokens: bool = False) -> torch.Tensor:
        """Mock processing decode output on host."""
        if is_tokens:
            # Already token IDs
            return tt_out
        else:
            # Convert logits to token IDs
            return torch.argmax(tt_out, dim=-1).squeeze(-1)


class MockReadEvent:
    """Mock read event for async operations."""
    pass


def mock_event_synchronize(event):
    """Mock event synchronization."""
    pass


# Mock ttnn module functions that might be called
class MockTTNN:
    @staticmethod
    def get_arch_name():
        return "wormhole_b0"
    
    @staticmethod
    def event_synchronize(event):
        mock_event_synchronize(event)
    
    @staticmethod
    def get_device_ids():
        return [0, 1]  # Mock 2 devices
    
    @staticmethod
    def open_mesh_device(*args, **kwargs):
        return MockTTDevice(num_devices=2)
    
    @staticmethod
    def close_mesh_device(*args, **kwargs):
        pass
    
    @staticmethod
    def ReadDeviceProfiler(*args, **kwargs):
        pass
    
    @staticmethod
    def set_fabric_config(*args, **kwargs):
        pass
    
    # Mock ttnn enums and classes
    class DispatchCoreAxis:
        ROW = "row"
        COL = "col"
    
    class DispatchCoreConfig:
        def __init__(self, axis=None):
            self.axis = axis
    
    class FabricConfig:
        DISABLED = "disabled"
        FABRIC_1D = "fabric_1d"
        FABRIC_1D_RING = "fabric_1d_ring"
        FABRIC_2D = "fabric_2d"
        CUSTOM = "custom"
    
    class MeshShape:
        def __init__(self, *args):
            self.shape = args
    
    class cluster:
        class ClusterType:
            GALAXY = "galaxy"
        
        @staticmethod
        def get_cluster_type():
            return MockTTNN.cluster.ClusterType.GALAXY


def register_mock_tt_models():
    """Register mock TT models with vLLM - mimics offline_inference_tt.py pattern."""
    
    # Create a mock module that can be imported - mimic the real TT models structure
    import sys
    
    class MockTTModelModule:
        MockTTModel = MockTTModel
        # Create aliases for all the model classes that real TT models would have
        LlamaForCausalLM = MockTTModel
        MllamaForConditionalGeneration = MockTTModel
        QwenForCausalLM = MockTTModel
        MistralForCausalLM = MockTTModel
        GemmaForCausalLM = MockTTModel
        Gemma2ForCausalLM = MockTTModel
        Gemma3ForConditionalGeneration = MockTTModel
    
    # Register mock modules to mimic the real TT model paths
    sys.modules['models'] = type('MockModule', (), {})()
    sys.modules['models.tt_transformers'] = type('MockModule', (), {})()
    sys.modules['models.tt_transformers.tt'] = type('MockModule', (), {})()
    sys.modules['models.tt_transformers.tt.generator_vllm'] = MockTTModelModule()
    
    # Register models exactly like offline_inference_tt.py does
    # Llama3.1/3.2 - Text
    ModelRegistry.register_model("TTLlamaForCausalLM", 
                                "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM")
    
    # Llama3.2 - Vision  
    ModelRegistry.register_model("TTMllamaForConditionalGeneration",
                                "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration")
    
    # Qwen2.5 - Text
    ModelRegistry.register_model("TTQwen2ForCausalLM", 
                                "models.tt_transformers.tt.generator_vllm:QwenForCausalLM")
    ModelRegistry.register_model("TTQwen3ForCausalLM", 
                                "models.tt_transformers.tt.generator_vllm:QwenForCausalLM")
    
    # Mistral
    ModelRegistry.register_model("TTMistralForCausalLM",
                                "models.tt_transformers.tt.generator_vllm:MistralForCausalLM")
    
    # Gemma
    ModelRegistry.register_model("TTGemmaForCausalLM",
                                "models.tt_transformers.tt.generator_vllm:GemmaForCausalLM")
    ModelRegistry.register_model("TTGemma2ForCausalLM", 
                                "models.tt_transformers.tt.generator_vllm:Gemma2ForCausalLM")
    
    # Gemma3
    ModelRegistry.register_model("TTGemma3ForConditionalGeneration",
                                "models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration")
    
    print("Mock TT models registered successfully (mimicking offline_inference_tt.py pattern)!")


def check_tt_model_supported(model_name: str):
    """Mock function to check if TT model is supported."""
    print(f"Mock: TT model {model_name} is supported (mock mode)")
    return True


def setup_comprehensive_tt_mocks():
    """Set up comprehensive mocking that works across processes."""
    
    # Create a comprehensive mock ttnn module
    mock_ttnn = MockTTNN()
    
    # Mock ttnn and all related modules BEFORE any imports
    sys.modules['ttnn'] = mock_ttnn
    
    # Create a sitecustomize.py file that will be imported by Python automatically
    # This ensures our mocks are loaded in child processes
    import os
    import tempfile
    
    # Create the mock initialization code
    mock_init_code = f'''
# Auto-generated TT mocking code - loaded via sitecustomize.py
import sys
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

# Only activate mocking if we're in mock mode
if os.environ.get('VLLM_TT_MOCK_MODE') == '1':
    # Prevent real TT device initialization
    os.environ['TT_METAL_DEVICE_DISABLE'] = '1'
    os.environ['TTNN_DEVICE_DISABLE'] = '1'

    class MockTTDevice:
        def __init__(self, num_devices=2):
            self._num_devices = num_devices
        
        def get_num_devices(self):
            return self._num_devices

    class MockTTModel(nn.Module):
        \"\"\"Mock TT model that produces pseudo-random tokens efficiently.\"\"\"
        
        def __init__(self, config: PretrainedConfig, device: Any, max_num_seqs: int, 
                     max_seq_len: int, tt_data_parallel: int = 1):
            super().__init__()
            self.config = config
            self.device = device
            self.max_num_seqs = max_num_seqs
            self.max_seq_len = max_seq_len
            self.tt_data_parallel = tt_data_parallel
            
            # Model dimensions
            self.vocab_size = getattr(config, 'vocab_size', 32000)
            self.hidden_size = getattr(config, 'hidden_size', 4096)
            self.num_attention_heads = getattr(config, 'num_attention_heads', 32)
            self.num_key_value_heads = getattr(config, 'num_key_value_heads', 
                                              getattr(config, 'num_attention_heads', 32))
            self.head_dim = self.hidden_size // self.num_attention_heads
            self.num_layers = getattr(config, 'num_hidden_layers', 32)
            
            # For encoder-decoder models
            self.max_cross_attn_tokens = getattr(config, 'max_position_embeddings', 2048)
            
            # Random seed for consistent pseudo-random output
            self.generator = torch.Generator()
            self.generator.manual_seed(42)
            
            print(f\"MockTTModel initialized: vocab_size={{self.vocab_size}}, \"
                  f\"hidden_size={{self.hidden_size}}, num_layers={{self.num_layers}}\")
        
        @classmethod
        def initialize_vllm_model(cls, config: PretrainedConfig, device: Any, 
                                 max_num_seqs: int, max_seq_len: int, 
                                 tt_data_parallel: int = 1) -> 'MockTTModel':
            \"\"\"Initialize the mock TT model for vLLM.\"\"\"
            return cls(config, device, max_num_seqs, max_seq_len, tt_data_parallel)
        
        def allocate_kv_cache(self, kv_cache_shape: Tuple[int, ...], 
                             dtype: torch.dtype, num_layers: int) -> List[List[torch.Tensor]]:
            \"\"\"Allocate KV cache tensors.\"\"\"
            kv_cache = []
            for _ in range(num_layers):
                # Create K and V cache tensors
                cache_k = torch.zeros(kv_cache_shape, dtype=dtype, device=\"cpu\")
                cache_v = torch.zeros(kv_cache_shape, dtype=dtype, device=\"cpu\")
                kv_cache.append([cache_k, cache_v])
            return kv_cache
        
        def prefill_forward(self, tokens: torch.Tensor, page_table: torch.Tensor,
                           kv_cache: List[List[torch.Tensor]], prompt_lens: List[int],
                           empty_slots: Optional[List[int]] = None,
                           cross_page_table: Optional[torch.Tensor] = None,
                           sampling_params: Optional[Any] = None,
                           **kwargs) -> torch.Tensor:
            \"\"\"Mock prefill forward pass.\"\"\"
            batch_size = tokens.shape[0]
            max_seq_len = tokens.shape[1] if len(tokens.shape) > 1 else 1
            
            # Simulate some processing time
            time.sleep(0.001)  # 1ms delay to simulate computation
            
            if sampling_params is not None:
                # Return token IDs directly (sampling on device)
                return torch.randint(0, self.vocab_size, (batch_size,), 
                                   dtype=torch.long, generator=self.generator)
            else:
                # Return logits for host-side sampling
                return torch.randn(batch_size, max_seq_len, self.vocab_size, 
                                 generator=self.generator)
        
        def decode_forward(self, tokens: torch.Tensor, page_table: torch.Tensor,
                          kv_cache: List[List[torch.Tensor]], start_pos: torch.Tensor,
                          enable_trace: bool = True, read_from_device: bool = False,
                          cross_page_table: Optional[torch.Tensor] = None,
                          sampling_params: Optional[Any] = None,
                          prefill_cross_attention_masks: Optional[List] = None,
                          prefill_full_text_row_masked_out_mask: Optional[List] = None,
                          decode_cross_attention_masks: Optional[List] = None,
                          decode_full_text_row_masked_out_mask: Optional[List] = None,
                          rot_mats_all_users: Optional[List] = None,
                          **kwargs) -> torch.Tensor:
            \"\"\"Mock decode forward pass.\"\"\"
            batch_size = tokens.shape[0]
            
            # Simulate some processing time
            time.sleep(0.001)  # 1ms delay to simulate computation
            
            if sampling_params is not None:
                # Return token IDs directly (sampling on device)
                return torch.randint(0, self.vocab_size, (batch_size,), 
                                   dtype=torch.long, generator=self.generator)
            else:
                # Return logits for host-side sampling (last token only for decode)
                return torch.randn(batch_size, 1, self.vocab_size, 
                                 generator=self.generator)
        
        def read_decode_output(self, tt_out: torch.Tensor, 
                              async_read: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
            \"\"\"Mock reading decode output from device.\"\"\"
            if async_read:
                # Return tensor and mock read event
                mock_event = type('MockReadEvent', (), {{}})()
                return tt_out, [mock_event]
            else:
                return tt_out
        
        def process_decode_output_host(self, tt_out: torch.Tensor, 
                                      is_tokens: bool = False) -> torch.Tensor:
            \"\"\"Mock processing decode output on host.\"\"\"
            if is_tokens:
                # Already token IDs
                return tt_out
            else:
                # Convert logits to token IDs
                return torch.argmax(tt_out, dim=-1).squeeze(-1)

    class MockTTNN:
        @staticmethod
        def get_arch_name():
            return "wormhole_b0"
        
        @staticmethod
        def event_synchronize(event):
            pass
        
        @staticmethod
        def get_device_ids():
            return [0, 1]
        
        @staticmethod
        def open_mesh_device(*args, **kwargs):
            print("Mock: TT mesh device opened (child process)")
            return MockTTDevice(num_devices=2)
        
        @staticmethod
        def close_mesh_device(*args, **kwargs):
            print("Mock: TT mesh device closed (child process)")
            pass
        
        @staticmethod
        def ReadDeviceProfiler(*args, **kwargs):
            pass
        
        @staticmethod
        def set_fabric_config(*args, **kwargs):
            pass
        
        class DispatchCoreAxis:
            ROW = "row"
            COL = "col"
        
        class DispatchCoreConfig:
            def __init__(self, axis=None):
                self.axis = axis
        
        class FabricConfig:
            DISABLED = "disabled"
            FABRIC_1D = "fabric_1d"
            FABRIC_1D_RING = "fabric_1d_ring"
            FABRIC_2D = "fabric_2d"
            CUSTOM = "custom"
        
        class MeshShape:
            def __init__(self, *args):
                self.shape = args
        
        class cluster:
            class ClusterType:
                GALAXY = "galaxy"
            
            @staticmethod
            def get_cluster_type():
                return MockTTNN.cluster.ClusterType.GALAXY

    # Install mocks in sys.modules
    mock_ttnn = MockTTNN()
    sys.modules['ttnn'] = mock_ttnn

    # Mock all TT-related modules
    tt_modules = [
        'tt_lib', 'tt_eager', 'tt_metal', 'tt_cluster', 'tt_device',
        'ttnn.device', 'ttnn.operations', 'ttnn.core'
    ]
    for module_name in tt_modules:
        sys.modules[module_name] = MagicMock()

    # Create mock modules to mimic the real TT model paths
    class MockTTModelModule:
        MockTTModel = MockTTModel
        # Create aliases for all the model classes that real TT models would have
        LlamaForCausalLM = MockTTModel
        MllamaForConditionalGeneration = MockTTModel
        QwenForCausalLM = MockTTModel
        MistralForCausalLM = MockTTModel
        GemmaForCausalLM = MockTTModel
        Gemma2ForCausalLM = MockTTModel
        Gemma3ForConditionalGeneration = MockTTModel
    
    # Register mock modules to mimic the real TT model paths
    sys.modules['models'] = type('MockModule', (), {{}})()
    sys.modules['models.tt_transformers'] = type('MockModule', (), {{}})()
    sys.modules['models.tt_transformers.tt'] = type('MockModule', (), {{}})()
    sys.modules['models.tt_transformers.tt.generator_vllm'] = MockTTModelModule()

    # Register mock models with vLLM immediately at module level
    # This ensures they're available in all processes
    try:
        from vllm import ModelRegistry
        
        # Register models exactly like offline_inference_tt.py does
        # Llama3.1/3.2 - Text
        ModelRegistry.register_model(\"TTLlamaForCausalLM\", 
                                    \"models.tt_transformers.tt.generator_vllm:LlamaForCausalLM\")
        
        # Llama3.2 - Vision  
        ModelRegistry.register_model(\"TTMllamaForConditionalGeneration\",
                                    \"models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration\")
        
        # Qwen2.5 - Text
        ModelRegistry.register_model(\"TTQwen2ForCausalLM\", 
                                    \"models.tt_transformers.tt.generator_vllm:QwenForCausalLM\")
        ModelRegistry.register_model(\"TTQwen3ForCausalLM\", 
                                    \"models.tt_transformers.tt.generator_vllm:QwenForCausalLM\")
        
        # Mistral
        ModelRegistry.register_model(\"TTMistralForCausalLM\",
                                    \"models.tt_transformers.tt.generator_vllm:MistralForCausalLM\")
        
        # Gemma
        ModelRegistry.register_model(\"TTGemmaForCausalLM\",
                                    \"models.tt_transformers.tt.generator_vllm:GemmaForCausalLM\")
        ModelRegistry.register_model(\"TTGemma2ForCausalLM\", 
                                    \"models.tt_transformers.tt.generator_vllm:Gemma2ForCausalLM\")
        
        # Gemma3
        ModelRegistry.register_model(\"TTGemma3ForConditionalGeneration\",
                                    \"models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration\")
        
        print(\"Mock: TT models registered successfully at module level!\")
        
        # Debug: Check if models are registered
        supported_archs = ModelRegistry.get_supported_archs()
        tt_models = [arch for arch in supported_archs if arch.startswith('TT')]
        print(f\"Mock: TT models available: {{tt_models}}\")
        
    except ImportError as e:
        print(f\"Mock: Could not register models at module level: {{e}}\")

    # Hook into the import system to patch TT modules when they're loaded
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        # Always return our mock for ttnn
        if name == 'ttnn':
            return mock_ttnn
        
        try:
            module = original_import(name, *args, **kwargs)
        except ImportError as e:
            # If it's a TT-related module that fails, return a mock
            if any(tt_name in name for tt_name in ['tt_', 'ttnn']):
                print(f"Mock: Intercepted failed TT import: {{name}}")
                return MagicMock()
            raise e
        
        # Patch specific vLLM TT modules after they're imported
        if name == 'vllm.worker.tt_worker':
            print("Mock: Patching TT worker functions (child process)")
            module.ttnn = mock_ttnn
            module.open_mesh_device = lambda *args, **kwargs: MockTTDevice(num_devices=2)
            module.close_mesh_device = lambda *args, **kwargs: None
            module.get_num_available_blocks_tt = lambda *args, **kwargs: 1000
            
            # Mock all the device functions
            if hasattr(module, 'device_params_from_override_tt_config'):
                module.device_params_from_override_tt_config = lambda *args, **kwargs: {{}}
            if hasattr(module, 'set_fabric'):
                module.set_fabric = lambda *args, **kwargs: None
            if hasattr(module, 'reset_fabric'):
                module.reset_fabric = lambda *args, **kwargs: None
            if hasattr(module, 'get_dispatch_core_config'):
                module.get_dispatch_core_config = lambda *args, **kwargs: mock_ttnn.DispatchCoreConfig()
            if hasattr(module, 'get_fabric_config'):
                module.get_fabric_config = lambda *args, **kwargs: None
                
        elif name == 'vllm.worker.tt_model_runner':
            print("Mock: Patching TT model runner (child process)")
            module.ttnn = mock_ttnn
            
        elif name == 'vllm.platforms.tt':
            print("Mock: Patching TT platform (child process)")
            if hasattr(module, 'ttnn'):
                module.ttnn = mock_ttnn
        
        return module

    builtins.__import__ = mock_import

    print("Mock: TT device mocking initialized (child process)")
'''
    
    # Find a directory in sys.path to write sitecustomize.py
    site_dir = None
    for path in sys.path:
        if os.path.isdir(path) and os.access(path, os.W_OK):
            site_dir = path
            break
    
    if site_dir is None:
        # Create a temporary directory and add it to sys.path
        site_dir = tempfile.mkdtemp()
        sys.path.insert(0, site_dir)
    
    sitecustomize_path = os.path.join(site_dir, 'sitecustomize.py')
    
    # Write the sitecustomize.py file
    with open(sitecustomize_path, 'w') as f:
        f.write(mock_init_code)
    
    print(f"Mock: Created sitecustomize.py at {sitecustomize_path}")
    
    # Also set up mocking in the current process
    import builtins
    original_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'ttnn':
            return mock_ttnn
        
        try:
            module = original_import(name, *args, **kwargs)
        except ImportError as e:
            if any(tt_name in name for tt_name in ['tt_', 'ttnn']):
                print(f"Mock: Intercepted failed TT import: {name}")
                return MagicMock()
            raise e
        
        # Patch TT modules in parent process too
        if name == 'vllm.worker.tt_worker':
            print("Mock: Patching TT worker functions (parent process)")
            module.ttnn = mock_ttnn
            module.open_mesh_device = lambda *args, **kwargs: MockTTDevice(num_devices=2)
            module.close_mesh_device = lambda *args, **kwargs: None
            module.get_num_available_blocks_tt = lambda *args, **kwargs: 1000
            
        elif name == 'vllm.worker.tt_model_runner':
            print("Mock: Patching TT model runner (parent process)")
            module.ttnn = mock_ttnn
            
        elif name == 'vllm.platforms.tt':
            print("Mock: Patching TT platform (parent process)")
            if hasattr(module, 'ttnn'):
                module.ttnn = mock_ttnn
        
        return module
    
    builtins.__import__ = mock_import
    
    def cleanup():
        builtins.__import__ = original_import
        try:
            os.unlink(sitecustomize_path)
        except:
            pass
    
    return [cleanup]


def setup_tt_mocks():
    """Wrapper for backward compatibility."""
    return setup_comprehensive_tt_mocks()


def main():
    parser = argparse.ArgumentParser(description="Mock TT vLLM Server")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=32,
        help="Maximum number of sequences to be processed in a single iteration",
    )
    parser.add_argument(
        "--num_scheduler_steps", 
        type=int, 
        default=10, 
        help="Number of scheduler steps"
    )
    args, unknown_args = parser.parse_known_args()

    print(f"Starting mock TT vLLM server with model: {args.model}")
    print("Mock TT models will be registered automatically")
    
    # Set environment variables to indicate we're in mock mode
    import os
    os.environ['VLLM_TT_MOCK_MODE'] = '1'
    os.environ['TT_METAL_DEVICE_DISABLE'] = '1'
    os.environ['TTNN_DEVICE_DISABLE'] = '1'
    os.environ['VLLM_TARGET_DEVICE'] = 'tt'  # Force TT device selection
    
    # Set up comprehensive TT mocking BEFORE any vLLM imports
    patches = setup_tt_mocks()
    
    try:
        # Register mock TT models in parent process (like offline_inference_tt.py does)
        register_mock_tt_models()
        
        # Mock the TT model check
        check_tt_model_supported(args.model)
        
        # Prepare arguments for the API server
        server_args = [
            "--model", args.model,
            "--block_size", "64",
            "--max_num_seqs", str(args.max_num_seqs),
            "--num_scheduler_steps", str(args.num_scheduler_steps),
            # Note: Device selection is automatic based on platform detection
            # Our mocks ensure TT platform is detected and used
        ]
        
        # Add any additional arguments passed to the script
        server_args.extend(unknown_args)
        
        # Update sys.argv for the API server
        sys.argv = ["vllm.entrypoints.openai.api_server"] + server_args
        
        print(f"Starting vLLM API server with args: {server_args}")
        
        runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
        
    except Exception as e:
        print(f"Error starting server: {e}")
        print("This is expected in mock mode - the server may not start properly without real TT hardware")
        print("But the mock models have been registered and would work if the TT platform was available")
    finally:
        # Clean up patches and import hooks
        for cleanup_func in patches:
            try:
                cleanup_func()
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")

# Set up mocks and register models at module level (like offline_inference_tt.py)
patches = setup_tt_mocks()
register_mock_tt_models()

if __name__ == "__main__":
    main()