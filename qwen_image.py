from diffusers import DiffusionPipeline
from torchviz import make_dot
import torch
from torchinfo import summary

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.float16)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}
# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

negative_prompt = " " # using an empty string if you do not have specific concept to remove

# Print model summary - first let's see what's available in the pipeline
print("Pipeline components:")
print([attr for attr in dir(pipe) if not attr.startswith('_')])

# Print model summary of the main model
if hasattr(pipe, 'transformer'):
    print("\nTransformer Model Summary:")
    # Get the model's dtype and device
    model_dtype = next(pipe.transformer.parameters()).dtype
    model_device = next(pipe.transformer.parameters()).device
    print(f"Model dtype: {model_dtype}, device: {model_device}")
    
    # Create proper input data for the transformer
    # The transformer expects hidden_states, timestep, and encoder_hidden_states
    batch_size = 1
    seq_len = 77
    hidden_size = 64  # This might need adjustment based on actual model
    
    # Create input tensors with proper dtype and device
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=model_dtype, device=model_device)
    timestep = torch.tensor([0], dtype=torch.long, device=model_device)
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=model_dtype, device=model_device)
    
    try:
        print(summary(pipe.transformer, input_data=(hidden_states, timestep, encoder_hidden_states)))
    except Exception as e:
        print(f"Error with full input: {e}")
        # Try with just hidden_states if the full input fails
        print("Trying with just hidden_states...")
        print(summary(pipe.transformer, input_data=hidden_states))
        
elif hasattr(pipe, 'model'):
    print("\nModel Summary:")
    model_dtype = next(pipe.model.parameters()).dtype
    model_device = next(pipe.model.parameters()).device
    print(f"Model dtype: {model_dtype}, device: {model_device}")
    print(summary(pipe.model, device=model_device))
else:
    print("\nPipeline type:", type(pipe))
    print("Available models in pipeline:")
    for attr in dir(pipe):
        if hasattr(getattr(pipe, attr), 'parameters'):
            print(f"- {attr}: {type(getattr(pipe, attr))}")


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
).images[0]


# Create the graph
dot = make_dot(image, params=dict(pipe.named_parameters()))
dot.render("qwen_unet.gv", format="pdf")