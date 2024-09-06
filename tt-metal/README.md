
## Environment Creation 

To setup the tt-metal environment with vllm, follow the instructions in setup-metal.sh

## Accessing the Meta-Llama-3.1 Model

To run Meta-Llama-3.1, it is required to have access to the model on Hugging Face. 
Steps:
1. Request access on [https://huggingface.co/meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B).
2. Once you have received access, create and copy your access token from the settings tab on Hugging Face.
3. Run this code in python and paste your access token:
    ```python
    from huggingface_hub import notebook_login
    notebook_login()
    ```



