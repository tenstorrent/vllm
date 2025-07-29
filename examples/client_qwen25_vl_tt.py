import base64
import json

# Path to your image
image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

payload = {
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    }
                },
                {
                    "type": "text",
                    "text": "Is there a cat in this image? If not, what animal do you see in the image? Describe the image in detail."
                }
            ]
        }
    ],
    "max_tokens": 128,
}

# Save to a JSON file
with open("server-instruct-mm-prompt-demo.json", "w") as json_file:
    json.dump(payload, json_file, indent=4)