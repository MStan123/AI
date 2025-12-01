# https://huggingface.co/allenai/olmOCR-2-7B-1025

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import gc


# Clear cache
torch.cuda.empty_cache()
gc.collect()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("⚠️ WARNING: Running on CPU will be VERY slow (10-20 min)")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("allenai/olmOCR-2-7B-1025",
                                                           torch_dtype=torch.bfloat16,
                                                           device_map="auto"
                                                           ).eval()

processor = AutoProcessor.from_pretrained("allenai/olmOCR-2-7B-1025")

print(f"Model device: {next(model.parameters()).device}")

# Load your image
image_path = "numuna.jpg"
image = Image.open(image_path).convert("RGB")

# Build the prompt (no anchoring = best for clean documents like passports)
prompt = ('extract everything that you see in the picture of document, take into an account unique Azerbaijani letters such as Ə,Ö,Ü,Ğ,Ç,Ş,İ.')

# Messages format expected by olmOCR
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},  # First image placeholder
            {"type": "text", "text": prompt}
        ]
    }
]

# Important: Use the processor correctly for vision models
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# Process text + image together
inputs = processor(
    text=text_prompt,
    images=image,        # Pass PIL Image directly, not list if single
    return_tensors="pt"
).to(model.device)

inputs = {key: value.to(model.device) for (key, value) in inputs.items()}

import time
start_time = time.time()

# Generate the output
output = model.generate(
            **inputs,
            temperature=0.1,
            max_new_tokens=1500,
            num_return_sequences=1,
            do_sample=True,
        )

inference_time = time.time() - start_time
print(f"\n⏱️ Inference time: {inference_time:.2f} seconds")

# Decode the output
prompt_length = inputs["input_ids"].shape[1]
new_tokens = output[:, prompt_length:]
text_output = processor.tokenizer.batch_decode(
    new_tokens, skip_special_tokens=True
)

print(text_output)