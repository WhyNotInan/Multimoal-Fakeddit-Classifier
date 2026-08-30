import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# 1. Hardware setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the CLIP model
print("Loading CLIP model...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# 3. Load your prototype dataset
df = pd.read_csv("prototype_5k.csv")
extracted_features = []

print("Starting offline extraction (This will take a few minutes)...")

# 4. Extract features safely without tracking gradients
with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Load image and text
            image_path = f"images/{row['id']}.jpg"
            image = Image.open(image_path).convert("RGB")
            text = str(row['clean_title'])

            # Process and move to GPU
            inputs = processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            outputs = model(**inputs)
            img_embed = outputs.image_embeds
            txt_embed = outputs.text_embeds

            # Calculate Cosine Similarity
            cos_sim = torch.nn.functional.cosine_similarity(img_embed, txt_embed)

            # Save the data
            extracted_features.append({
                "id": row['id'],
                "label_2way": row['2_way_label'],
                "label_6way": row['6_way_label'],
                "image_vector": img_embed.cpu().numpy().flatten(),
                "text_vector": txt_embed.cpu().numpy().flatten(),
                "cosine_similarity": cos_sim.cpu().item()
            })
        except Exception:
            pass # Gracefully skip any corrupted image files

# 5. Save to disk
torch.save(extracted_features, "fakeddit_features_5k.pt")
print("\nExtraction complete! Saved to fakeddit_features_5k.pt")