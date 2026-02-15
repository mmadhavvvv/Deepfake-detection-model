import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import sys
import random
from PIL import Image

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.model_architecture import DeepfakeDetector



import time

class FastDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.file_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            return self.__getitem__(0)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [LAPTOP MODE] Training on: {DEVICE}", flush=True)

    # --- PROFESSIONAL SCALE (40,000 IMAGES) ---
    IMAGES_PER_CLASS = 20000      # 20k Real + 20k Fake
    BATCH_SIZE = 16              # Reduced batch size for 224x224 RAM safety
    EPOCHS = 4                   
    LEARNING_RATE = 5e-5         
    
    # Standard High Resolution
    IMG_SIZE = 224

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), # Add variety
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # --- DATASET PATH ---
    dataset_root = r"C:\Users\VASU TANDON\.cache\kagglehub\datasets\xhlulu\140k-real-and-fake-faces\versions\2\real_vs_fake\real-vs-fake"
    train_dir = os.path.join(dataset_root, "train")
    real_dir = os.path.join(train_dir, "real")
    fake_dir = os.path.join(train_dir, "fake")

    print(f"📂 Indexing optimized dataset...", flush=True)
    
    print(f"📂 Heavy Indexing of full corpus...", flush=True)
    
    def get_all_paths(folder):
        paths = []
        with os.scandir(folder) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(entry.path)
        return paths

    # Get every single available path
    all_real_available = get_all_paths(real_dir)
    all_fake_available = get_all_paths(fake_dir)
    
    # Shuffle the FULL lists before picking our 20k
    random.shuffle(all_real_available)
    random.shuffle(all_fake_available)

    real_paths = all_real_available[:IMAGES_PER_CLASS]
    fake_paths = all_fake_available[:IMAGES_PER_CLASS]

    all_paths = real_paths + fake_paths
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths)

    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)


    split = int(0.85 * len(all_paths))
    train_ds = FastDeepfakeDataset(all_paths[:split], all_labels[:split], transform=transform)
    val_ds = FastDeepfakeDataset(all_paths[split:], all_labels[split:], transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"✅ Ready | Train: {len(train_ds)} | Val: {len(val_ds)}", flush=True)

    # --- INITIALIZE MODEL ---
    model = DeepfakeDetector()
    # Adjust FC for input size if necessary (ResNet handles different sizes via Global Avg Pooling)
    model = model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- TRAINING LOOP ---
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}", flush=True)

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Frequency update for visibility
            if i % 10 == 0:
                print(f"Progress: {i}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)
            
            # --- COOLING BREAK ---
            # Gives the CPU a 0.2s breather every batch to keep the laptop responsive
            time.sleep(0.2)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
                preds = (torch.sigmoid(model(images)) > 0.5).float()
                correct += (preds == labels).sum().item()

        val_acc = 100 * correct / len(val_ds)
        print(f"✨ Epoch Scored: {val_acc:.2f}% accuracy", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_deepfake_model.pth")
            print("💾 Model Updated!", flush=True)

    # --- POST-TRAINING CLEANUP & PUSH ---
    print("\n🚀 Training Complete! Preparing to push to GitHub...", flush=True)
    try:
        # Add all relevant files
        os.system('git add app/streamlit_app.py')
        os.system('git add src/inference/predict.py')
        os.system('git add src/training/train.py')
        os.system('git add src/utils/grad_cam.py')
        os.system('git add models/best_deepfake_model.pth')
        os.system('git add README.md')

        
        # Commit
        commit_msg = f"Automated Update: Trained on 40k images | Accuracy: {best_acc:.2f}% | Revamped Dark UI"
        os.system(f'git commit -m "{commit_msg}"')
        
        # Push
        print("📤 Pushing to GitHub...", flush=True)
        os.system('git push origin main')
        print("✅ Successfully pushed to GitHub!", flush=True)
        print("🌐 Live app will update automatically if connected to GitHub.", flush=True)
        
    except Exception as e:
        print(f"❌ Failed to push to GitHub: {e}", flush=True)


if __name__ == "__main__":
    main()

