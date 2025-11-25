import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import time
import random
from tqdm import tqdm

# Import your model definitions
from DAMSM import RNN_ENCODER, CNN_ENCODER

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'BATCH_SIZE': 48,
    'EPOCHS': 100,
    'LR': 0.0002,
    'EMBED_DIM': 256,
    'MAX_SEQ_LENGTH': 20,
    'IMAGE_SIZE': 299,
    'DATA_DIR': './dataset',
    'DICT_DIR': './dictionary',
    'SAVE_DIR': './DAMSMencoders',
    'GPU_ID': 0
}

if not os.path.exists(CONFIG['SAVE_DIR']):
    os.makedirs(CONFIG['SAVE_DIR'])

# Set Device
device = torch.device(f"cuda:{CONFIG['GPU_ID']}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# DATASET
# ==============================================================================
class TextImageDataset(Dataset):
    def __init__(self, data_path, dict_dir, max_len=20, img_size=299):
        self.df = pd.read_pickle(data_path)
        self.img_size = img_size
        self.max_len = max_len
        
        # Load Vocab
        self.word2Id = dict(np.load(os.path.join(dict_dir, 'word2Id.npy')))
        self.pad_id = int(self.word2Id.get('<PAD>', 0)) # Default to 0 if not found
        
        # Flatten dataset (Image, Caption)
        self.captions = []
        self.img_paths = []
        
        print("Processing dataset...")
        for idx, row in self.df.iterrows():
            img_path = row['ImagePath']
            caps = row['Captions']
            for cap in caps:
                self.captions.append(cap)
                self.img_paths.append(img_path)
                
        print(f"Total pairs: {len(self.captions)}")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image in case of error
            image = Image.new('RGB', (self.img_size, self.img_size))
            
        image = self.transform(image)
        
        # Process Caption
        cap = self.captions[idx]
        cap = [int(x) for x in cap]
        cap_len = len(cap)
        
        if cap_len < self.max_len:
            pad = [self.pad_id] * (self.max_len - cap_len)
            cap = cap + pad
        else:
            cap = cap[:self.max_len]
            cap_len = self.max_len
            
        return image, torch.LongTensor(cap), cap_len

def collate_fn(data):
    # Sort by caption length for pack_padded_sequence
    data.sort(key=lambda x: x[2], reverse=True)
    images, captions, cap_lens = zip(*data)
    
    images = torch.stack(images, 0)
    captions = torch.stack(captions, 0)
    cap_lens = torch.LongTensor(cap_lens)
    
    return images, captions, cap_lens

# ==============================================================================
# LOSS FUNCTION
# ==============================================================================
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def sent_loss(cnn_code, rnn_code, labels, gamma=10.0):
    # cnn_code: [B, Dim]
    # rnn_code: [B, Dim]
    # labels: [B] (0, 1, 2...)
    
    # Normalize
    cnn_code = torch.nn.functional.normalize(cnn_code, p=2, dim=1)
    rnn_code = torch.nn.functional.normalize(rnn_code, p=2, dim=1)
    
    # Similarity Matrix [B, B]
    scores = torch.mm(cnn_code, rnn_code.t())
    scores = scores * gamma
    
    # Cross Entropy
    loss0 = nn.CrossEntropyLoss()(scores, labels)
    loss1 = nn.CrossEntropyLoss()(scores.t(), labels)
    
    return loss0 + loss1

# ==============================================================================
# MAIN TRAINING
# ==============================================================================
def train():
    # Load Vocab Size
    # Determine vocab size from word2Id to ensure coverage of all IDs
    word2Id = dict(np.load(os.path.join(CONFIG['DICT_DIR'], 'word2Id.npy')))
    max_id = 0
    for v in word2Id.values():
        try:
            val = int(v)
            if val > max_id: max_id = val
        except: pass
    vocab_size = max_id + 1
    print(f"Vocab Size: {vocab_size} (Max ID: {max_id})")
    
    # Initialize Models
    text_encoder = RNN_ENCODER(ntoken=vocab_size, nhidden=256)
    image_encoder = CNN_ENCODER(nef=CONFIG['EMBED_DIM'])
    
    text_encoder = text_encoder.to(device)
    image_encoder = image_encoder.to(device)
    
    # Optimizers
    optimizer_text = optim.Adam(text_encoder.parameters(), lr=CONFIG['LR'], betas=(0.5, 0.999))
    optimizer_image = optim.Adam(image_encoder.parameters(), lr=CONFIG['LR'], betas=(0.5, 0.999))
    
    # Dataset
    dataset = TextImageDataset(
        os.path.join(CONFIG['DATA_DIR'], 'text2ImgData.pkl'),
        CONFIG['DICT_DIR']
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    print("Starting Training...")
    best_r_prec = 0.0
    
    for epoch in range(CONFIG['EPOCHS']):
        start_time = time.time()
        total_loss = 0
        steps = 0
        
        text_encoder.train()
        image_encoder.train()
        
        for images, captions, cap_lens in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device)
            
            # Zero Grads
            optimizer_text.zero_grad()
            optimizer_image.zero_grad()
            
            # Forward
            # Text: hidden is initialized inside forward if not provided? 
            # DAMSM.py init_hidden needs batch size
            hidden = text_encoder.init_hidden(images.size(0))
            words_emb, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_emb = words_emb.detach() # Stop grad for words part if not used in sent loss? 
            # Actually for sent_loss we need grads on sent_emb
            
            # Image
            local_img, global_img = image_encoder(images)
            
            # Loss
            labels = torch.arange(images.size(0)).to(device)
            loss = sent_loss(global_img, sent_emb, labels)
            
            # Backward
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 0.25)
            
            optimizer_text.step()
            optimizer_image.step()
            
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / steps
        
        # Evaluate
        r_prec = evaluate_r_precision(text_encoder, image_encoder, dataloader, device)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} | R-Precision: {r_prec:.2%} | Time: {time.time()-start_time:.1f}s")
        
        if r_prec > best_r_prec:
            best_r_prec = r_prec
            print("  >> New Best! Saving best models...")
            torch.save(text_encoder.state_dict(), os.path.join(CONFIG['SAVE_DIR'], 'text_encoder_best.pth'))
            torch.save(image_encoder.state_dict(), os.path.join(CONFIG['SAVE_DIR'], 'image_encoder_best.pth'))
        
        # Save Checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Saving checkpoints...")
            torch.save(text_encoder.state_dict(), os.path.join(CONFIG['SAVE_DIR'], f'text_encoder{epoch+1}.pth'))
            torch.save(image_encoder.state_dict(), os.path.join(CONFIG['SAVE_DIR'], f'image_encoder{epoch+1}.pth'))

def evaluate_r_precision(text_encoder, image_encoder, dataloader, device, num_batches=10):
    """
    Computes R-Precision (Batch-wise) on a subset of data.
    Checks if the correct caption is retrieved for each image within the batch.
    """
    text_encoder.eval()
    image_encoder.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, (images, captions, cap_lens) in enumerate(dataloader):
            if i >= num_batches: break
            
            images = images.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device)
            
            hidden = text_encoder.init_hidden(images.size(0))
            _, sent_emb = text_encoder(captions, cap_lens, hidden)
            _, global_img = image_encoder(images)
            
            # Normalize
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            global_img = torch.nn.functional.normalize(global_img, p=2, dim=1)
            
            # Similarity [B, B]
            sim = torch.mm(global_img, sent_emb.t())
            
            # Accuracy
            preds = torch.argmax(sim, dim=1)
            targets = torch.arange(images.size(0)).to(device)
            
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)
            
    return total_correct / total_samples if total_samples > 0 else 0.0

if __name__ == "__main__":
    train()
