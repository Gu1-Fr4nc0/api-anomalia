import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# --- CONFIGURAÇÕES ---
# OBS: Ajuste o DATA_PATH para a pasta onde você descompactou o dataset MVTec Bottle
DATA_PATH = r"C:\Users\guifr\OneDrive\Área de Trabalho\Pen-Drive\Projetos\anomalia-mvtec\data\bottle"

if DATA_PATH is None:
    print("❌ ERRO: Não achei a pasta 'screw'. Edite o DATA_PATH no script manualmente!")
    exit()

print(f"📂 Usando dados em: {DATA_PATH}")

# --- CONFIGURAÇÕES ---
EPOCHS = 60 # Um pouco mais para garantir nitidez
BATCH_SIZE = 16
LR = 1e-3

# --- ARQUITETURA MODERNA (Upsample + Conv) ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder (Reduz a imagem)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 128x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # -> 64x64
            nn.Conv2d(32, 64, 3, padding=1),  # 64x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # -> 32x32
            nn.Conv2d(64, 128, 3, padding=1), # 32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                # -> 16x16
        )
        # Decoder (Aumenta a imagem SEM Xadrez)
        self.decoder = nn.Sequential(
            # Bloco 1: 16 -> 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            # Bloco 2: 32 -> 64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            # Bloco 3: 64 -> 128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid() # Saída entre 0 e 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Iniciando Retreino V3 (Anti-Aliasing) em: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Log mais limpo a cada 5 épocas
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.6f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/autoencoder.pth") # Salva direto no nome certo
    print("✅ Modelo V3 salvo! Copie para a pasta da API se necessário.")

if __name__ == "__main__":
    train()