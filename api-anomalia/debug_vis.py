import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from app.model import Autoencoder

# --- CONFIGURAÇÕES ---
# Ajuste para ONDE está UMA imagem boa e UMA ruim na sua pasta bottle agora
IMG_GOOD_PATH = r"C:\Users\guifr\OneDrive\Área de Trabalho\Pen-Drive\Projetos\anomalia-mvtec\data\bottle\test\good\000.png"
IMG_BAD_PATH = r"C:\Users\guifr\OneDrive\Área de Trabalho\Pen-Drive\Projetos\anomalia-mvtec\data\bottle\test\broken_large\001.png"
MODEL_PATH = "weights/autoencoder.pth" # Certifique-se que é o modelo NOVO

def visualize():
    device = torch.device("cpu")
    model = Autoencoder().to(device)
    
    # Carrega pesos
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Modelo carregado.")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Função auxiliar para processar
    def process(path, title, ax_orig, ax_recon, ax_heat):
        if not os.path.exists(path):
            print(f"Arquivo não encontrado: {path}")
            return
            
        img = Image.open(path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            recon = model(tensor)
            mse = nn.MSELoss()(recon, tensor).item()
            err = (recon - tensor).abs().mean(dim=1).squeeze()  # [H, W]
            err = (err - err.min()) / (err.max() - err.min() + 1e-8)
        
        # Plot Original
        ax_orig.imshow(img.resize((128,128)))
        ax_orig.set_title(f"{title}\nOriginal")
        ax_orig.axis('off')
        
        # Plot Reconstrução
        recon_img = recon.squeeze().permute(1, 2, 0).numpy()
        ax_recon.imshow(recon_img)
        ax_recon.set_title(f"Reconstrução\nScore: {mse:.5f}")
        ax_recon.axis('off')
        
        # Heatmap
        ax_heat.imshow(err.cpu().numpy(), cmap='inferno')
        ax_heat.set_title("Heatmap")
        ax_heat.axis('off')

    # Gera figura
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    process(IMG_GOOD_PATH, "PEÇA BOA", axes[0,0], axes[0,1], axes[0,2])
    process(IMG_BAD_PATH, "PEÇA DEFEITO", axes[1,0], axes[1,1], axes[1,2])
    
    plt.tight_layout()
    plt.savefig("diagnostico.png")
    print("📸 Diagnóstico salvo em 'diagnostico.png'. Abra para ver!")
    plt.show()

if __name__ == "__main__":
    visualize()
