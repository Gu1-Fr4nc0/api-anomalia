import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from .model import Autoencoder

# Configurações Globais
MODEL_PATH = "weights/autoencoder.pth"
DEVICE = torch.device("cpu") # API geralmente roda em CPU para inferência leve

# Threshold ajustado empiricamente para a categoria 'bottle'
# Valores típicos: Good (~0.0015) vs Bad (~0.0030)
THRESHOLD = 0.002 # Ajuste conforme seus testes anteriores (limiar de defeito)

class AnomalyService:
    def __init__(self):
        self.model = None
        self.criterion = nn.MSELoss()
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def load_model(self):
        """Carrega o modelo e coloca em modo de avaliação (Eval)"""
        print("📥 Carregando modelo...")
        self.model = Autoencoder().to(DEVICE)
        # map_location garante que carregue na CPU mesmo se treinou em GPU
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval() # CRÍTICO: Desliga dropout/batchnorm training
        print("✅ Modelo carregado e pronto!")

    def predict(self, image_bytes):
        """Recebe bytes da imagem, processa e retorna score"""
        if self.model is None:
            raise RuntimeError("Modelo não carregado!")

        # 1. Pré-processamento
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        

        # 2. Inferência
        with torch.no_grad():
            reconstructed = self.model(input_tensor)
            loss = self.criterion(reconstructed, input_tensor)
        
        score = loss.item()
        
        return {
            "anomaly_score": score,
            "threshold": THRESHOLD,
            "is_anomaly": score > THRESHOLD
        }

    def predict_heatmap(self, image_bytes):
        if self.model is None:
            raise RuntimeError("Modelo não carregado!")
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            reconstructed = self.model(input_tensor)
        err = (reconstructed - input_tensor).abs().mean(dim=1)  # [1, H, W]
        err_min = err.min()
        err_max = err.max()
        norm = (err - err_min) / (err_max - err_min + 1e-8)
        heat = (norm.squeeze().cpu().numpy() * 255.0).astype("uint8")
        heat_img = Image.fromarray(heat, mode="L")
        buf = io.BytesIO()
        heat_img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

# Instância global para ser importada
service = AnomalyService()
