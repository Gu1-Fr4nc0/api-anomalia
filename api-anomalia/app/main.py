from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from contextlib import asynccontextmanager
from .services import service

# --- Lifespan (Gerenciamento de Ciclo de Vida) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Executa antes da API começar
    try:
        service.load_model()
    except Exception as e:
        print(f"❌ Erro fatal ao carregar modelo: {e}")
        # Em produção, você talvez queira impedir o app de subir
    yield
    # Executa quando a API desliga (limpeza)
    print("👋 Desligando API...")

app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)

# --- Endpoints ---

@app.get("/")
def home():
    return {"message": "Bem-vindo à API de Detecção de Anomalias Industrial"}

@app.get("/health")
def health_check():
    """Endpoint para monitoramento (K8s/AWS usam isso)"""
    if service.model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    return {"status": "ok", "service": "ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recebe imagem e retorna probabilidade de anomalia"""
    
    # Validação básica de tipo
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Apenas JPG e PNG são suportados")
    
    try:
        contents = await file.read()
        result = service.predict(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

@app.post("/heatmap")
async def heatmap(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Apenas JPG e PNG são suportados")
    try:
        contents = await file.read()
        img_bytes = service.predict_heatmap(contents)
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
