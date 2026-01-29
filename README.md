# 🏭 Industrial Anomaly Detection API (MVTec AD)

<div align="center">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Task-Unsupervised_Anomaly_Detection-blue?style=for-the-badge" alt="Task">
</div>

## 📋 Sobre o Projeto

Este projeto implementa um sistema de **Inspeção Visual Automatizada** focado em ambientes industriais onde a coleta de dados de defeitos é difícil ou custosa.

Diferente de sistemas de classificação supervisionados, esta solução utiliza **Aprendizado Não-Supervisionado** para aprender a representação latente de peças manufaturadas normais (foco atual: **Garrafas**) e identificar anomalias com base em desvios na reconstrução aprendida.

O sistema foi validado utilizando o dataset **MVTec AD**, amplamente reconhecido como benchmark acadêmico e industrial para inspeção visual, permitindo comparação direta com abordagens de estado da arte.

### 💡 Decisão de Arquitetura: Por que Autoencoders?
A escolha por uma arquitetura **Autoencoder Convolucional** baseia-se na realidade do chão de fábrica:
* **Escassez de Defeitos:** Em linhas de produção eficientes, defeitos são eventos raros. Treinar um classificador supervisionado exigiria milhares de exemplos de defeitos que muitas vezes não existem.
* **Variabilidade:** Novos tipos de defeitos podem surgir a qualquer momento. O Autoencoder detecta "qualquer coisa que foge do padrão aprendido", garantindo robustez contra falhas inéditas.

---

## 🔄 Ciclo de Vida & Pipeline

O projeto simula um pipeline de ML simplificado, focado na reprodutibilidade e na separação entre treino e inferência:

1.  **Treinamento Offline (`retrain.py`):**
    * Consome dados brutos (imagens) do diretório local.
    * Executa o treinamento do Autoencoder (Encoder + Decoder).
    * Gera e serializa o artefato do modelo (`autoencoder.pth`).
2.  **Gerenciamento de Artefato:**
    * O arquivo de pesos (`.pth`) é tratado como um artefato imutável.
    * O threshold de decisão é calibrado nesta etapa com base no conjunto de validação.
3.  **Serving (API):**
    * A aplicação FastAPI carrega o artefato em memória durante o evento de `startup`.
    * A inferência ocorre em tempo real, sem re-treinamento durante a operação.

---

## ⚙️ Funcionamento Técnico

O modelo atua como um compressor e reconstrutor de imagens:
1.  **Encoder:** Reduz a imagem de entrada a um vetor latente (Bottleneck), forçando o modelo a aprender as características essenciais da peça.
2.  **Decoder:** Tenta reconstruir a imagem original a partir desse vetor.
3.  **Cálculo de Score:** O sistema calcula o **Erro Quadrático Médio (MSE)** entre a entrada e a reconstrução.
    * *Peça Boa:* O erro é baixo (reconstrução fiel).
    * *Anomalia:* O erro é alto (o modelo falha em reconstruir defeitos que nunca viu durante o treino).

> **Destaque de Engenharia:** O decoder utiliza camadas de `Upsample` + `Conv2d` (em vez de ConvTranspose) para eliminar artefatos visuais ("checkerboard artifacts") que poderiam introduzir ruído no cálculo do score e gerar falsos positivos.

---

## 📷 Resultados Experimentais

### 1. Diagnóstico Visual
Comparação entre entrada e saída. Note que o modelo "suaviza" ou remove o defeito na reconstrução, gerando um resíduo mensurável que aciona o alerta.

<img src="https://github.com/user-attachments/assets/8527f724-77d0-4ff5-873f-893662184766" alt="Diagnóstico Visual" width="1400">

### 2. Exemplo de Resposta da API
Detecção de uma garrafa contaminada com score acima do limiar seguro.

<img src="https://github.com/user-attachments/assets/87aa2d6c-555c-4503-aaba-57154e057620" alt="API Response" width="1394">

---

## 🚀 Como Rodar

### 1. Instalação
```bash
git clone [https://github.com/Gu1-Fr4nc0/api-anomalia](https://github.com/Gu1-Fr4nc0/api-anomalia)
cd industrial-anomaly-api
pip install -r requirements.txt
```

### 2. Configuração dos Dados
Este projeto utiliza a categoria bottle do dataset MVTec AD.

Baixe os dados no site oficial da MVTec.

Extraia para a pasta data/bottle na raiz do projeto (necessário apenas para retreino).

### 3. Execução do Serviço
```bash
uvicorn app.main:app --reload
```
A API estará ativa em http://127.0.0.1:8000.

## 🔌 API Reference
POST /predict
Endpoint síncrono para inferência online de imagens individuais.

Request: multipart/form-data (Arquivo de imagem)

Response (JSON):

```bash
{
  "anomaly_score": 0.00345,
  "threshold": 0.002,
  "is_anomaly": true
}
```
Sobre o Threshold (0.002)
O limiar de decisão não é arbitrário. Ele foi definido empiricamente analisando a distribuição de erros no conjunto de validação de peças normais (aprox. percentil 95), visando minimizar falsos positivos em um ambiente de produção conservador.

## ⚠️ Limitações Conhecidas
Como todo sistema de ML, existem fronteiras operacionais:

Sensibilidade à Iluminação: O modelo assume condições de luz controladas (padrão industrial). Mudanças drásticas de brilho podem elevar o erro de reconstrução incorretamente (Domain Shift).

Calibração Específica: O threshold atual é otimizado para a categoria bottle. Novos objetos (hazelnut, screw) exigem recalibração do limiar devido às diferenças na textura e complexidade da imagem.

Defeitos Globais vs Locais: O uso de MSE global funciona bem para defeitos estruturais, mas pode diluir defeitos muito pequenos (ex: micro-riscos) se a resolução da imagem for muito alta.

## 🔮 Extensibilidade
O pipeline foi projetado para ser agnóstico à categoria. O script retrain.py permite adaptar o sistema para outros objetos do MVTec AD ou dados proprietários com ajustes mínimos nos hiperparâmetros, simulando um pipeline de adaptação rápida.

<div align="center">
  
Desenvolvido por Guilherme Pança Franco Machine Learning Engineer | Computer Vision | Industrial AI & Anomaly Detection

</div>
