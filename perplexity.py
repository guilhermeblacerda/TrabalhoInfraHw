import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import math
import gc
import warnings

# Silenciar os avisos chatos para o terminal ficar limpo
warnings.filterwarnings("ignore")

model_name = "gpt2"
prompt = "Explain artificial intelligence"

print("Inicializando o tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(prompt, return_tensors="pt")

# Mover o input para a GPU se ela estiver disponível
if torch.cuda.is_available():
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

# Dicionário para guardar os resultados
resultados = {}

# ----------------------------------------------------
# 1. Executando a versão FP32
# ----------------------------------------------------
print("\n[1/3] Calculando Perplexity para FP32 (Baseline)...")
try:
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        resultados["FP32"] = math.exp(outputs.loss.item())
    
    # Limpar memória da GPU/RAM antes do próximo teste
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
except Exception as e:
    resultados["FP32"] = f"Erro: {e}"

# ----------------------------------------------------
# 2. Executando a versão FP16
# ----------------------------------------------------
print("[2/3] Calculando Perplexity para FP16...")
if torch.cuda.is_available():
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            resultados["FP16"] = math.exp(outputs.loss.item())
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        resultados["FP16"] = f"Erro: {e}"
else:
    resultados["FP16"] = "Não executado (Requer GPU com CUDA)"

# ----------------------------------------------------
# 3. Executando a versão INT8
# ----------------------------------------------------
print("[3/3] Calculando Perplexity para INT8...")
try:
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        resultados["INT8"] = math.exp(outputs.loss.item())
    
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
except Exception as e:
    resultados["INT8"] = f"Erro (Verifique se o bitsandbytes está instalado): {e}"


# ----------------------------------------------------
# IMPRESSÃO DOS RESULTADOS FINAIS
# ----------------------------------------------------
print("\n" + "="*40)
print("         RELATÓRIO DE PERPLEXITY        ")
print("="*40)
for precisao, valor in resultados.items():
    if isinstance(valor, float):
        print(f" -> {precisao}: {valor:.4f}")
    else:
        print(f" -> {precisao}: {valor}")
print("="*40)
