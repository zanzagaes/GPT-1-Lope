import torch
import pickle
from torch import nn

from model import GPT

import argparse
parser = argparse.ArgumentParser("Sample GPT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--prompt", help="Texto de estímulo del modelo", type=str, default = "\n", dest = "prompt")
parser.add_argument("--temp", "--temperature", help="Temperatura", type=float, default = 0.9, dest = "temperature")
parser.add_argument("-s", "--samples", help="Número de muestras generadas", type=int, default = 5, dest = "samples")
parser.add_argument("-t", "--tokens", help="Símbolos generados por muestra", type=int, default = 200, dest = "tokens")
parser.add_argument("--seed", help="Semilla del generador de números pseudoaleatorios", type=int, default = None)
parser.add_argument("--gpt2", help="Si está presente, se utiliza el modelo GPT-2", action="store_true", default = False)
args = parser.parse_args()

useGPT2 = args.gpt2

## Descomentar y fijar semilla para obtener resultados replicables
seed = args.seed 
if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Se usa la tarjeta gráfica si la hay, si no la cpu
device = "cpu"
# Precisión numérica de trabajo
dtype = 'float16'

# Iniciar modelo y cargar pesos
checkpoint_name = "checkpoint_gpt2.pt" if useGPT2 else "checkpoint.pt"
checkpoint = torch.load(checkpoint_name, map_location = device)
model = GPT(**checkpoint["model_args"], device = device)
weights = checkpoint["model"]
model.load_state_dict(weights)

# Modo de evaluación: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
model.eval()
# Si está activada, envía el modelo a la tarjeta gráfica
model.to(device)

# Cargar el tokenizador. En este caso, es solo un diccionario de caracteres a enteros
def load_encoder_gpt1():
    with open("meta.pkl", 'rb') as file:
        meta = pickle.load(file)
        encode_map, decode_map = meta['encode'], meta['decode']
    encode = lambda str: [encode_map[char] for char in str]
    decode = lambda res: ''.join([decode_map[num] for num in res])
    return encode, decode

def load_encoder_gpt2():
    from transformers import AutoTokenizer
    enc = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
    return enc.encode, enc.decode

encode, decode = load_encoder_gpt2() if useGPT2 else load_encoder_gpt1()

@torch.no_grad()
def generate_tokens(model, x, generated_tokens, temperature=0.8):
    for _ in range(generated_tokens):
        # recortar si se ha sobrepasado el tamaño de contexto
        x_cropped = x if x.size(1) <= model.block_size else x[:, -model.block_size:]
        logits = model(x_cropped)
        logits = logits[:, -1, :] / temperature
        probs = nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        yield next_token

prompt = args.prompt
max_new_tokens = args.tokens
num_samples = args.samples
start_ids = encode(prompt)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
with torch.no_grad():
    for _ in range(num_samples):
        print("\n==== Nueva muestra =====")
        if prompt != "\n":
            print(prompt, end="")
        for token in generate_tokens(model, x, max_new_tokens, temperature = args.temperature):
            print(decode(token[0].tolist()), end="")
