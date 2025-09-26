import os
from groq import Groq
from pinecone import Pinecone, ServerlessSpec  # Nova versão (sem 'client')
from sentence_transformers import SentenceTransformer
import yfinance as yf
import pandas as pd
import ta
from flask import Flask, request, jsonify

# Pega as chaves das variáveis de ambiente (seguro!)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializa os serviços
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("trading-memory")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

def pegar_dados(ativo):
    mapa = {"GC": "GC=F", "ES": "^GSPC", "NQ": "^IXIC", "CL": "CL=F", "6E": "EURUSD=X"}
    if ativo not in mapa:
        return None, f"Ativo {ativo} não suportado."
    try:
        df = yf.download(mapa[ativo], interval='5m', period='1d')
        if df.empty: return None, "Sem dados"
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['ema20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df.fillna(method='ffill', inplace=True)
        return {
            "preco": round(df['Close'].iloc[-1], 2),
            "rsi": round(df['rsi'].iloc[-1], 2),
            "ema20": round(df['ema20'].iloc[-1], 2),
            "volume": int(df['Volume'].iloc[-1]) if 'Volume' in df else 0
        }, None
    except Exception as e:
        return None, str(e)

def analisar_com_ia(ativo):
    dados, erro = pegar_dados(ativo)
    if erro: return f"⚠️ {erro}"
    
    query = f"Regras e setups para {ativo} na Topstep"
    query_vetor = embedder.encode(query).tolist()
    resultado = index.query(vector=query_vetor, top_k=3, include_metadata=True)
    contexto = "\n".join([match['metadata']['conteudo'] for match in resultado['matches']])
    
    contexto_atual = f"""
Dados em tempo real:
- Preço: ${dados['preco']}
- RSI: {dados['rsi']}
- EMA20: ${dados['ema20']}
- Volume: {dados['volume']}

Regras e setups da Topstep:
{contexto}
"""
    
    resposta = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "Você é um trader profissional especializado em futuros da Topstep. Use os dados em tempo real e as regras da Topstep para dar uma análise prática."},
            {"role": "user", "content": f"Ativo: {ativo}\n\n{contexto_atual}\n\nDê uma recomendação clara: comprar, vender ou esperar? Inclua entrada, stop loss, take profit e risco."}
        ],
        temperature=0.7,
        max_tokens=600
    )
    return resposta.choices[0].message.content

@app.route('/analise')
def analise():
    ativo = request.args.get('ativo', 'GC')
    return jsonify({"analise": analisar_com_ia(ativo)})

@app.route('/')
def home():
    return "IA Trader Topstep está online! Use /analise?ativo=GC"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
