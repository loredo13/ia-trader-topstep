import os
from flask import Flask, request, jsonify

# ---- IA / Vetores ----
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ---- Dados de mercado (AV -> fallback Yahoo) ----
import requests
import pandas as pd
import ta
import yfinance as yf

# ======================
# Config / chaves
# ======================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
ALPHA_KEY        = os.getenv("ALPHA_VANTAGE_KEY", "")  # opcional: se tiver, usamos AV

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY não definido.")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY não definido.")

# ======================
# Pinecone (SDK novo)
# ======================
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "trading-memory"
# cria o índice se não existir (dimension 384 = all-MiniLM-L6-v2)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ======================
# Embedder + Groq
# ======================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

# ======================
# Dados de mercado
# ======================
# Proxies via ETF (cobrem intraday free na Alpha Vantage)
MAPA_ETF = {
    "ES": "SPY",   # S&P500
    "NQ": "QQQ",   # Nasdaq100
    "CL": "USO",   # WTI
    "GC": "GLD",   # Ouro
    "6E": "FXE",   # Euro
}

# Yahoo como fallback (nem sempre traz volume/colunas em 5m)
MAPA_YF = {"GC": "GC=F", "ES": "^GSPC", "NQ": "^IXIC", "CL": "CL=F", "6E": "EURUSD=X"}


def _av_stock_intraday(symbol: str, interval: str = "5min"):
    """Alpha Vantage intraday via ETF (se ALPHA_VANTAGE_KEY estiver setada)."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "compact",
        "datatype": "json",
        "apikey": ALPHA_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    series = data.get("Time Series (5min)")
    if not series:
        raise RuntimeError(data.get("Note") or data.get("Error Message") or "Sem dados (AV).")
    df = pd.DataFrame(series).T.rename(columns={
        "1. open":"open","2. high":"high","3. low":"low","4. close":"close","5. volume":"volume"
    })
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df


def _yf_intraday(symbol: str):
    """Yahoo intraday 5m 1d como plano B."""
    df = yf.download(symbol, interval="5m", period="1d", progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError("Yahoo sem dados.")
    # normaliza nomes
    out = pd.DataFrame({
        "open":  df["Open"],
        "high":  df["High"],
        "low":   df["Low"],
        "close": df["Close"],
        "volume": df["Volume"] if "Volume" in df.columns else 0,
    })
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


def pegar_dados(ativo):
    """Tenta Alpha Vantage (ETF). Se não houver chave/erro, cai para Yahoo."""
    try:
        if ativo in MAPA_ETF and ALPHA_KEY:
            df = _av_stock_intraday(MAPA_ETF[ativo], "5min")
            fonte = "Alpha Vantage"
        else:
            df = _yf_intraday(MAPA_YF[ativo])
            fonte = "Yahoo Finance"

        if df.empty or "close" not in df.columns:
            return None, "Sem dados."

        # indicadores
        rsi  = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
        ema20 = ta.trend.EMAIndicator(close=df["close"], window=20).ema_indicator()
        df["rsi"], df["ema20"] = rsi, ema20
        df = df.ffill().dropna()

        row = df.iloc[-1]
        return {
            "preco": round(float(row["close"]), 2),
            "rsi": round(float(row["rsi"]), 2),
            "ema20": round(float(row["ema20"]), 2),
            "volume": int(row["volume"]) if "volume" in df.columns and pd.notna(row["volume"]) else 0,
            "ts": df.index[-1].isoformat(),
            "fonte": fonte,
        }, None

    except Exception as e:
        return None, f"Erro dados: {e}"


def analisar_com_ia(ativo):
    dados, erro = pegar_dados(ativo)
    if erro:
        return f"⚠️ {erro}"

    # busca memórias no Pinecone
    query = f"Regras e setups para {ativo} na Topstep"
    q_vec = embedder.encode(query).tolist()
    res = index.query(vector=q_vec, top_k=3, include_metadata=True)

    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    contexto = "\n".join([m["metadata"]["conteudo"] for m in matches if "metadata" in m and "conteudo" in m["metadata"]])

    contexto_atual = (
        f"Dados em tempo real ({dados['fonte']}):\n"
        f"- Preço: ${dados['preco']}\n"
        f"- RSI: {dados['rsi']}\n"
        f"- EMA20: ${dados['ema20']}\n"
        f"- Volume: {dados['volume']}\n\n"
        f"Regras/Setups relevantes:\n{contexto}\n"
    )

    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um trader profissional especializado em Topstep. "
                    "Use os dados atuais + regras para dar uma recomendação objetiva."
                ),
            },
            {
                "role": "user",
                "content": f"Ativo: {ativo}\n\n{contexto_atual}\n"
                           f"Diga: comprar, vender ou esperar. Inclua entrada, stop, alvo e risco por trade."
            },
        ],
        temperature=0.7,
        max_tokens=600,
    )
    return resp.choices[0].message.content


@app.route("/")
def home():
    return "IA Trader Topstep online! Use /analise?ativo=GC"

@app.route("/analise")
def analise():
    ativo = request.args.get("ativo", "GC").upper()
    if ativo not in MAPA_ETF and ativo not in MAPA_YF:
        return jsonify({"erro": f"Ativo {ativo} não suportado."}), 400
    try:
        texto = analisar_com_ia(ativo)
        return jsonify({"analise": texto})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    # Render expõe a PORT via env var
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
