from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import urllib.request
import json

app = Flask(__name__)
CORS(app)

print("Modeller yukleniyor...")
quality_model = joblib.load('model_quality.pkl')
sl_model = joblib.load('model_sl.pkl')
FEATURE_COLS = joblib.load('model_features.pkl')
params = joblib.load('optimal_params.pkl')
print("Modeller yuklendi!")

def fetch_binance_data(symbol):
    try:
        symbol_clean = symbol.replace('/', '')
        url = "https://api.binance.com/api/v3/klines?symbol=" + symbol_clean + "&interval=1h&limit=100"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        df.set_index('timestamp', inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

def fetch_coingecko_data(coin_id):
    try:
        url = "https://api.coingecko.com/api/v3/coins/" + coin_id + "/ohlc?vs_currency=usd&days=7"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['volume'] = 1000000.0
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        df.set_index('timestamp', inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

def get_data(symbol):
    df, error1 = fetch_binance_data(symbol)
    if df is not None and len(df) >= 50:
        return df, 'binance', None
    coin_id = 'bitcoin' if 'BTC' in symbol else 'ethereum'
    df, error2 = fetch_coingecko_data(coin_id)
    if df is not None and len(df) >= 50:
        return df, 'coingecko', None
    return None, None, "Binance: " + str(error1) + ", CoinGecko: " + str(error2)

def calculate_indicators(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_normalized'] = df['atr'] / df['close']
    df['adx'] = 25.0
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bollinger_width'] = (4 * std20) / df['close']
    df['std_dev_20'] = std20 / df['close']
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ma_ratio'] = df['volume'] / df['volume_ma']
    df['volume_spike'] = (df['volume'] > df['volume_ma'] * 2).astype(int)
    df['recent_high'] = df['high'].rolling(window=20).max()
    df['recent_low'] = df['low'].rolling(window=20).min()
    df['resistance_distance'] = (df['recent_high'] - df['close']) / df['close']
    df['support_distance'] = (df['close'] - df['recent_low']) / df['close']
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['candle_body_ratio'] = df['candle_body'] / df['candle_range'].replace(0, 0.0001)
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['trend_strength'] = (df['ema_20'] - df['ema_50']) / df['close']
    df['rolling_max'] = df['close'].rolling(window=20).max()
    df['recent_drawdown'] = (df['rolling_max'] - df['close']) / df['rolling_max']
    df['choppiness'] = 50.0
    df['regime_encoded'] = 0
    df['confidence_score'] = 0.5
    return df

def predict_trade(symbol):
    df, source, error = get_data(symbol)
    if df is None:
        return {'error': error, 'symbol': symbol}
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    features = {}
    for col in FEATURE_COLS:
        features[col] = float(latest[col]) if col in df.columns and pd.notna(latest[col]) else 0.0
    X = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
    ml_confidence = float(quality_model.predict_proba(X)[0, 1])
    optimal_sl = max(float(sl_model.predict(X)[0]), 0.005)
    if ml_confidence >= 0.85:
        quality = 'EXCELLENT'
    elif ml_confidence >= 0.75:
        quality = 'HIGH'
    elif ml_confidence >= 0.65:
        quality = 'MEDIUM'
    else:
        quality = 'LOW'
    should_trade = ml_confidence >= 0.70
    return {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'price': float(latest['close']),
        'ml_confidence': round(ml_confidence, 4),
        'quality': quality,
        'should_trade': should_trade,
        'signal': 'BUY' if should_trade else 'WAIT',
        'recommended_sl_pct': round(optimal_sl * 100, 2),
        'recommended_tp_pct': round(optimal_sl * 200, 2),
        'data_source': source,
        'indicators': {
            'rsi': round(float(latest['rsi']), 2) if pd.notna(latest['rsi']) else None,
            'adx': round(float(latest['adx']), 2) if pd.notna(latest['adx']) else None,
            'atr_pct': round(float(latest['atr_normalized']) * 100, 3) if pd.notna(latest['atr_normalized']) else None,
            'trend': 'RANGE',
            'volume_ratio': round(float(latest['volume_ma_ratio']), 2) if pd.notna(latest['volume_ma_ratio']) else None
        }
    }

@app.route('/')
def home():
    return jsonify({'status': 'online', 'name': 'ML Trading Bot', 'coins': ['BTC/USDT', 'ETH/USDT']})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/debug')
def debug():
    results = {}
    try:
        url = "https://api.binance.com/api/v3/ping"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            results['binance_ping'] = 'OK'
    except Exception as e:
        results['binance_ping'] = str(e)
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=5"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            results['binance_klines'] = 'OK - ' + str(len(data)) + ' candles'
    except Exception as e:
        results['binance_klines'] = str(e)
    try:
        url = "https://api.coingecko.com/api/v3/ping"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            results['coingecko_ping'] = 'OK'
    except Exception as e:
        results['coingecko_ping'] = str(e)
    return jsonify(results)

@app.route('/signal/<symbol>')
def get_signal(symbol):
    symbol = symbol.upper()
    if 'BTC' in symbol:
        symbol = 'BTC/USDT'
    elif 'ETH' in symbol:
        symbol = 'ETH/USDT'
    else:
        return jsonify({'error': 'Only BTC and ETH supported'}), 400
    return jsonify(predict_trade(symbol))

@app.route('/signals')
def get_all_signals():
    btc = predict_trade('BTC/USDT')
    eth = predict_trade('ETH/USDT')
    signals = [btc, eth]
    tradeable = [s for s in signals if s.get('should_trade', False)]
    return jsonify({'timestamp': datetime.now().isoformat(), 'signals': signals, 'tradeable': tradeable})

@app.route('/position', methods=['POST'])
def calc_position():
    data = request.get_json() or {}
    capital = data.get('capital', 10000)
    sl_pct = data.get('sl_pct', 1.5)
    risk = capital * 0.01
    pos = risk / (sl_pct / 100)
    margin = pos / 10
    return jsonify({'position_size': round(pos, 2), 'margin': round(margin, 2), 'risk': round(risk, 2)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
