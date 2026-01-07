from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import ta
import os

app = Flask(__name__)
CORS(app)

print("Modeller yukleniyor...")
quality_model = joblib.load('model_quality.pkl')
sl_model = joblib.load('model_sl.pkl')
FEATURE_COLS = joblib.load('model_features.pkl')
params = joblib.load('optimal_params.pkl')
print("Modeller yuklendi!")

exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

def fetch_ohlcv(symbol, timeframe='1h', limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        return None

def calculate_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['atr_normalized'] = df['atr'] / df['close']
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bollinger_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
    df['std_dev_20'] = df['close'].rolling(window=20).std() / df['close']
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
    atr_sum = df['atr'].rolling(window=14).sum()
    high_low_diff = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
    df['choppiness'] = 100 * np.log10(atr_sum / high_low_diff.replace(0, 0.0001)) / np.log10(14)
    df['regime_encoded'] = (df['adx'] > 25).astype(int)
    df['confidence_score'] = 0.5 + (df['adx'] / 100) * 0.3
    return df

def predict_trade(symbol):
    df = fetch_ohlcv(symbol, '1h', 100)
    if df is None or len(df) < 50:
        return {'error': 'Data error', 'symbol': symbol}
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
        'indicators': {
            'rsi': round(float(latest['rsi']), 2) if pd.notna(latest['rsi']) else None,
            'adx': round(float(latest['adx']), 2) if pd.notna(latest['adx']) else None,
            'atr_pct': round(float(latest['atr_normalized']) * 100, 3) if pd.notna(latest['atr_normalized']) else None,
            'trend': 'TREND' if latest['regime_encoded'] == 1 else 'RANGE',
            'volume_ratio': round(float(latest['volume_ma_ratio']), 2) if pd.notna(latest['volume_ma_ratio']) else None
        }
    }

@app.route('/')
def home():
    return jsonify({'status': 'online', 'name': 'ML Trading Bot', 'coins': ['BTC/USDT', 'ETH/USDT']})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

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
