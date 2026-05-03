import pandas as pd
import pickle

# ======================
# モデル読み込み
# ======================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ======================
# 予想したいデータ読み込み
# （例：今日の出走表）
# ======================
df = pd.read_csv("predict_data.csv")

# ======================
# 特徴量（trainと同じ）
# ======================
features = [
    "競走得点",
    "勝率",
    "２連 対率",
    "３連 対率",
    "逃",
    "捲",
    "差",
    "マ",
    "年齢"
]

# 数値化
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# ======================
# 予想
# ======================
df["予想確率"] = model.predict_proba(df[features])[:, 1]

# 並び替え
df = df.sort_values("予想確率", ascending=False)

# 出力
print(df[["RaceID", "車", "選手名", "予想確率"]].head(10))