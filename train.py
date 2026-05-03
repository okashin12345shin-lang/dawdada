import pandas as pd
from xgboost import XGBClassifier
import pickle

# ======================
# データ読み込み
# ======================
df = pd.read_csv("keirin_data.csv")

# ======================
# 特徴量
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

# 数値変換（重要）
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

X = df[features]
y = df["result"]

# ======================
# モデル
# ======================
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8
)

model.fit(X, y)

# ======================
# 保存
# ======================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("モデル完成")