import streamlit as st
import pandas as pd
import pickle

st.title("🚴 競輪AI予想")

# ======================
# モデル読み込み
# ======================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ======================
# CSVアップロード
# ======================
uploaded_file = st.file_uploader("出走表CSVをアップロード", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

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

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # 予想
    df["予想確率"] = model.predict_proba(df[features])[:, 1]
    df = df.sort_values("予想確率", ascending=False)

    st.subheader("🔥 予想結果")
    st.dataframe(df[["RaceID", "車", "選手名", "予想確率"]])