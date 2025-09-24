import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import altair as alt

from data import generate_data


# --- 專案設定 ---
st.set_page_config(page_title="互動式線性迴歸視覺化工具", page_icon="📊", layout="wide")


# --- CRISP-DM 流程 ---

# 1. 商業理解 (Business Understanding)
st.header("1. 商業理解")
st.markdown(
    """
在這個專案中，我們的目標是建立一個互動式的網頁應用程式，讓使用者可以透過視覺化的方式來理解簡易線性迴歸。

目標:
- 讓使用者可以動態調整參數 (斜率 `a`、截距 `b`、雜訊 `noise`、資料點數量 `n`)。
- 即時視覺化資料點和迴歸線的變化。
- 透過 CRISP-DM 的框架來解釋整個流程。
"""
)


# 2. 資料理解 (Data Understanding)
st.header("2. 資料理解")
st.markdown(
    """
我們將會生成一組符合 `y = ax + b` 的合成資料，並加入一些隨機雜訊。您可以透過左側的滑桿來調整資料的特性。
"""
)


# --- 使用者輸入 (Sidebar) ---
st.sidebar.header("參數設定")
a = st.sidebar.slider("斜率 a", -5.0, 5.0, 1.0, 0.1)
b = st.sidebar.slider("截距 b", -20.0, 20.0, 10.0, 0.5)
noise = st.sidebar.slider("雜訊 noise (std)", 0.0, 5.0, 0.5, 0.1)
n_points = st.sidebar.slider("資料點數量 n", 10, 2000, 100, 10)
random_seed = st.sidebar.number_input("random seed (None 為 -1)", value=-1, step=1)
seed = None if random_seed < 0 else int(random_seed)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "提示 (Prompt) - 請說明你要探索的問題，例如：`我想看 a=2, b=5, noise=0.5 下模型的表現`。應用將展示完整流程與結果。"
)


# Data understanding & generation
@st.cache_data
def get_data(a, b, noise, n_points, seed):
    # try to use external data generator, fallback to local generator if it fails
    try:
        return generate_data(a=a, b=b, noise=noise, n_points=n_points, random_state=seed)
    except Exception:
        rng = np.random.RandomState(seed)
        x = rng.rand(n_points) * 10
        error = rng.randn(n_points) * noise
        y = a * x + b + error
        return pd.DataFrame({"x": x, "y": y})


data = get_data(a, b, noise, n_points, seed)

st.header("Data Understanding & Preparation")
st.markdown(
    "說明：我們使用合成資料 y = a*x + b + noise 以便清晰展示模型行為。下面列出前 5 筆資料："
)
st.dataframe(data.head())

st.markdown("資料摘要:")
st.write(data.describe())


# Modeling
st.header("Modeling")
X = data[["x"]]
y = data["y"]
model = LinearRegression()
model.fit(X, y)

# Evaluation
st.header("Evaluation")
st.markdown("模型訓練完成後，顯示模型參數與績效指標。以下同時顯示訓練時的真實參數（a, b）。")
st.write({
    "true_a": a,
    "true_b": b,
    "model_coef": float(model.coef_[0]),
    "model_intercept": float(model.intercept_),
    "r_squared": float(model.score(X, y)),
})

# Visualization / Deployment
st.header("Visualization")
chart = alt.Chart(data).mark_circle(size=60).encode(x="x", y="y", tooltip=["x", "y"]).interactive()
line_df = pd.DataFrame({"x": [0, 10], "y": [model.coef_[0] * 0 + model.intercept_, model.coef_[0] * 10 + model.intercept_]})
line = alt.Chart(line_df).mark_line(color="red").encode(x="x", y="y")
st.altair_chart(chart + line, use_container_width=True)

st.markdown("\n---\n流程紀錄 (Process & Prompt)\n")
st.markdown(
    """
Prompt 範例：

```
生成資料的參數：a={a}, b={b}, noise={noise}, n={n_points}, seed={seed}
我想知道模型是否能恢復真實的斜率/截距，並觀察 R-squared。
```

過程紀錄（簡化）：

1. 依使用者輸入生成合成資料。
2. 使用 LinearRegression 擬合資料。
3. 列出模型係數，並與真實參數比較。
4. 顯示散佈圖與擬合直線。
""".format(a=a, b=b, noise=noise, n_points=n_points, seed=seed)
)
