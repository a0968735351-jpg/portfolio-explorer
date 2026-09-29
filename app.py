import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize

# 針對手機版最佳化版面
st.set_page_config(page_title="我的資產配置組合試算", layout="centered", page_icon="🧭")

st.title("🧭 我的投資性格與黃金配比")
st.write("請輸入感興趣的股票或 ETF！")
# ==================== 頁面最上方：免責聲明與風險提醒 ====================
st.warning(
    "⚠️ **【投資免責聲明與風險提示】**\n\n"
    "本工具與模擬模型僅供個人輔助參考，**不構成任何形式之投資建議、推薦或要約**。"
    "歷史回測數據與統計估計不代表未來績效保證，金融市場投資具備本金虧損風險，投資人應獨立審慎評估並自負投資風險。"
)

# 1. 標的輸入區（預設給幾檔常見標的）
default_tickers = "0050, 2330, VOO, BND"
user_input = st.text_input(
    "👉 請輸入 3 ~ 10 檔股票代碼（以逗號或空格隔開）：",
    value=default_tickers,
    help="美股直接輸入代碼（如 VOO, QQQ, BND, GLD）；台股輸入純數字（如 0050, 2330）"
)

# 2. 投資性格滑桿（互動機制）
risk_tolerance = st.slider(
    "🎯 調整風險偏好（投入切點組合的比例 %）：",
    min_value=0, max_value=100, value=70, step=5
)

# 性格稱號判定
if risk_tolerance <= 30:
    badge = "🛡️ 保守型穩健防禦家（重視本金安全與息收）"
elif risk_tolerance <= 70:
    badge = "⚖️ 均衡型成長探索者（攻守兼備，享受市場平均增長）"
else:
    badge = "🔥 積極型動能追尋者（極大化長期資本利得）"

st.info(f"當前性格定位：**{badge}**")

# 代碼解析與清洗
tokens = [t.strip().upper() for t in user_input.replace(',', ' ').split() if t.strip()]
unique_tokens = list(dict.fromkeys(tokens))

if len(unique_tokens) < 3:
    st.warning("請至少輸入 3 檔標的，才能發揮資產分散效果喔！")
    st.stop()

tickers = {t: f"{t}.TW" if t.isdigit() else t for t in unique_tokens}

# 3. 數據下載與最佳化計算
with st.spinner("連線市場抓取歷史數據中..."):
    try:
        raw_prices = yf.download(list(tickers.values()), period='3y')['Close']
        data = raw_prices.dropna().rename(columns={v: k for k, v in tickers.items()})
    except Exception as e:
        st.error(f"數據下載失敗，請檢查代碼拼寫是否正確：{e}")
        st.stop()

if data.empty or len(data.columns) < 2:
    st.error("有效標的不足，請確認代碼是否正確！")
    st.stop()

returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_assets = len(data.columns)
rf = 0.04

def get_p_stats(w):
    ret = np.dot(w, mean_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    return ret, vol

bounds = tuple((0.0, 1.0) for _ in range(num_assets))
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
init_weights = [1.0 / num_assets] * num_assets

# 求解最大夏普切點 T
opt_res = minimize(lambda w: -(get_p_stats(w)[0] - rf) / get_p_stats(w)[1],
                   init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = opt_res.x
ret_T, vol_T = get_p_stats(best_weights)
sharpe_T = (ret_T - rf) / vol_T

# 計算使用者當前滑桿下的數值
y = risk_tolerance / 100.0
user_ret = rf + y * (ret_T - rf)
user_vol = y * vol_T

# 4. 指標卡片（手機好讀格式）
col1, col2 = st.columns(2)
col1.metric("預期年化報酬率", f"{user_ret * 100:.2f} %")
col2.metric("預期年化波動度", f"{user_vol * 100:.2f} %")

# 5. 繪製手機適配的互動圖表
targets = np.linspace(mean_returns.min() * 0.9, mean_returns.max() * 1.05, 40)
frontier_vols, frontier_rets = [], []
for tr in targets:
    res = minimize(lambda w: get_p_stats(w)[1], init_weights, method='SLSQP', bounds=bounds,
                   constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                                {'type': 'eq', 'fun': lambda w, target=tr: get_p_stats(w)[0] - target}))
    if res.success:
        frontier_vols.append(res.fun)
        frontier_rets.append(tr)

fig = go.Figure()
fig.add_trace(go.Scatter(x=frontier_vols, y=frontier_rets, mode='lines', name='效率前緣', line=dict(color='#E91E63', width=3)))

cml_x = np.linspace(0, max(frontier_vols) * 1.25, 40)
cml_y = rf + sharpe_T * cml_x
fig.add_trace(go.Scatter(x=cml_x, y=cml_y, mode='lines', name='資本市場線 (CML)', line=dict(color='#007ACC', dash='dash')))

# 個別標的點
for name in data.columns:
    fig.add_trace(go.Scatter(x=[np.sqrt(cov_matrix.loc[name, name])], y=[mean_returns[name]],
                             mode='markers+text', name=name, text=[name], textposition="top right",
                             marker=dict(size=8, color='#78909C')))

# 標記切點與玩家位置
fig.add_trace(go.Scatter(x=[vol_T], y=[ret_T], mode='markers', name='切點組合 T', marker=dict(size=12, color='black')))
fig.add_trace(go.Scatter(x=[user_vol], y=[user_ret], mode='markers', name='你的目前位置', marker=dict(size=14, color='#00E676', symbol='diamond')))

fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="風險 (年化波動度)",
    yaxis_title="預期年化報酬率",
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
)
st.plotly_chart(fig, use_container_width=True)

# 6. 配置配方清單
st.subheader("📋 我的專屬資金分配比率")
weights_data = [{"標的名稱": "無風險公債 / 定存", "配置比例": f"{(1 - y) * 100:.2f} %"}]
for name, w in zip(data.columns, best_weights):
    if (y * w) >= 0.0001:
        weights_data.append({"標的名稱": name, "配置比例": f"{(y * w) * 100:.2f} %"})

st.dataframe(pd.DataFrame(weights_data), use_container_width=True, hide_index=True)

# ==================== 頁面最下方：技術與方法論附錄 ====================
st.markdown("---")

with st.expander("📚 查看技術附錄：資料來源、統計方法與程式架構說明", expanded=False):
    st.markdown("### 🛠️ 專案技術與方法論說明 (Methodology & Appendix)")

    tab_data, tab_model, tab_code = st.tabs(["1. 資料來源與處理", "2. 財務模型與計算", "3. 核心程式碼架構"])

    # ---------------- 頁籤 1：資料來源與處理 ----------------
    with tab_data:
        st.markdown("""
        #### 📌 資料來源與取樣設計
        * **資料來源**：透過開源套件 `yfinance` 串接 **Yahoo Finance** API 下載歷史交易行情。
        * **代碼自動解析**：
          * 支援輸入純數字代碼（例如 `0050`、`2330`），系統會自動後綴 `.TW` 轉換為台灣市場格式。
          * 支援輸入美股英文代碼（例如 `VOO`、`QQQ`、`NVDA`）。
        * **回溯歷史期間 (Lookback Period)**：
          * 採用 **過去 3 年（`period='3y'`）** 之歷史日頻率交易數據（約 750 個交易日）。
          * *選擇理由*：3 年歷史數據能有效反映近期市場的結構與各標的波動特徵，避免過於久遠的資料失真，同時具備足夠的樣本點以穩定估計共變異數矩陣。
        * **價格欄位**：採用每日 **收盤價（Close）**。
        * **缺漏值處理**：跨市場休市日不一致時，使用 `dropna()` 剔除未對齊的日期，確保時間序列資料完整一致。
        """)

    # ---------------- 頁籤 2：財務模型與計算 ----------------
    with tab_model:
        st.markdown(r"""
        #### 📐 現代投資組合理論 (MPT) 與參數估算
        * **日收益率計算**：
          採用離散百分比報酬率：
          $$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
        * **參數年化方式 (Annualization)**：
          以一年 252 個交易日進行折算：
          * **年化預期報酬率**：$\mu = \bar{R}_{\text{daily}} \times 252$
          * **年化共變異數矩陣**：$\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_{\text{daily}} \times 252$
        * **投資組合風險指標**：
          * **組合年化波動度 (標準差)**：$\sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}$
          * **夏普值 (Sharpe Ratio)**：$\text{SR} = \frac{E(R_p) - R_f}{\sigma_p}$（衡量承擔每單位總風險所獲得的超額回報）
        * **分散風險核心依據**：
          * 當納入新標的 $E$ 時，若 $\text{Cov}(R_E, R_p) < \sigma_p^2$，該標的就能在邊際上壓低投資組合總波動度，實現資產配置的分散效果。
        """)

    # ---------------- 頁籤 3：核心程式碼架構 ----------------
    with tab_code:
        st.markdown("""
        #### 💻 核心資料抓取與矩陣運算片段
        ```python
        import yfinance as yf

        # 1. 代碼解析與 3 年歷史收盤價下載
        tickers = {t: f"{t}.TW" if t.isdigit() else t for t in unique_tokens}
        raw_prices = yf.download(list(tickers.values()), period='3y')['Close']
        data = raw_prices.dropna().rename(columns={v: k for k, v in tickers.items()})

        # 2. 計算日報酬率並做年化參數估算 (252 個交易日)
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        ```
        """)

    st.caption("© 投資組合簡易指南 (Portfolio Easy Guide) · 基於現代投資組合理論 (Markowitz MPT) 設計")
