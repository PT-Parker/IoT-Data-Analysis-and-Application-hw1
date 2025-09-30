import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io
import zipfile
import datetime

st.set_page_config(page_title="Interactive Linear Regression Visualizer", layout="wide")

st.title("HW1-1: Interactive Linear Regression Visualizer")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# 1. Data Generation Parameters
n_points = st.sidebar.slider("Number of data points (n)", 100, 1000, 500)
coefficient_a = st.sidebar.slider("Coefficient 'a' (y = ax + b + noise)", -10.0, 10.0, 2.0, 0.1)
intercept_b = st.sidebar.number_input("Intercept b", value=5.0, step=0.5)
noise_variance = st.sidebar.slider("Noise Variance (var)", 0, 1000, 100)
random_seed = st.sidebar.number_input("random seed (None = -1)", value=-1, step=1)
seed = None if int(random_seed) < 0 else int(random_seed)

st.sidebar.markdown('---')
st.sidebar.markdown('Use the controls above to modify data generation parameters.')

# Generate data
rng = np.random.RandomState(seed)
x = rng.rand(n_points) * 10
y_true = coefficient_a * x + intercept_b
noise = rng.normal(0, np.sqrt(max(noise_variance, 0)), n_points)
y = y_true + noise

df = pd.DataFrame({'x': x, 'y': y})

st.subheader("Generated Data and Linear Regression")

# Perform Linear Regression
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
y_pred = model.predict(x.reshape(-1, 1))

# Calculate residuals for outlier detection
residuals = np.abs(y - y_pred)
df['residuals'] = residuals

# Identify top 5 outliers
outliers = df.nlargest(5, 'residuals')

# Plotting with matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['x'], df['y'], label='Generated Data', alpha=0.6)
# sort by x for a clean line
order = np.argsort(df['x'].values)
ax.plot(df['x'].values[order], y_pred[order], color='red', label='Linear Regression', linewidth=2)

# Annotate outliers
for i, row in outliers.iterrows():
    ax.annotate(f'Outlier {i}', (row['x'], row['y']), textcoords="offset points", xytext=(0,10), ha='center', color='purple')
    ax.scatter(row['x'], row['y'], color='purple', s=100, edgecolors='black', zorder=5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Linear Regression with Outliers")
ax.legend()

st.pyplot(fig)

st.subheader("Model Coefficients")
st.write(f"Coefficient (a): {model.coef_[0]:.2f}")
st.write(f"Intercept (b): {model.intercept_:.2f}")

# Evaluation metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
st.subheader("Evaluation")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"R-squared: {r2:.4f}")

st.subheader("Top 5 Outliers")
st.dataframe(outliers[['x', 'y', 'residuals']])

# --- CRISP-DM explanation panel ---
with st.expander("CRISP-DM: steps and how this app maps to them", expanded=False):
     st.markdown("""
     **CRISP-DM 步驟對應說明**

     1. Business / Domain Understanding
         - 目標：以簡單線性模型 y = a x + b + noise 示範模型行為與雜訊影響。
     2. Data Understanding
         - 透過隨機產生資料 (x, y)，觀察資料分布與雜訊。
     3. Data Preparation
         - 在本例中資料已簡化為數值欄位；使用者可透過側邊欄調整參數產生新資料。
     4. Modeling
         - 使用 scikit-learn 的 LinearRegression 進行擬合，並顯示回歸線與係數。
     5. Evaluation
         - 顯示 MSE、R²、殘差與前 5 個離群點（outliers）。
     6. Deployment
         - 本應用展示如何在 Streamlit 上部署與互動；可下載分析報告供匯出與評分。
     """)

# --- Report download ---
st.markdown("---")
if st.button("Prepare downloadable report (zip)"):
     # Create report text
     ts = datetime.datetime.now().isoformat()
     report_lines = []
     report_lines.append(f"CRISP-DM report generated: {ts}")
     report_lines.append("")
     report_lines.append("Parameters:")
     report_lines.append(f"  n_points = {n_points}")
     report_lines.append(f"  coefficient_a = {coefficient_a}")
     report_lines.append(f"  intercept_b = {intercept_b}")
     report_lines.append(f"  noise_variance = {noise_variance}")
     report_lines.append(f"  random_seed = {seed}")
     report_lines.append("")
     report_lines.append("Model results:")
     report_lines.append(f"  Coefficient (a): {model.coef_[0]:.6f}")
     report_lines.append(f"  Intercept (b): {model.intercept_:.6f}")
     report_lines.append(f"  MSE: {mse:.6f}")
     report_lines.append(f"  R2: {r2:.6f}")

     report_text = "\n".join(report_lines)

     # Save plot to PNG bytes
     img_bytes = io.BytesIO()
     fig.savefig(img_bytes, format='png', bbox_inches='tight')
     img_bytes.seek(0)

     # Save dataframe CSV bytes
     csv_bytes = df.to_csv(index=False).encode('utf-8')

     # Create zip in-memory
     zip_buffer = io.BytesIO()
     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
          zf.writestr('report.txt', report_text)
          zf.writestr('plot.png', img_bytes.getvalue())
          zf.writestr('data.csv', csv_bytes)
     zip_buffer.seek(0)

     # Provide download button
     st.download_button(
          label='Download analysis report (.zip)',
          data=zip_buffer.getvalue(),
          file_name=f'crispdm_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
          mime='application/zip'
     )
