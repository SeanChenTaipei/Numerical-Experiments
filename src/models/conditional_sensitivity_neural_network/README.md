# Conditional Sensitivity Neural Network (CSNN)

ConditionalSNNRegressor 是一個可解釋的階層式加法模型：先以 BaseGAM 學習全域特徵形狀，再透過 group adapter (GroupResidualHead 或 GroupAffineHead) 注入群組特化行為。此模組整合 PyTorch Lightning、Ray Tune、以及一套追蹤特徵貢獻的工具，方便在具時間 / 群組結構的 tabular regression 上進行研究與部署。

## 模組重點
- **MixedGAMRegressor**：主要實作者，提供 it / predict / predict_and_contrib / plot_feature_shapes / save / load 等 API。
- **ConditionalSNNRegressor**：MixedGAMRegressor 的語義別名，額外包含 ay_tune_search 進行超參搜尋。
- **GroupCrossEncoder**：處理任意群組列 (machine_id, atch...)，自動映射到 adapter index，並將未知群組歸到 __unknown__。
- **_IdentityScaler / StandardScaler / RobustScaler**：依設定選擇特徵縮放方式；目標值則獨立維持一組 scaler 以利 inverse-transform。

## 依賴與安裝
`ash
pip install -r requirements.txt
# Ray Tune 為選配，若要啟用搜尋：
pip install "ray[tune]"  # 可加上 'default' 或 'gpu' extra
`

## 資料需求
`	ext
X: numpy.ndarray 或 pandas.DataFrame，shape = (n_samples, n_features)
y: Sequence[float]
groups: Sequence[Any] (選填) － 如果未提供會全部視為 __global__
`
- 特徵必須是數值型（類別先自行 encoding）。
- it 會自動建立 eature_names、記錄特徵範圍，後續 predict 需維持相同欄位順序。

## 快速上手
`python
from pathlib import Path
from src.models.conditional_sensitivity_neural_network.model import ConditionalSNNRegressor

model = ConditionalSNNRegressor(
    mode="group_residual",          # or "group_affine"
    n_epochs=200,
    base_hidden_units=(128, 64, 32),
    residual_hidden_units=(64, 32),
    batch_size=512,
    patience=20,
)
model.fit(X_train, y_train, groups=g_train)
preds = model.predict(X_val, groups=g_val)
`

### 取得貢獻與形狀圖
`python
preds, contribs = model.predict_and_contrib(X_test, groups=g_test)
base = contribs["base_per_feature"]          # 全域形狀 (n_samples, n_features)
per_group = contribs.get("residual_per_feature")
model.plot_feature_shapes(feature_name="runtime_hours", mode="auto")
`
> plot_feature_shapes 可顯示全域/群組對比；per_feature 為最終貢獻，ias 為常數截距。這些輸出皆已依目標 scaler 還原單位。

### 儲存 / 載入
`python
model.save("artifacts/csnn.pt", "artifacts/csnn_meta.pkl")
restored = ConditionalSNNRegressor.load("artifacts/csnn.pt", "artifacts/csnn_meta.pkl")
`
儲存內容包含：
- PyTorch state dict（base GAM + group head）
- scaler、target scaler、feature metadata、群組 encoder 狀態、預設群組清單，以及快取的訓練貢獻。

## Ray Tune 搜尋
`python
search = ConditionalSNNRegressor.ray_tune_search(
    X_train,
    y_train,
    groups=g_train,
    num_samples=30,
    metric="val_loss",
    mode="min",
    val_ratio=0.2,
)
best_cfg = search["best_config"]
model = ConditionalSNNRegressor(**best_cfg)
model.fit(X_train, y_train, groups=g_train)
`
- 若未提供 param_space，會自動使用 default_ray_search_space（learning rate / hidden units / dropouts / reg / mode / scaler / batch size）。
- 可傳入 esources_per_trial 指定 GPU / CPU 數，及自訂 scheduler (預設 ASHAScheduler)。

## 訓練技巧
1. **群組平衡**：若群組分布極度不均，可先做抽樣或在 it 前分層切割；ay_tune_search 內部已支援對群組 stratify。
2. **Callbacks**：若未額外傳入，模組會自動注入 EarlyStopping、LearningRateMonitor、ModelCheckpoint，並使用 TQDM 進度條。
3. **GPU 加速**：自動偵測 	orch.cuda.is_available()，並讓 Lightning 使用單 GPU；CPU 模式則會落在 ccelerator="cpu"。
4. **解讀**：利用 predict_and_contrib / plot_feature_shapes 觀察 Global vs Group 行為。mode="group_affine" 代表 per-feature affine warp，而 group_residual 代表殘差加法。

## 檔案結構速覽
`
model.py                  # 主體 (MixedGAM + ConditionalSNNRegressor)
helpers.py                # BaseGAM / ConditionalSNN / group heads / dataclasses
loss.py, metrics.py       # Lightning module使用的客製 loss、評估指標
`

如需整合到其他 pipeline，可在 src/models/conditional_sensitivity_neural_network/__init__.py 直接匯入 ConditionalSNNRegressor。若有更多客製需求（例：多任務、不同 regularizer），建議以 MixedGAMConfig dataclass 為入口覆寫對應欄位。
