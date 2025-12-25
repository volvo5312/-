import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def set_chinese_font():
    plt.rcParams['axes.unicode_minus'] = False 
    # 判斷是否為 Windows，如果是則使用微軟正黑體
    if os.name == 'nt':
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    else:
        # 如果是 Linux/Colab，檢查是否有 Noto 字型，沒有則下載
        print("檢測到非 Windows 環境，準備設定中文字型...")
        font_path = 'NotoSansCJKtc-Regular.otf'
        
        if not os.path.exists(font_path):
            print("正在下載 NotoSansCJKtc-Regular 字型...")
            # 使用 wget 下載字型
            os.system('wget https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf')
        
        if os.path.exists(font_path):
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = ['Noto Sans CJK TC']
            print("字型設定完成！")
        else:
            print("字型下載失敗，將使用預設字型 (中文可能會顯示為方框)。")

set_chinese_font()

# ==========================================
# 1. 資料準備 (模擬 UCI Student Performance Dataset)
# ==========================================


np.random.seed(42)
n_samples = 600

data = {
    'sex': np.random.choice(['F', 'M'], n_samples),
    'age': np.random.randint(15, 22, n_samples),
    'address': np.random.choice(['U', 'R'], n_samples), # Urban, Rural
    'studytime': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    'failures': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.05, 0.05]),
    'schoolsup': np.random.choice(['yes', 'no'], n_samples), # 學校輔導
    'internet': np.random.choice(['yes', 'no'], n_samples, p=[0.8, 0.2]),
    'goout': np.random.randint(1, 6, n_samples), # 外出頻率
    'absences': np.random.randint(0, 30, n_samples),
    'G1': np.random.normal(11, 3, n_samples).astype(int), # 第一階段成績
    'G2': np.random.normal(11, 3, n_samples).astype(int)  # 第二階段成績
}

df = pd.DataFrame(data)

# 為了讓數據更真實，讓 G2 與 G1 高度相關，G3 與 G2 高度相關
df['G2'] = df['G1'] + np.random.randint(-2, 3, n_samples)
df['G3'] = df['G2'] + np.random.randint(-2, 3, n_samples)
# 確保成績在 0-20 範圍內
for col in ['G1', 'G2', 'G3']:
    df[col] = df[col].clip(0, 20)

print("--- 資料預覽 ---")
print(df.head())

# ==========================================
# 2. 資料前處理與探索 (EDA)
# ==========================================

# 將類別資料轉為數值
le = LabelEncoder()
categorical_cols = ['sex', 'address', 'schoolsup', 'internet']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 繪製相關係數熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特徵相關係數熱力圖')
plt.tight_layout()
plt.show() # 請截圖此圖表放入 PPT

# 繪製成績分佈
plt.figure(figsize=(8, 6))
sns.histplot(df['G3'], bins=20, kde=True, color='skyblue')
plt.title('期末成績 (G3) 分佈圖')
plt.xlabel('分數 (0-20)')
plt.ylabel('人數')
plt.show()

# ==========================================
# 3. 監督式學習：預測是否及格 (Binary Classification)
# ==========================================
print("\n--- 監督式學習：隨機森林預測及格與否 ---")

# 定義目標：G3 >= 10 為及格 (1)，否則為不及格 (0)
df['pass'] = np.where(df['G3'] >= 10, 1, 0)

# 特徵選擇 (移除 G3 與 pass 本身)
X = df.drop(['G3', 'pass'], axis=1)
y = df['pass']

# 切分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 預測
y_pred = rf_model.predict(X_test)

# 評估
print(f"準確率 (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print("\n混淆矩陣 (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\n分類報告:")
print(classification_report(y_test, y_pred))

# 特徵重要性視覺化
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh', color='lightgreen')
plt.title('隨機森林模型 - 特徵重要性')
plt.xlabel('重要性分數')
plt.show() # 請截圖此圖表放入 PPT

# ==========================================
# 4. 非監督式學習：學生行為分群 (Clustering)
# ==========================================
print("\n--- 非監督式學習：K-Means 學生分群 ---")

# 選擇行為相關特徵進行分群 (不包含成績，看能否單純依行為分出優劣)
features_clustering = ['studytime', 'absences', 'goout', 'failures']
X_cluster = df[features_clustering]

# 資料標準化 (K-Means 對尺度敏感)
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# 建立 K-Means 模型 (假設分為 3 群)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_cluster_scaled)

# 分析各群的平均表現 (包含成績 G3 做驗證)
cluster_analysis = df.groupby('cluster')[features_clustering + ['G3']].mean()
print("各群組平均特徵值：")
print(cluster_analysis)

# 視覺化分群結果 (以 缺席數 vs 讀書時間 為例)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='studytime', y='absences', hue='cluster', palette='viridis', s=100)
plt.title('K-Means 學生分群結果 (讀書時間 vs 缺席數)')
plt.xlabel('每週讀書時間 (等級)')
plt.ylabel('缺席次數')
plt.legend(title='群組 (Cluster)')
plt.show() # 請截圖此圖表放入 PPT

print("\n程式執行完畢。")
