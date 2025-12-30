import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 1. DANH SÁCH 10 ĐẶC TRƯNG QUAN TRỌNG NHẤT (Đã chọn lọc)
# Đây là những yếu tố ảnh hưởng mạnh nhất đến việc bỏ học
SELECTED_FEATURES = [
    "Tuition fees up to date",          # Đóng học phí đủ không (Quan trọng nhất)
    "Curricular units 2nd sem (approved)", # Số môn đậu kỳ 2 (Quan trọng nhì)
    "Curricular units 2nd sem (grade)",    # Điểm TB kỳ 2
    "Curricular units 1st sem (approved)", # Số môn đậu kỳ 1
    "Curricular units 1st sem (grade)",    # Điểm TB kỳ 1
    "Age at enrollment",                # Tuổi
    "Debtor",                           # Có nợ môn/tiền không
    "Scholarship holder",               # Có học bổng không
    "Gender",                           # Giới tính
    "Displaced"                         # Sống xa nhà
]

# Phân loại cột số và cột chữ trong danh sách rút gọn này
NUMERICAL_COLS = [
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Age at enrollment"
]

CATEGORICAL_COLS = [
    "Tuition fees up to date", "Debtor", "Scholarship holder", "Gender", "Displaced"
]

# 2. LOAD & XỬ LÝ DỮ LIỆU
print("⏳ Đang xử lý dữ liệu...")
df = pd.read_csv('dataset.csv')

# Xử lý nhãn (Target)
df['Target'] = df['Target'].astype(str).str.strip()
mapping = {'Dropout': 1, 'Graduate': 0, 'Enrolled': 0}
df['Target'] = df['Target'].replace(mapping)
df = df[df['Target'].isin([0, 1])]

# CHỈ LẤY 10 CỘT ĐÃ CHỌN + CỘT TARGET
X = df[SELECTED_FEATURES]
y = df['Target'].astype(int)

# 3. CHIA TẬP TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TẠO PREPROCESSOR (Chỉ cho 10 cột này)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
    ]
)

# 5. HUẤN LUYỆN (Vẫn dùng SMOTE cho tốt)
print("⚙️ Đang huấn luyện mô hình rút gọn...")
# Fit preprocessor
X_train_processed = preprocessor.fit_transform(X_train)

# SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Train Model
model = LogisticRegression(max_iter=3000, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 6. ĐÓNG GÓI PIPELINE
pipeline_lite = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Test nhanh
acc = pipeline_lite.score(X_test, y_test)
print(f"✅ Độ chính xác của bản rút gọn: {acc:.4f}")

# Lưu file tên khác để không nhầm
joblib.dump(pipeline_lite, 'dropout_pipeline_lite.pkl')
print("💾 Đã lưu 'dropout_pipeline_lite.pkl'")