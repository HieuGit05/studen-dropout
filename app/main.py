import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. LOAD PIPELINE
# (Thay đổi tên file nếu bạn dùng file lite hay file full)
pipeline_file = 'dropout_pipeline_lite.pkl' 
# Hoặc 'dropout_pipeline.pkl' nếu bạn muốn kiểm tra bản full

try:
    pipeline = joblib.load(pipeline_file)
    print(f"✅ Đã load {pipeline_file}")
except:
    print("❌ Không tìm thấy file model (.pkl). Hãy chạy train trước.")
    exit()

# 2. LẤY MÔ HÌNH VÀ BỘ XỬ LÝ TỪ PIPELINE
try:
    # Lấy model ra
    model = pipeline.named_steps['model'] # Hoặc 'classifier' tùy tên bạn đặt trong train.py
    
    # Lấy preprocessor ra
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Lấy tên các cột sau khi đã One-Hot Encoding
    feature_names = preprocessor.get_feature_names_out()
    
    # Lấy trọng số (Coefficients) - Mức độ quan trọng
    # Logistic Regression trả về coef_ dạng [[w1, w2, ...]] nên lấy [0]
    coefficients = model.coef_[0]

except Exception as e:
    print(f"Lỗi khi trích xuất thông tin: {e}")
    print("Gợi ý: Kiểm tra lại tên bước trong Pipeline (ví dụ: 'model' hay 'classifier')")
    exit()

# 3. TẠO DATAFRAME ĐỂ DỄ NHÌN
df_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coefficients
})

# Sắp xếp theo trị tuyệt đối (để xem cái nào tác động mạnh nhất, bất kể chiều nào)
df_imp['Abs_Importance'] = df_imp['Importance'].abs()
df_imp = df_imp.sort_values(by='Abs_Importance', ascending=False).head(15) # Top 15

# 4. VẼ BIỂU ĐỒ
plt.figure(figsize=(12, 8))
# Tô màu: Xanh (Dương) = Tăng nguy cơ Bỏ học | Đỏ (Âm) = Tăng khả năng Tốt nghiệp
colors = ['red' if x < 0 else 'blue' for x in df_imp['Importance']]

sns.barplot(x='Importance', y='Feature', data=df_imp, palette=colors)

plt.title('TOP YẾU TỐ ẢNH HƯỞNG ĐẾN QUYẾT ĐỊNH CỦA MÔ HÌNH', fontsize=15, fontweight='bold')
plt.xlabel('Mức độ ảnh hưởng (Trọng số)', fontsize=12)
plt.ylabel('Tên đặc trưng', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Ghi chú
plt.text(0.5, 0.5, "DƯƠNG (+) --> Đẩy về BỎ HỌC (Dropout)\n(Cột bên Phải)", 
         transform=plt.gca().transAxes, color='blue', fontweight='bold')
plt.text(0.1, 0.5, "ÂM (-) --> Đẩy về AN TOÀN (Graduate)\n(Cột bên Trái)", 
         transform=plt.gca().transAxes, color='red', fontweight='bold')

plt.tight_layout()
plt.show()