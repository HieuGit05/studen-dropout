Dự án Dự đoán Học sinh Bỏ học
Giới thiệu dự án
Dự án này nhằm khám phá các yếu tố tiên đoán học sinh bỏ học trong các cơ sở giáo dục đại học. Dữ liệu được cung cấp từ bộ dataset có sẵn, với mục tiêu xác định các yếu tố ảnh hưởng đến việc học sinh có bỏ học hay hoàn thành chương trình học. Bộ dữ liệu bao gồm nhiều yếu tố như tình trạng kinh tế xã hội, thành tích học tập, và các yếu tố khác, sẽ được sử dụng để xây dựng các mô hình dự báo.

Dataset
Bộ dữ liệu bao gồm hai tệp:

Dữ liệu đầy đủ (dataset.csv): Đây là tệp chứa bộ dữ liệu đầy đủ với tất cả các bản ghi có sẵn. Tệp này bao gồm tất cả các đặc tính và quan sát để phục vụ cho việc mô hình hóa và phân tích.
Dữ liệu mẫu (sample_data.csv): Đây là tệp chứa 50 dòng đầu tiên của bộ dữ liệu đầy đủ. Tệp này có thể được sử dụng cho việc thử nghiệm nhanh hoặc phân tích ban đầu khi làm việc với các tập dữ liệu nhỏ.
Link bộ dataset
Bộ dữ liệu có thể được tải xuống từ Kaggle tại đường link sau: Higher Education Predictors of Student Retention - Kaggle

Các cột trong dataset
Bộ dữ liệu bao gồm một số cột quan trọng (đặc tính) như:

Inflation rate: Tỷ lệ lạm phát trong nền kinh tế tại thời điểm đó.
GDP: Tổng sản phẩm quốc nội của quốc gia.
Target: Biến mục tiêu, cho biết kết quả của học sinh, liệu họ bỏ học, tốt nghiệp hay vẫn đang theo học.
Hướng dẫn sử dụng
Yêu cầu
Python 3.x
pandas
matplotlib (tuỳ chọn, dùng cho việc vẽ biểu đồ)
scikit-learn (tuỳ chọn, dùng cho mô hình học máy)
streamlit (để chạy ứng dụng web)
Cài đặt
Clone dự án (nếu cần) hoặc tải bộ dữ liệu xuống.
Cài đặt các thư viện yêu cầu bằng pip:
pip install pandas matplotlib scikit-learn streamlit
