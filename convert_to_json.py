import pandas as pd
import json
import os

# Hàm chuyển CSV sang JSON theo định dạng yêu cầu
def convert_csv_to_json(csv_file):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_file)

    # Danh sách để lưu các đối tượng JSON
    json_data = []

    # Duyệt qua các dòng dữ liệu trong CSV và tạo câu hỏi/câu trả lời
    for index, row in df.iterrows():
        date = row['Date']           # Cột 'Date' (mm/dd/yyyy)
        predicted_price = row['Forecasting']  # Cột 'Forecasting' (giá cổ phiếu)

        # Tạo câu hỏi và câu trả lời
        instruction = f"Dự đoán giá cổ phiếu vào ngày {date}?"
        output = f"Giá của cổ phiếu vào ngày {date} là {predicted_price} đô."

        # Thêm vào danh sách JSON
        json_data.append({
            "instruction": instruction,
            "output": output
        })

    # Trả về dữ liệu JSON
    return json_data

# Hàm lưu JSON vào file
def save_json_to_file(json_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

# Danh sách các file CSV và tên file JSON tương ứng
csv_files = [
    r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_AAPL).csv",
    r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_GOOGL).csv",
    r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_MSFT).csv",
    r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_TSLA).csv",
    r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_AMZN).csv"
]




# Vòng lặp để xử lý tất cả các file
for csv_file in csv_files:
    # Tạo tên file JSON tương ứng
    file_name = os.path.basename(csv_file)
    json_file = os.path.splitext(file_name)[0] + ".json"  # Đổi đuôi thành .json
    json_file_path = os.path.join(os.path.dirname(csv_file), json_file)  # Lưu cùng thư mục với file CSV

    # Chuyển đổi dữ liệu và lưu ra file JSON
    json_data = convert_csv_to_json(csv_file)
    save_json_to_file(json_data, json_file_path)

    print(f"Đã lưu: {json_file_path}")
