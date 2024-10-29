from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

# Đường dẫn đến ChromeDriver (chỉnh lại đường dẫn theo máy bạn)
PATH = "E:\chromedriver-win64\chromedriver-win64\chromedriver"  
driver = webdriver.Chrome(PATH)

# Bước 1: Mở trang Yahoo Finance
driver.get("https://finance.yahoo.com/quote/NVDA/financials/")

# Bước 2: Đợi trang tải (có thể cần tăng thời gian nếu mạng chậm)
time.sleep(5)

# Bước 3: Cào dữ liệu
# Tìm các phần tử chứa thông tin cần thiết (ví dụ: giá hiện tại, tỉ lệ thay đổi, ...)
try:
    stock_name = driver.find_element(By.XPATH, '//h1').text  # Lấy tên công ty
    stock_price = driver.find_element(By.XPATH, '//fin-streamer[@data-symbol="AAPL"]/span').text  # Lấy giá chứng khoán
    stock_change = driver.find_element(By.XPATH, '//fin-streamer[@data-field="regularMarketChangePercent"]/span').text  # Lấy tỉ lệ phần trăm thay đổi

    # In thông tin ra màn hình
    print(f"Stock Name: {stock_name}")
    print(f"Stock Price: {stock_price}")
    print(f"Stock Change (%): {stock_change}")
    
    # Tạo một DataFrame lưu lại dữ liệu
    data = {
        'Stock Name': [stock_name],
        'Stock Price': [stock_price],
        'Stock Change (%)': [stock_change]
    }

    df = pd.DataFrame(data)
    print(df)

    # Lưu dữ liệu vào file CSV (nếu cần)
    df.to_csv('stock_data.csv', index=False)

except Exception as e:
    print("Có lỗi xảy ra:", e)

# Bước 4: Đóng trình duyệt
driver.quit()
