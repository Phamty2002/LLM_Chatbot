import requests
import pandas as pd
import os
import time
from datetime import datetime
from newspaper import Article
import nltk

# Tải xuống dữ liệu ngôn ngữ cho newspaper
nltk.download('punkt')

def fetch_full_article(url):
    """
    Trích xuất nội dung đầy đủ của bài viết từ URL sử dụng Newspaper3k.

    Parameters:
        url (str): URL của bài viết.

    Returns:
        str: Nội dung đầy đủ của bài viết hoặc None nếu có lỗi xảy ra.
    """
    try:
        # Kiểm tra xem URL có chứa các từ khóa thường xuất hiện trong trang đồng ý hay không
        consent_keywords = ['consent', 'privacy', 'cookie', 'agreement']
        if any(keyword in url.lower() for keyword in consent_keywords):
            print(f"URL chứa từ khóa đồng ý. Bỏ qua: {url}")
            return None

        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except requests.exceptions.HTTPError as http_err:
        print(f"Lỗi HTTP khi lấy nội dung bài viết từ {url}: {http_err}")
        return None
    except Exception as e:
        print(f"Lỗi khi lấy nội dung bài viết từ {url}: {e}")
        return None

def get_newsapi_news(api_key, tickers, from_date, to_date, language='en', page_size=100, sleep_time=1):
    """
    Lấy dữ liệu tin tức từ NewsAPI.org cho nhiều mã cổ phiếu.

    Parameters:
        api_key (str): API key từ NewsAPI.org.
        tickers (list): Danh sách mã cổ phiếu.
        from_date (str): Ngày bắt đầu theo định dạng 'YYYY-MM-DD'.
        to_date (str): Ngày kết thúc theo định dạng 'YYYY-MM-DD'.
        language (str): Ngôn ngữ của bài viết.
        page_size (int): Số lượng kết quả mỗi trang (tối đa 100).
        sleep_time (int): Thời gian nghỉ giữa các yêu cầu API để tránh vượt quá giới hạn.

    Returns:
        list: Danh sách các bài viết tin tức.
    """
    base_url = 'https://newsapi.org/v2/everything'
    all_news = []

    total_tickers = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        params = {
            'q': ticker,
            'from': from_date,
            'to': to_date,
            'language': language,
            'sortBy': 'relevancy',
            'pageSize': page_size,
            'apiKey': api_key
        }

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"[{idx}/{total_tickers}] Đã tìm thấy {len(articles)} bài viết cho {ticker}.")

                for article in articles:
                    url = article.get('url')
                    if url:
                        full_text = fetch_full_article(url)
                        article['full_content'] = full_text if full_text else article.get('content', '')
                    else:
                        article['full_content'] = article.get('content', '')

                    # Chỉ thêm bài viết nếu có nội dung đầy đủ hoặc tóm tắt
                    if article['full_content']:
                        all_news.append(article)
                    else:
                        print(f"Bài viết không có nội dung đầy đủ và tóm tắt. Bỏ qua: {url}")
            else:
                print(f"[{idx}/{total_tickers}] Lỗi khi lấy tin tức cho {ticker}: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[{idx}/{total_tickers}] Lỗi yêu cầu khi lấy tin tức cho {ticker}: {e}")

        time.sleep(sleep_time)  # Chờ để tránh vượt quá giới hạn API

    return all_news

def save_news_to_csv(news, directory_path, filename='newsapi_stock_news.csv'):
    """
    Lưu danh sách các bài viết tin tức vào file CSV tại thư mục chỉ định.

    Parameters:
        news (list): Danh sách các bài viết tin tức.
        directory_path (str): Đường dẫn đến thư mục lưu file CSV.
        filename (str): Tên file CSV để lưu.
    """
    if not news:
        print("Không có dữ liệu để lưu.")
        return

    # Tạo DataFrame từ danh sách bài viết
    df = pd.DataFrame(news)

    # Chọn các cột cần thiết và đổi tên nếu cần
    columns = {
        'source': 'Source',
        'author': 'Author',
        'title': 'Title',
        'description': 'Description',
        'url': 'URL',
        'publishedAt': 'Published At',
        'content': 'Content',
        'full_content': 'Full Content'
    }

    # Kiểm tra các cột có tồn tại trong dữ liệu và đổi tên
    available_columns = {k: v for k, v in columns.items() if k in df.columns}
    df = df[list(available_columns.keys())].rename(columns=available_columns)

    # Làm phẳng cột 'Source' nếu nó là một dictionary
    if 'Source' in df.columns:
        df['Source'] = df['Source'].apply(lambda x: x.get('name') if isinstance(x, dict) else x)

    # Chuyển đổi 'Published At' sang định dạng datetime
    if 'Published At' in df.columns:
        df['Published At'] = pd.to_datetime(df['Published At'], errors='coerce')

    # Kiểm tra và tạo thư mục nếu nó không tồn tại
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Đã tạo thư mục: {directory_path}")
        except OSError as e:
            print(f"Không thể tạo thư mục {directory_path}. Lỗi: {e}")
            return

    # Tạo đường dẫn đầy đủ đến file CSV
    file_path = os.path.join(directory_path, filename)

    # Lưu DataFrame vào file CSV
    try:
        df.to_csv(file_path, index=False, encoding='utf-8-sig')  # Sử dụng encoding để hỗ trợ ký tự đặc biệt
        print(f"Dữ liệu tin tức đã được lưu vào {file_path}")
    except Exception as e:
        print(f"Không thể lưu dữ liệu vào file CSV. Lỗi: {e}")

def main():
    # API key của bạn từ NewsAPI.org
    API_KEY = 'ecb8fd4dc58649b1a4cdaeab879165e1'  # **Lưu ý**: Không chia sẻ API key công khai

    # Danh sách mã cổ phiếu để lấy tin tức
    tickers = [
        'NVDA', 'INTC', 'PLTR', 'TSLA', 'AAPL', 'BBD', 'T', 'SOFI',
        'WBD', 'SNAP', 'NIO', 'BTG', 'F', 'AAL', 'NOK', 'BAC',
        'CCL', 'ORCL', 'AMD', 'PFE', 'KGC', 'MARA', 'SLB', 'NU',
        'MPW', 'MU', 'LCID', 'NCLH', 'RIG', 'AMZN', 'ABEV', 'U',
        'LUMN', 'AGNC', 'VZ', 'WBA', 'WFC', 'RIVN', 'UPST', 'CFE',
        'CSCO', 'VALE', 'AVGO', 'PBR', 'GOOGL', 'SMMT', 'GOLD',
        'NGC', 'BCS', 'UAA'
    ]

    # Khoảng ngày để lấy tin tức
    from_date = '2024-09-01'  # Ngày bắt đầu mới (không quá xa trong quá khứ)
    to_date = '2024-09-19'    # Ngày kết thúc

    # Đường dẫn thư mục để lưu file CSV
    directory_path = r'C:\Users\Pham Ty\Desktop\Thesis\Data CSV Collection'

    # Ngôn ngữ của bài viết
    language = 'en'  # Bạn có thể thay đổi thành 'vi' nếu muốn lấy tin tức tiếng Việt

    # Số lượng kết quả mỗi trang (tối đa 100)
    page_size = 100

    # Thời gian nghỉ giữa các yêu cầu API để tránh vượt quá giới hạn
    sleep_time = 1  # Điều chỉnh nếu cần thiết

    # Lấy dữ liệu tin tức
    news = get_newsapi_news(
        api_key=API_KEY,
        tickers=tickers,
        from_date=from_date,
        to_date=to_date,
        language=language,
        page_size=page_size,
        sleep_time=sleep_time
    )

    # Lưu dữ liệu tin tức vào file CSV
    if news:
        save_news_to_csv(news, directory_path, 'newsapi_stock_news.csv')

if __name__ == "__main__":
    main()
