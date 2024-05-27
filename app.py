import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import pandas as pd
from openpyxl import Workbook
from flask import Flask, render_template
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def insighter(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        shop_index_link = soup.find('a', href='/shop')
        if shop_index_link:
            shop_index_url = urljoin(url, shop_index_link['href'])
            response = requests.get(shop_index_url)
            
            if response.status_code == 200:
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "Shop Data"
                sheet.append(['Title', 'Sale Price', 'Original Price', 'Detail', 'Type', 'Category'])

                soup = BeautifulSoup(response.content, 'html.parser')
                shop_blocks = soup.find_all('a', class_='grid-item-link product-lists-item')
                product_urls = [urljoin(url, shop_block.get('href')) for shop_block in shop_blocks]
                
                for product_url in product_urls:
                    response = requests.get(product_url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        all_page = soup.find(class_='content')
                        articles = soup.find_all('article')
                        tags = ""
                        if len(articles) >= 2:
                            second_article = articles[1]
                            class_name = second_article.get('class')
                            if class_name:
                                class_name_text = ' '.join(class_name)
                                tags = re.findall(r'tag-(\w+)', class_name_text)
                                if tags:
                                    tags_text = ', '.join(tags)

                        product_item_title_element = all_page.find('h1', class_='ProductItem-details-title')
                        product_item_title = product_item_title_element.text.strip() if product_item_title_element else 'N/A'
                        print(product_item_title + " is being scraped")

                        product_price_element = all_page.find('div', class_='product-price')
                        
                        if product_price_element:
                            sale_price_element = product_price_element.find('span', class_='visually-hidden', text='Sale Price:')
                            original_price_element = product_price_element.find('span', class_='original-price')
                            
                            if sale_price_element and original_price_element:
                                sale_price = sale_price_element.next_sibling.strip() if sale_price_element else 'N/A'
                                original_price = original_price_element.text.strip() if original_price_element else 'N/A'
                            else:
                                sale_price = product_price_element.text.strip()
                                original_price = 'N/A'
                        else:
                            sale_price = 'N/A'
                            original_price = 'N/A'

                        product_item_details_element = all_page.find('div', class_='ProductItem-details-excerpt-below-price')
                        product_item_details = product_item_details_element.text.strip() if product_item_details_element else 'N/A'
                        
                        element = all_page.find('div', class_='variant-radiobtn-wrapper')
                        if element:
                            labels = ', '.join(label.text.strip() for label in element.find_all('label'))
                        else:
                            labels = "N/A"

                        category = ', '.join(tags)
                        sheet.append([product_item_title, sale_price, original_price, product_item_details, labels, category])
                
                workbook.save('shop.xlsx')
            else:
                print("Failed to retrieve the shop index page")
        else:
            print("Shop index link not found")
    else:
        print("Failed to retrieve the webpage")

def load_data():
    df = pd.read_excel('shop.xlsx')

    df['Sale Price'] = df['Sale Price'].replace('[\$,]', '', regex=True).astype(float)
    df['Original Price'] = df['Original Price'].replace('[\$,]', '', regex=True).astype(float)
    
    return df

app = Flask(__name__)

@app.route('/')
def index():
    df = load_data()

    summary = df.describe()
    
    summary_formatted = summary.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else f'{int(x)}')
    summary_html = summary_formatted.to_html()

    category_counts = df['Category'].value_counts()
    fig_bar = px.bar(category_counts, x=category_counts.index, y=category_counts.values, title="Products per Category")

    price_trends = df.groupby('Category')['Sale Price'].mean()
    fig_line = px.line(price_trends, x=price_trends.index, y=price_trends.values, title="Average Sale Price per Category")

    le = LabelEncoder()
    df['Category_encoded'] = le.fit_transform(df['Category'])
    X = df[['Category_encoded']].values
    y = df['Sale Price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Price', 'y': 'Predicted Price'}, title="Actual vs Predicted Prices")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Evaluate model by categorizing prices into bins
    bins = np.linspace(0, max(max(y_test), max(y_pred)), num=10)
    y_test_binned = np.digitize(y_test, bins)
    y_pred_binned = np.digitize(y_pred, bins)

    accuracy = accuracy_score(y_test_binned, y_pred_binned)
    precision = precision_score(y_test_binned, y_pred_binned, average='weighted', zero_division=0)
    recall = recall_score(y_test_binned, y_pred_binned, average='weighted')
    f1 = f1_score(y_test_binned, y_pred_binned, average='weighted')
    conf_matrix = confusion_matrix(y_test_binned, y_pred_binned)

    metrics_html = f"""
    <h3>Regression Metrics</h3>
    <p>Mean Squared Error (MSE): {mse:.2f}</p>
    <p>Root Mean Squared Error (RMSE): {rmse:.2f}</p>
    <p>Mean Absolute Error (MAE): {mae:.2f}</p>
    <p>R-squared (R²): {r2:.2f}</p>
    <h3>Classification Metrics</h3>
    <p>Accuracy: {accuracy:.2f}</p>
    <p>Precision: {precision:.2f}</p>
    <p>Recall: {recall:.2f}</p>
    <p>F1 Score: {f1:.2f}</p>
    <h3>Confusion Matrix</h3>
    <pre>{conf_matrix}</pre>
    """

    bar_html = fig_bar.to_html(full_html=False)
    line_html = fig_line.to_html(full_html=False)
    scatter_html = fig_scatter.to_html(full_html=False)

    return render_template('index.html', summary=summary_html, bar_html=bar_html, line_html=line_html, scatter_html=scatter_html, metrics_html=metrics_html)

if __name__ == '__main__':
    insighter("https://saricastudio.com")
    app.run(debug=True)
