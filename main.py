import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
import re

def insighter(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        shop_index_link = soup.find('a', href='/shop-1')
        if shop_index_link:
            # Combine the relative URL with the base URL
            shop_index_url = urljoin(url, shop_index_link['href'])
            
            # Visit the "shop index" page
            response = requests.get(shop_index_url)
            
            if response.status_code == 200:
                # Open a csv to write at the end
                with open('shop.csv', mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Title', 'Product Price', 'ProductItem-details-excerpt', "Type","Category"])

                    # Listing all product items
                    soup = BeautifulSoup(response.content, 'html.parser')
                    shop_blocks = soup.find_all('a',class_='grid-item-link product-lists-item')
                    
                    # Formatting the acquired links to page links
                    product_urls = []
                    for shop_block in shop_blocks:
                        product_link = shop_block.get('href')
                        product_url = "https://saricastudio.com" + product_link  
                        product_urls.append(product_url)
                    
                    # Visiting all product pages
                    for product_url in product_urls:
                            response = requests.get(product_url)
                            if response.status_code == 200:

                                # Scrapping the page
                                soup = BeautifulSoup(response.content, 'html.parser')
                                all_page = soup.find(class_='content')
                                articles = soup.find_all('article')
                                tags = ""
                                if len(articles) >= 2:
                                    # Get the second article element
                                    second_article = articles[1]  # Index 1 for the second element since indexing starts from 0

                                    # Get the class attribute of the second <article> element
                                    class_name = second_article.get('class')
                                    if class_name:
                                        class_name_text = ' '.join(class_name)  # Convert list of class names to string
                                        
                                        # Extract tags from class attribute using regex
                                        tags = re.findall(r'tag-(\w+)', class_name_text)

                                        # If there are tags, concatenate them with commas
                                        if tags:
                                            tags_text = ', '.join(tags)
                                        else:
                                            print("No tags found.")
                                    else:
                                        print("No class attribute found for the second <article> element.")
                                else:
                                    print("There is no second <article> element on the webpage.")                               
                                
                                #Scraping the product title
                                product_item_title = all_page.find('h1', class_='ProductItem-details-title').text.strip()
                                print(product_item_title + "is being scrapped")
                                
                                #Scraping the product price
                                product_price = all_page.find('div', class_='product-price').text.strip()
                                
                                #Scraping the product details
                                product_item_details = all_page.find('div', class_='ProductItem-details-excerpt').text.strip()
                                
                                #Scraping the all product labels
                                element = all_page.find('div', class_='variant-radiobtn-wrapper')
                                if element:
                                    labels = ' '.join(label.text.strip() for label in element.find_all('label'))
                                else:
                                    # Handle the case where the element is not found
                                    labels = "N/A"

                                # Format the category
                                category = ', '.join(tags)
                                writer.writerow([product_item_title, product_price, product_item_details, labels, category])
                                               
            else:
                print("Failed to retrieve the recipe index page")
        else:
            print("Recipe index link not found")
    else:
        print("Failed to retrieve the webpage")

# Main
url = "https://saricastudio.com"
insighter(url)