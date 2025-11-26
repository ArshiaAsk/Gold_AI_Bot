from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd


driver = webdriver.Firefox()
driver.get('https://www.tgju.org/profile/geram18/history')
driver.maximize_window()


time.sleep(5)
end_of_scroll = driver.execute_script('return document.body.scrollHeight')
while True:
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    my_scroll = driver.execute_script('return document.body.scrollHeight')
    if my_scroll == end_of_scroll:
        break
    end_of_scroll = my_scroll    


data = []

try:
    
    while True:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="DataTables_Table_0"]'))
        )

        rows = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="DataTables_Table_0"]/tbody/tr')))

        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            row_data = [col.text.strip() for col in cols]
            data.append(row_data)

        next_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'DataTables_Table_0_next')))

        if 'disabled' in next_button.get_attribute('class'):
            break

        driver.execute_script("arguments[0].click();", next_button)
        # click_count += 1
        
        time.sleep(2)

except Exception as e:
    print('خطا:', e)

finally:
    driver.quit()


columns = ['open','low','high','close','change','percent','date1','date2']
df = pd.DataFrame(data[1:], columns=columns)
df.to_csv('gold_price.csv', index=False)
info = pd.read_csv('gold_price.csv')
