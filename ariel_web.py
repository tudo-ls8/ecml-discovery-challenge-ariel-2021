import time

from selenium import webdriver
from selenium.webdriver import DesiredCapabilities


class ARIEL_PAGE:
    LOGIN = 'https://www.ariel-datachallenge.space/accounts/login/'
    PERSONAL = 'https://www.ariel-datachallenge.space/ML/personal/'
    UPLOAD = 'https://www.ariel-datachallenge.space/ML/upload/'
    LEADERBOARD = 'https://www.ariel-datachallenge.space/ML/leaderboard/'


def fill_input_with(driver, input_id, value):
    elem = driver.find_element_by_id(input_id)
    elem.clear()
    elem.send_keys(value)


def find_btn_by_string(driver, btn_label):
    elements = driver.find_elements_by_class_name('btn')
    for elem in elements:
        if elem.text == btn_label:
            return elem


def login(driver, username, password):
    driver.get(ARIEL_PAGE.LOGIN)

    fill_input_with(driver, 'id_username', username)
    fill_input_with(driver, 'id_password', password)

    login_btn = find_btn_by_string(driver, 'Login')
    login_btn.click()


def get_driver(url=None):
    if url is None:
        return webdriver.Firefox()
    else:
        return webdriver.Remote(url, DesiredCapabilities.FIREFOX)


if __name__ == '__main__':
    driver = webdriver.Firefox()
    login(driver, 'TheReturnOfBasel321', 'Gwkilab123')

    time.sleep(1)

    driver.get(ARIEL_PAGE.PERSONAL)

    time.sleep(1)

    elem = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div/div/table/tbody/tr[1]/td[2]')
    print(elem)
    print(elem.text)

    elems = driver.find_elements_by_class_name('tbl')
    print(elems)
    for elem in elems:
        print(elem)

    time.sleep(10)
    driver.quit()
