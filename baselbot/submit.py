"""Submit a predictions file"""
import argparse
import re
import time
from datetime import datetime, timedelta
from pytz import timezone
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

LOGIN_URL = 'https://www.ariel-datachallenge.space/accounts/login/'
PERSONAL_URL = 'https://www.ariel-datachallenge.space/ML/personal/'
UPLOAD_URL = 'https://www.ariel-datachallenge.space/ML/upload/'
LEADERBOARD_URL = 'https://www.ariel-datachallenge.space/ML/leaderboard/'
TIMELEFT_XPATH = "/html/body/div[3]/div[2]/div/h3"

def submit(url, username, password, path, wait=False):
    driver = get_driver(url)
    login(driver, username, password)

    # go to the upload page
    driver.get(PERSONAL_URL)
    driver.get(UPLOAD_URL)

    # Upload data
    print(f"Uploading {path} as {username} via {url}")
    driver.find_element_by_id('id_data').send_keys(path)
    elem = find_btn_by_string(driver, 'Upload')
    elem.click()

    print(f"Time of clicking the upload button: {fmt_time(datetime.now())}...")

    # wait until the button disappears
    timeleft = -1 # need to wait until the next submission opens?
    try:
        WebDriverWait(driver, 300).until(ec.staleness_of(elem))
        time.sleep(5) # ensure that the page is actually loaded

        # is the upload still forbidden?
        timeleft_elem = driver.find_elements_by_xpath(TIMELEFT_XPATH)
        if len(timeleft_elem) > 0:
            print(f"UPLOAD FORBIDDEN: next upload allowed in {timeleft_elem[0].text}")
            timeleft = seconds_left(timeleft_elem[0].text)
        else:
            print(f"After 5sec, the upload SEEMS alright. Waiting for another 60sec")
            time.sleep(60)

            # try again
            timeleft_elem = driver.find_elements_by_xpath(TIMELEFT_XPATH)
            if len(timeleft_elem) > 0:
                print(f"UPLOAD FORBIDDEN: next upload allowed in {timeleft_elem[0].text}")
                timeleft = seconds_left(timeleft_elem[0].text)
            else:
                print(f"The upload still SEEMS alright, even after 65sec.")

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        driver.close()

    # want to wait until the next upload is allowed?
    if wait and timeleft > 0:
        until = datetime.now() + timedelta(seconds=timeleft)
        print(f"Waiting until {fmt_time(until)}...")
        try:
            time.sleep(timeleft)
        except (KeyboardInterrupt, SystemExit):
            print("Waiting aborted")
            return # only resubmit if the waiting is not aborted
        submit(url, username, password, path, wait)

def fmt_time(time):
    return time.astimezone(timezone("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S CEST")

def seconds_left(timeleft_text):
    timeleft = 30 # always add 30 seconds
    for regex, factor in [("\d+ hours", 60*60), ("\d+ minutes", 60)]:
        m = re.findall(regex, timeleft_text)
        if len(m) > 0:
            timeleft += factor * int(re.findall("\d+", m[0])[0])
    return timeleft

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
    driver.get(LOGIN_URL)
    fill_input_with(driver, 'id_username', username)
    fill_input_with(driver, 'id_password', password)
    login_btn = find_btn_by_string(driver, 'Login')
    login_btn.click()

def get_driver(url=None):
    if url is None:
        return webdriver.Firefox()
    else:
        return webdriver.Remote(url, webdriver.DesiredCapabilities.FIREFOX)

def get_key(path):
    try:
        with open(path, 'r') as f:
            return f.read().replace('\n', '')
    except:
        raise ValueError(f"Cannot read the password from {path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit a predictions file', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--url', default=None, help="Optional remote driver, e.g. 'http://localhost:4444/wd/hub'") 
    parser.add_argument('--user', default="TheReturnOfBasel321", help="the name of the user to log in as")
    parser.add_argument('--key', default="/mnt/home/.BASELBOT", help='the file to read the user\'s password from')
    parser.add_argument('--wait', action="store_true", help='whether to sleep until the next submission is allowed')
    parser.add_argument('input', help='the file to submit')
    args = parser.parse_args()
    submit(args.url, args.user, get_key(args.key), args.input, args.wait)
