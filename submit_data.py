import argparse
import time

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from ariel_web import get_driver, ARIEL_PAGE, login, find_btn_by_string

ARIEL_LOGIN_PAGE = 'https://www.ariel-datachallenge.space/accounts/login/'
ARIEL_PERSONAL_PAGE = 'https://www.ariel-datachallenge.space/ML/personal/'
ARIEL_UPLOAD_PAGE = 'https://www.ariel-datachallenge.space/ML/upload/'


def main(args):
    print(f"Creating a driver for {args.url}")
    driver = get_driver(args.url)

    print(f"Logging in as {args.username}")
    login(driver, args.username, args.password)

    # GO to upload page
    driver.get(ARIEL_PAGE.PERSONAL)
    driver.get(ARIEL_PAGE.UPLOAD)

    # Upload data
    print(f"Uploading {args.input}")
    driver.find_element_by_id('id_data').send_keys(args.input)

    # Submit data
    elem = find_btn_by_string(driver, 'Upload')
    elem.click()

    # print('CONTROL RETURNED')
    print(f"Waiting for upload... (300s max)")
    try:
        # wait until the button leaves
        WebDriverWait(driver, 300).until(EC.staleness_of(elem))

        # need to wait for next upload?
        elems_timeleft = driver.find_elements_by_xpath("/html/body/div[3]/div[2]/div/h3")
        if len(elems_timeleft) > 0:
            print(f"ERROR: The next upload is only possible in {elems_timeleft[0].text}")
        else:
            print(f"The upload SEEMS alright") # TODO check

    # Shutdown browser
    finally:
        print(f"Closing the driver")
        driver.close()


# --username TheReturnOfBasel321 --password Gwkilab123 --input /tmp/test.data
# --username ecmlreviewer63 --password Testaccount --input /tmp/test.data
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--url', default=None)  # default='http://localhost:4444/wd/hub'

    parser.add_argument('--username', required=True)
    parser.add_argument('--password', required=True)
    parser.add_argument('-i', '--input', required=True)

    args = parser.parse_args()
    main(args)
