import argparse
import smtplib
import time
from email.message import EmailMessage

import pandas as pd

from ariel_web import login, ARIEL_PAGE, get_driver


def get_key(path):
    try:
        with open(path, 'r') as f:
            return f.read().replace('\n', '')
    except:
        raise ValueError(f"Cannot read the password from {path}")


def get_current_score(driver):
    driver.get(ARIEL_PAGE.PERSONAL)
    try:
        elem = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div/div/table/tbody/tr[1]/td[2]')
    except:
        return -1
    return elem.text


def get_best_score(driver):
    driver.get(ARIEL_PAGE.PERSONAL)
    time.sleep(1)
    try:
        elem = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div/div/table/tbody/tr[2]/td[2]')
    except:
        return -1
    return elem.text


def get_best_date(driver):
    driver.get(ARIEL_PAGE.PERSONAL)
    time.sleep(1)
    try:
        elem = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div/div/table/tbody/tr[3]/td[2]')
    except:
        return -1
    return elem.text


def get_latest_date(driver):
    driver.get(ARIEL_PAGE.PERSONAL)
    time.sleep(1)
    try:
        elem = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div/div/table/tbody/tr[4]/td[2]')
    except:
        return -1
    return elem.text


def main(args):
    tblz = pd.read_html(ARIEL_PAGE.LEADERBOARD)

    tbl = tblz[0]

    place = tbl[tbl['Name'] == args.username]

    top10 = tbl.head(10)
    top10 = top10.to_markdown()

    msg = EmailMessage()

    driver = get_driver(args.url)

    login(driver, args.username, args.password)

    best_score = get_best_score(driver=driver)
    current_score = get_current_score(driver=driver)

    latest_date = get_latest_date(driver)
    best_date = get_best_date(driver)

    msg.set_content(
        'Your current position: {}\n\nBest score: {} ({}) \n\nCurrent score: {} ({})\n\n{}'.format(place['Rank'].item(),
                                                                                                   best_score, best_date,
                                                                                                   current_score, latest_date,
                                                                                                   top10))

    recipients = ['lukas.heppe@tu-dortmund.de', 'mirko.bunse@tu-dortmund.de']

    msg['Subject'] = f'ECML PKDD 2021 - Ariel challenge'
    msg['From'] = 'lukas.heppe@tu-dortmund.de'
    msg['To'] = ", ".join(recipients)

    server = smtplib.SMTP(args.smtp_host, args.smtp_port)

    server.ehlo()
    server.starttls()
    server.ehlo()

    key = get_key(args.mailpassword)

    server.login(args.maillogin, key)

    server.send_message(msg)
    server.quit()

    driver.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--url', default=None)  # default = 'http://localhost:4444/wd/hub'

    parser.add_argument('--username', type=str, default='TheReturnOfBasel321')
    parser.add_argument('--password', type=str, default='Gwkilab123')

    parser.add_argument('--maillogin', type=str, default='smluhepp')
    parser.add_argument('--mailpassword', type=str, default='/home/lukas/.docker/secrets/mail.txt')

    parser.add_argument('--smtp-host', type=str, default='unimail.tu-dortmund.de')
    parser.add_argument('--smtp-port', type=int, default=25)

    args = parser.parse_args()
    main(args)
