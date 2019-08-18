# -*- coding:utf-8 -*-
from tkinter import *
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import threading
import time
import telegram

# 분반까지 해야함??
# 봉사 15904 1-0-2
url = "http://sugang.hannam.ac.kr/sugang/common/loginForm.do"

Login_OK = False
my_token = "607900915:AAHzxIB7OKud7KXzzf_gCCfW8qJd1UaIVoM"
bot = telegram.Bot(token = my_token)
try:
    chat_id = bot.getUpdates()[0].message.chat.id
except:
    chat_id = 683947176

class Sugang():
    def __init__(self, master):
        self.root = master
        self.idText = Text(self.root, width=10, height=1)
        self.pwText = Text(self.root, width=15, height=1)
        self.classText = Text(self.root, width=20, height=2)
        self.idLabel = Label(self.root, text="학번 입력 : ")
        self.pwLabel = Label(self.root, text="비밀번호 입력 : ")
        self.classLabel = Label(self.root, text="학수번호 입력 : ")
        self.exLabel = Label(self.root, text="*ex) 12345 95123 78945")
        self.startButton = Button(self.root, text="Start", fg="red", command=self.Sugang_Macro_Thread_Start)

        self.idLabel.grid(row=0, column=0)
        self.idText.grid(row=0, column=1)
        self.pwLabel.grid(row=1, column=0)
        self.pwText.grid(row=1, column=1)
        self.classLabel.grid(row=2, column=0)
        self.classText.grid(row=2, column=1)
        self.exLabel.grid(row=3, column=1)
        self.startButton.grid(row=3, column=2)

    def Sugang_Macro_Thread_Start(self):
        t = threading.Thread(target=self.Start_Sugang_Macro)
        t.start()

    def Start_Sugang_Macro(self):
        global Login_OK
        if Login_OK == False:
            self.Login_Hannam_Sugang()
        # WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, "//iframe[starts-with(@src, 'http://sugang.hannam.ac.kr/sugang/common/frame.do?method=left&UID=719a4369:1690f0fdf9d:-7b7c')]")))
        a = self.driver.execute_script("return NetFunnel.TS_BYPASS")
        print("before NetFunnel : {}".format(a))
        self.driver.execute_script("NetFunnel.TS_BYPASS = true")
        b = self.driver.execute_script("return NetFunnel.TS_BYPASS")
        print("after NetFunnel : {}".format(b))
        # self.driver.switch_to.frame('fraemLeft')
        # self.driver.find_element_by_name('img_baguni').click()
        # self.driver.switch_to.default_content() #프레임안에서 나오기
        self.driver.switch_to.frame('frameMain2')
        self.Apply_Class()

    def Login_Hannam_Sugang(self):
        global Login_OK
        id = self.idText.get("1.0", END)
        pw = self.pwText.get("1.0", END)
        self.driver = webdriver.Chrome(executable_path=r'C:\Python27\ChromeDriver\chromedriver.exe')
        self.driver.get(url)
        time.sleep(0.5)
        Login_OK = True
        while 1:
            # self.driver.find_element_by_name('memberNo').send_keys("20140765")
            # self.driver.find_element_by_name('password').send_keys("QboWdg@12")
            self.driver.find_element_by_name('memberNo').send_keys(id)
            self.driver.find_element_by_name('password').send_keys(pw)
            try:
                self.driver.execute_script("doLogin()")
                # print("A")
                # WebDriverWait(driver, 3).until(EC.alert_is_present())
                # print("B")
                Login_OK = True
                break
            except:
                # print("C")
                try:
                    alrt = self.driver.switch_to.alert
                    alrt.accept()
                    pass
                except:
                    break
                # print("D")
                Login_OK = True
                pass

    # 20896 24246 23778
    # 취전   생활it 스마
    # QboWdg@12
    def Apply_Class(self):
        # classList = sel   f.Calc_Sugang_List()
        classList = self.classText.get("1.0", END)[:-1].split(" ")
        # classList = self.classText.get("1.0", END).split(" ")
        while 1:
            print("\nclassList : {}".format(classList))
            for v in classList:
                time.sleep(0.5)
                # print("v : {}".format(v))
                try:
                    # print("A")
                    self.driver.find_element_by_xpath('//*[contains(@onclick,'+v+')]').click()
                    # WebDriverWait(driver, 5).until(EC.alert_is_present())
                    alrt = self.driver.switch_to.alert
                    # print("alrt.text : {}".format(alrt.text))
                    if alrt.text == "신청 되었습니다." or "이미 신청되어 있는 과목입니다." in alrt.text:
                        # print("G")
                        alrt.accept()
                        del classList[v]
                        bot.sendMessage(chat_id=chat_id, text="취켓팅 성공!!!!!!!!!")
                    else:
                        # print("F")
                        alrt.accept()
                except:
                    try:
                        # WebDriverWait(driver, 5).until(EC.alert_is_present())
                        alrt = self.driver.switch_to.alert
                        if alrt.text == "신청 되었습니다." or "이미 신청되어 있는 과목입니다." in alrt.text:
                            # print("D")
                            alrt.accept()
                            del classList[v]
                        else:
                            # print("E")
                            alrt.accept()
                    except:
                        pass
                    pass



    # def Calc_Sugang_List(self):
    #     classText = self.classText.get("1.0", END).split(" ")
    #     # classTest_Split = classTest.split(" ")
    #     return classTest
            # else:
            #     print("G")
            #     try:
            #         driver.execute_script("doLogin();")
            #     except:
            #         pass
            # flag = False


                # alrt = driver.switch_to_alert()
                # alrt.accept()
                #
                # Login_OK = True
                # alrt = driver.switch_to_alert()
                # alrt.accept()
                # print("B")
                # print("C")
                # Login_OK = True
            # driver.switch_to.frame('frameMain2') ##총 강좌 리스트
            # driver.switch_to.frame('frameMain3') ##선택 강좌 리스트
            # break



if __name__ == '__main__':
    root = Tk()
    root.title("Sugang")
    guiWindow = Sugang(root)
    root.resizable(False,False)
    root.mainloop()
