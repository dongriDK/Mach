# -*- coding:utf-8 -*-
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller

import datetime

data = {	
	"LIST" : {
		"malwaredomainlist" : {
			"url" : "https://www.malwaredomainlist.com/mdl.php",
			"wait_class" : "ContentBox"
		},
		"mirror1.malwaredomain" : {
			"url" : "http://mirror1.malwaredomains.com/",
			"wait_class" : "abcde"
		},
		"contagio malwaredump" : {
			"url" : "http://contagiodump.blogspot.com/",
			"wait_class" : "content-fauxcolumns"
		},
		"contagio mobile malwaredump" : {
			"url" : "http://contagiominidump.blogspot.com",
			"wait_class" : "content-fauxcolumns"
		},
		"kernelmode" : {
			"url" : "https://www.kernelmode.info/forum/viewforum740d.html?f=21",
			"wait_class" : "space10"
		},
		"urlvoid" : {
			"url" : "http://blog.urlvoid.com",
			"wait_class" : "wrapper"
		},
		"malware intelligence" : {
			"url" : "https://twitter.com/jorgemieres",
			"wait_class" : "css-1dbjc4n r-13awgt0 r-12vffkv"
		},
		"any run" : {
			"url" : "https://app.any.run/submissions",
			"wait_class" : "open"
		},
		"DAS malwek" : {
			"url" : "https://das-malwerk.herokuapp.com/",
			"wait_class" : "container"
		},
		"Hybrid analysis" : {
			"url" : "https://www.hybrid-analysis.com/file-collections",
			"wait_class" : "submissions-table-container"
		},
		"malshare" : {
			"url" : "https://malshare.com",
			"wait_class" : "container"
		},
		"theZoo" : {
			"url" : "https://github.com/ytisf/theZoo/tree/master/malwares/Binaries",
			"wait_class" : "repository-content "
		},
		"objective" : {
			"url" : "https://objective-see.com/malware.html",
			"wait_class" : "wrapper"
		},
		"packet total" : {
			"url" : "https://packettotal.com/",
			"wait_class" : "text-center"
		},
		"URLhaus" : {
			"url" : "https://urlhaus.abuse.ch/api",
			"wait_class" : "list-group"
		}
	}
}

def Init_selenium():
    # chromedriver = "C://chromedriver/chromedriver.exe"
    chromedriver_autoinstaller.install()
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--verbose")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("window-size=1920, 1080")
    chrome_options.add_experimental_option("prefs", {
    "download.default_directory" : "C:/Users/STSC/Desktop/craw/",
    "download.prompt_for_download" : False,
    "download.directory_upgrade" : True,
    "safebrowsing_for_trusted_sources_enabled" : False,
    "safebrowsing.enabled" : False
    })
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(1000,1000)
    return driver
    
if __name__ == "__main__":
    driver = Init_selenium()
    today = str(datetime.date.today())
    f = open("./error_site.txt", "a")
    for i in data["LIST"]:
        url = data["LIST"].get(i).get('url')
        driver.get(url)
        try:
            if i == "mirror1.malwaredomain":
                check = driver.find_element_by_xpath("//p[2]").text
                if "malwaredomains.com" not in check:
                    raise TimeoutException
            else:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//*[@class='"+data["LIST"].get(i).get("wait_class")+"']")))
        except TimeoutException as ex:
            print("url : {}".format(url))
            f.write(today+"\t"+url+"\n")
    f.close()
    driver.quit()
