import scrapy
import os
import configparser
import polars as pl
import logging
import time

config = configparser.ConfigParser()
config.read("/home/sebacastillo/willow/config.ini")
home_dir = config["main"]["HOME_DIR"]
os.chdir(home_dir)
filename = os.path.basename(__file__)

# Fix path
urls = pl.read_csv("data/portals.csv")['newsportalurl'].to_list()

# logging
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
name_log = "logs/urlchecker" + start_time + ".log"
logging.basicConfig(
        filename=name_log,
        level=logging.INFO,
        format=f"%(asctime)s-{filename}-%(levelname)s-%(message)s",
    )

class URLCheckSpider(scrapy.Spider):

    name = 'url_check'
    
    start_urls = urls

    def parse(self, response):
        self.logger.info('Got response from %s with status %s', response.url, response.status)
