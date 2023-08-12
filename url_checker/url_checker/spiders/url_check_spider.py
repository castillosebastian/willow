import scrapy
import os
import configparser
import polars as pl
import logging

config = configparser.ConfigParser()
config.read("config.ini")
#home_dir = config["main"]["HOME_DIR"]
#os.chdir(home_dir)


# Fix path
urls = pl.read_csv("/home/sebacastillo/willow/data/portals.csv")['newsportalurl'].to_list()

class URLCheckSpider(scrapy.Spider):

    name = 'url_check'
    
    start_urls = urls

    def parse(self, response):
        self.logger.info('Got response from %s with status %s', response.url, response.status)
