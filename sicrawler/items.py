# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from w3lib.html import remove_tags
from scrapy.loader.processors import MapCompose


class SicrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    url = scrapy.Field()
    body = scrapy.Field(
        input_processor=MapCompose(remove_tags, lambda elem: elem.split())
    )
