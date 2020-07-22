import scrapy
from scrapy.http import Response
from ..items import SicrawlerItem


class MySpider(scrapy.Spider):
    name = 'singlePageSpider'

    def parse(self,response:Response):

        item = SicrawlerItem()
        item['url'] = response.url
        item['body'] = response.css('p::text').getall()
        yield item

