import scrapy
from scrapy.http import Response
from ..items import SicrawlerItem


class MySpider(scrapy.Spider):
    name = 'myspider'
    allowed_domains = ['foundation.wikimedia.org',
                       'wikimediafoundation.org',
                       'www.mediawiki.org',
                       'meta.wikimedia.org',
                       'creativecommons.org',
                       'en.wikipedia.org',
                       'phl.upr.edu',
                       'www.tibetanyouthcongress.org',
                       'www.studentsforafreetibet.org',
                       'jewishamericansongster.com',
                       'www.jta.org',
                       'www.klezmershack.com',
                        'www.acsmedchem.org',
                       'www.nap.edu',
                       'www.formula1.com',
                       'www.fifa.com',
                       'newrepublic.com',
                       'politicalticker.blogs.cnn.com',
                       'www.hollywoodreporter.com',
                       'www.nydailynews.com',
                       'mobile.nytimes.com',
                       'www.espn.com',
                       'www.newsweek.com',
                       'money.cnn.com',
                       'apnews.com',
                       'www.economist.com',
                       'www.cnbc.com',
                       'www.vox.com',
                       'www.nbcnews.com',
                       'www.donaldjtrump.com',
                       'www.newspapers.com',
                       'donaldjtrump.com',
                       'whitehouse.gov',
                       'keras.io'
                       ]

    start_urls = ['https://en.wikipedia.org/wiki/Mathematics',
                  'https://en.wikipedia.org/wiki/Harry_Potter',
                  'https://en.wikipedia.org/wiki/Donald_Trump',
                  'https://en.wikipedia.org/wiki/Breast_cancer',
                  'https://en.wikipedia.org/wiki/Programming_language',
                  'https://en.wikipedia.org/wiki/Leonardo_da_Vinci',
                  'https://en.wikipedia.org/wiki/Sport',
                  'https://en.wikipedia.org/wiki/Convolutional_neural_network'
                  ]

    def parse(self,response:Response):
        # urlfile = open('urls', 'w', encoding='utf-8')

        item = SicrawlerItem()
        item['url'] = response.url
        item['body'] = response.css('p::text').getall()
        # item['body'] = response.body
        yield item

        links = response.css('a::attr(href)').getall()
        # print(f'La longitud de links es : {len(links)}')
        # print(response.urljoin(links[0]))
        for a in links:
            if a == '/': continue
            urljoined = response.urljoin(a)
            # urlfile.write(urljoined+'\n')
            yield scrapy.Request(urljoined, callback=self.parse)
        # urlfile.close()
