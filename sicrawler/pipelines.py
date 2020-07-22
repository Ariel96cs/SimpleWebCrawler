# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy.exceptions import DropItem
import json
from nltk import word_tokenize
import re
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class SicrawlerPipeline(object):
    def process_item(self, item, spider):
        if not item.get('body'):
            raise DropItem(f'Missing body property in {item}')
        return item

class BodyCleaner:
    def tolower(self,body):
        result = []
        for text in body:

            a = ' '.join([w.lower() for w in word_tokenize(text)])
            result.append(a)
        body = result
        return body

    def remove_tags(self,body):
        result = []
        for text in body:
            clean_text = re.sub('<[^<]+?>','', text)
            result.append(clean_text)
        body = result
        return body

    def remove_numbers(self,body):
        result = []
        for text in body:
            clean_text = ''.join(c for c in text if not c.isdigit())
            result.append(clean_text)
        body = result
        return body

    def remove_punctuation(self,body):
        result = []
        for text in body:
            clean_text = ''.join(c for c in text if c not in punctuation)
            result.append(clean_text)
        body = result
        return body

    def lemmatize(self,body):
        '''
        Is based on The Porter Stemming Algorithm
        :param body: string list
        :return: body
        '''
        stopword = stopwords.words('english')
        wordnet_lemmatizer = WordNetLemmatizer()
        result = []
        for text in body:
            word_tokens = nltk.word_tokenize(text)
            lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens if word not in stopword]
            result += lemmatized_word
        body = result
        return body

    def process_item(self,item,spider):
        body = item['body']
        body = self.tolower(body)
        body = self.remove_tags(body)
        body = self.remove_numbers(body)
        body = self.remove_punctuation(body)
        body = self.lemmatize(body)
        item['body'] = body
        return item

class JsonWriterPipeline:
    def open_spider(self,spider):
        self.file = open(f'{spider.name}_items.jl','w',encoding='utf-8')

    def close_spider(self,spider):
        self.file.close()

    def process_item(self,item,spider):
        if not item.get('body'):
            raise DropItem(f'Missing body property in {item}')

        line = json.dumps(dict(item))+'\n'
        self.file.write(line)
        return item

class DuplicatesPipeline:

    def __init__(self):
        self.url_seen = set()

    def process_item(self,item,spider):
        if item['url'] in self.url_seen:
            raise DropItem(f'Item already visited {item}')
        else:
            self.url_seen.add(item['url'])
        return item