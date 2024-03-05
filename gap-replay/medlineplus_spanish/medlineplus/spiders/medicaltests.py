import scrapy


class MedicaltestsSpider(scrapy.Spider):
    name = 'medicaltests'
    allowed_domains = ['medlineplus.gov']
    start_urls = ['https://medlineplus.gov/lab-tests/']

    def parse(self, response):
        pass
