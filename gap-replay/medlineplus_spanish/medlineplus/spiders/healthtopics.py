import scrapy

class HealthTopicSpider(scrapy.Spider):
    name = 'healthtopics'
    allowed_domains = ['medlineplus.gov']
    start_urls = ['https://medlineplus.gov/spanish/druginformation.html']

    def parse(self, response):
        browse = response.xpath(".//ul[@class='alpha-links']//li")
        for link in browse:
            value = link.xpath("./a/@href").get()
            yield response.follow(url=value, callback=self.parse_healthtopics)

    def parse_healthtopics(self, response):
        healthtopics = response.xpath(".//ul[@id='index']//li")
        for healthtopic in healthtopics:
            name = healthtopic.xpath("./span/text()").get()
            healthtopics_link = healthtopic.xpath("./a/@href").get()
            yield response.follow(url=healthtopics_link, callback=self.healthtopics_info, meta={'healthtopics_name': name})

    def healthtopics_info(self, response):
        healthtopics_name = response.request.meta['healthtopics_name']
        info_1 = response.xpath(".//div[@id='why']//p/text()").get()
        info_2 = response.xpath(".//div[@id='how']//p/text()").getall()
        info_3 = response.xpath(".//div[@id='other-uses']//p/text()").getall()
        info_4 = response.xpath(".//div[@id='precautions']//li/text()").getall()
        info_5 = response.xpath(".//div[@id='special-dietary']//p/text()").getall()
        info_6 = response.xpath(".//div[@id='if-i-forget']//p/text()").getall()
        info_7 = response.xpath(".//div[@id='side-effects']//li/text()").getall()
        info_8 = response.xpath(".//div[@id='storage-conditions']//p/text()").getall()
        info_9 = response.xpath(".//div[@id='overdose']//p/text()").getall()
        info_10 = response.xpath(".//div[@id='other-information']//p/text()").getall()

        yield {
            'Healthtopics Name': healthtopics_name,
            'info 1': info_1,
            'info 2': info_2,
            'info 3': info_3,
            'info 4': info_4,
            'info 5': info_5,
            'info 6': info_6,
            'info 7': info_7,
            'info 8': info_8,
            'info 9': info_9,
            'info 10': info_10}
