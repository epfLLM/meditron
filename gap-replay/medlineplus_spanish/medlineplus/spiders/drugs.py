import scrapy


class DrugsSpider(scrapy.Spider):
    name = 'drugs'
    allowed_domains = ['medlineplus.gov']
    #start_urls = ['https://medlineplus.gov/druginformation.html']
    start_urls = ['https://medlineplus.gov/spanish/druginformation.html']
    def parse(self, response):
        browse = response.xpath(".//ul[@class='alpha-links']//li")
        for link in browse:
            value = link.xpath("./a/@href").get()
            yield response.follow(url=value, callback=self.parse_drugs)

    def parse_drugs(self, response):
        drugs = response.xpath(".//ul[@id='index']//li")
        for drug in drugs:
            name = drug.xpath("./span/text()").get()
            drug_link = drug.xpath("./a/@href").get()
            yield response.follow(url=drug_link, callback=self.drug_info, meta={'drug_name': name})

    def drug_info(self, response):
        drug_name = response.request.meta['drug_name']
        info_1 = response.xpath(".//div[@id='why']//p/text()").get()
        info_2 = response.xpath(".//div[@id='how']//p/text()").getall()
        info_3 = response.xpath(".//div[@id='other-uses']//p/text()").getall()
        info_4 = response.xpath(
            ".//div[@id='precautions']//li/text()").getall()
        info_5 = response.xpath(
            ".//div[@id='special-dietary']//p/text()").getall()
        info_6 = response.xpath(".//div[@id='if-i-forget']//p/text()").getall()
        info_7 = response.xpath(
            ".//div[@id='side-effects']//li/text()").getall()
        info_8 = response.xpath(
            ".//div[@id='storage-conditions']//p/text()").getall()
        info_9 = response.xpath(".//div[@id='overdose']//p/text()").getall()
        info_10 = response.xpath(
            ".//div[@id='other-information']//p/text()").getall()

        yield {
            'Drug Name': drug_name,
            'Why is this medication prescribed?': info_1,
            'How should this medicine be used?':  info_2,
            'Other uses for this medicine': info_3,
            'What special precautions should I follow?': info_4,
            'What special dietary instructions should I follow?': info_5,
            'What should I do if I forget a dose?': info_6,
            'What side effects can this medication cause?': info_7,
            'What should I know about storage and disposal of this medication?': info_8,
            'In case of emergency/overdose': info_9,
            'What other information should I know?': info_10}
