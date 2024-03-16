import scrapy

class MedlinePlusSpider(scrapy.Spider):
    name = 'medlineplus_spider'
    allowed_domains = ['medlineplus.gov']
    start_urls = [
        'https://medlineplus.gov/spanish/encyclopedia.html',
        'https://medlineplus.gov/spanish/pruebas-de-laboratorio/',
        'https://medlineplus.gov/spanish/genetica/',
        'https://medlineplus.gov/spanish/healthtopics.html',
        'https://medlineplus.gov/spanish/complementaryandintegrativemedicine.html',
        'https://medlineplus.gov/spanish/herbalmedicine.html',
        'https://medlineplus.gov/spanish/druginformation.html'
    ]

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
        titles = response.xpath("//h1/text()").getall()
        subtitles = response.xpath("//h2/text()").getall()
        paragraphs = response.xpath("//p/text()").getall()
        
        paragraphs = response.xpath("//p/text()").getall()
        cleaned_paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

        yield {
            'Healthtopics Name': healthtopics_name,
            'titles': titles,
            'subtitles': subtitles,
            'paragraphs': paragraphs
        }