'''
Script for 12/16 Chrome (Firefox, experimentally)-based guideline scrapers.

Sources supported:
AAFP - American Academy of Family Physicians
CCO - Cancer Care Ontario
CDC - Centers for Disease Control and Prevention
CMA - Canadian Medical Association
CPS - Canadian Paediatric Society
Drugs.com
GuidelineCentral
ICRC - International Committee of the Red Cross
IDSA - Infectious Diseases Society of America
MAGIC - Making GRADE the Irresistible Choice
SPOR - Strategy for Patient-Oriented Research
WHO - World Health Organization

The other 4/16 are Typescript-based (MayoClinic, NICE, RCH, WikiDoc) and can be found in scrapers/.
"""

"""
Important note to users:
  Scraping logic will inevitably rot over time, and probably quickly.

  These are best-effort contemporary (November 2023) reconstructions of our original data
  collection effort, which took place some months before. As you can see in places
  the logic is fairly hacky.

  We will support interested users in the immediate period after the code
  release, but it's impossible to imagine supporting the scraping logic
  beyond that.

  Put brutally: if you are reading this after about January 2024, it's almost
  inevitable that you'll have to patch up some of this logic.

  Best Wishes,

  Antoine, Alexandre, and Kyle
'''


# ------------------- Imports ------------------- #
import functools
import subprocess
import argparse
import os
import time
import json
import hashlib
import tarfile

from tqdm import tqdm
import markdownify
import scipdf
import requests

import PyPDF2

from selenium.common import exceptions
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import NoSuchElementException

# ------------------- Selenium Setup ------------------- #

DEFAULT_DRIVER = 'chrome'  # Switch to 'firefox' if you want to use Firefox instead of Chrome


def setup_firefox_driver(download_path):
    '''
    Set up Firefox driver instance with download path.
    '''
    profile = FirefoxOptions()
    profile.binary_location = "/usr/bin/firefox"
    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.dir", download_path)
    profile.set_preference("browser.download.useDownloadDir", True)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream")
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf,application/x-pdf")
    profile.set_preference("pdfjs.disabled", True)
    driver = webdriver.Firefox(options=profile)
    return driver


def setup_chrome_driver(
    include_experimental: bool = True,
    download_path: str = None,
    binary_location: str = None,
    driver_location: str = None,
    headless: bool = False,
    eager_mode: bool = False
):
    '''
    Set up Chrome driver instance with download path.
    '''
    chrome_options = ChromeOptions()

    if binary_location:
        chrome_options.binary_location = binary_location

    if eager_mode:
        chrome_options.page_load_strategy = 'eager'

    if headless:
        chrome_options.add_argument('--headless')

    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--dns-prefetch-disable")
    chrome_options.add_argument("--disable-gpu")

    if include_experimental:
        assert download_path is not None, "Download path must be provided if including experimental options."
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": download_path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False,
        })

    chrome_params = dict(options=chrome_options)

    if driver_location:
        webdriver_service = Service(driver_location)
        chrome_params.update(service=webdriver_service)

    driver = webdriver.Chrome(**chrome_params)

    return driver


# ------------------- Helper Functions ------------------- #

@functools.lru_cache
def start_grobid():
    process = subprocess.Popen("./serve_grobid.sh")


def pdf2text(src_path, dest_path):
    '''
    Convert all pdf files in src_path to text, then save as jsonl to dest_path.
    '''
    start_grobid()
    guidelines = []
    pdf_files = [file for file in os.listdir(src_path) if file.endswith(".pdf")]
    for file in tqdm(pdf_files):
        try:
            article = scipdf.parse_pdf_to_dict(os.path.join(src_path, file))
            text = str(article["abstract"]) + "\n".join(
                ["# " + sec['heading'] + "\n" + sec["text"] for sec in article["sections"]])
            guideline = {"file_name": file,
                         "text": text,
                         "doi": article["doi"]}
            guidelines.append(guideline)
        except Exception as e:
            print('Exception when parsing file: ', file, e)
    with open(dest_path, 'w') as f:
        for guideline in guidelines:
            f.write(f"{json.dumps(guideline)}\n")
    return guidelines


def scrape_SPOR_links():
    '''
    Helper to scrape CCO, CPS and SPOR links from a SPOR PDF.
    '''
    PDFFile = open("SPOR_links.pdf",'rb')
    PDF = PyPDF2.PdfReader(PDFFile)
    pages = len(PDF.pages)
    key = '/Annots'
    uri = '/URI'
    ank = '/A'
    print(pages)
    links = []
    for page in tqdm(range(pages)):
        pageSliced = PDF.pages[page]
        pageObject = pageSliced.get_object()
        if key in pageObject.keys():
            ann = pageObject[key]
            for a in ann:
                u = a.get_object()
                if ank in u.keys():
                    if uri in u[ank].keys():
                        links.append(u[ank][uri])
    links = list(set(links))
    return links

# ------------------- Scraper Classes ------------------- #


class Scraper():
    '''
    Web Scraper class to scrape clinical guidelines from a source.
    Downloads scraped articles to {path}/raw/{source}.jsonl
    '''

    def __init__(
        self,
        source: str,
        path: str,
        pdfs: bool = False,
        chrome_binary_location: str = None,
        chrome_driver_location: str = None
    ):
        setup_driver = setup_firefox_driver if DEFAULT_DRIVER == 'firefox' else setup_chrome_driver
        self.path = path
        self.source = source
        self.source_path = os.path.join(self.path, self.source)
        os.makedirs(self.source_path, exist_ok=True)
        self.pdfs = pdfs
        self.pdf_path = os.path.join(self.source_path, 'pdfs')
        self.links_path = os.path.join(self.source_path, 'links.txt')

        articles_dir = os.path.join(self.path, 'raw')
        os.makedirs(articles_dir, exist_ok=True)
        self.articles_path = os.path.join(articles_dir, f'{source}.jsonl')

        self.driver = setup_driver(
            include_experimental=self.pdfs,
            download_path=self.pdf_path if self.pdfs else None,
            binary_location=chrome_binary_location,
            driver_location=chrome_driver_location
        )

    def save_articles(self, articles, path=None):
        '''
        Save articles (list of dict) to self.articles_path (.jsonl)
        '''
        articles_path = self.articles_path if path is None else path
        with open(articles_path, 'w') as f:
            for article in articles:
                f.write(f"{json.dumps(article)}\n")

    def save_links(self, links):
        '''
        Save links (list of str) to self.links_path (.txt)
        '''
        links = list(set(links))
        with open(self.links_path, 'w') as f:
            for link in links:
                f.write(f"{link}\n")

    def reset_download_path(self):
        self.driver.execute_cdp_cmd(
            'Page.setDownloadBehavior', 
            {'behavior': 'allow', 'downloadPath': self.pdf_path})

    def _scrape_links(self):
        '''
        Scrape links from source to self.links_path.
        '''
        raise NotImplementedError

    def scrape_links(self):
        '''
        Scrape links from source to self.links_path.
        '''
        if os.path.exists(self.links_path):
            with open(self.links_path, "r") as f:
                return [i.strip() for i in f.readlines()]
        else:
            return self._scrape_links()

    def scrape_articles(self, links):
        '''
        Scrape articles from links to self.path/{source}/{source}.jsonl.
        '''
        raise NotImplementedError

    def scrape(self):
        '''
        Scrape articles to self.path/{source}/{source}.jsonl.
        1 - Download links to source articles.
        2 - Download articles from links.
        3 - Convert PDFs to text (if necessary)
        '''
        print(f"Stage 1: Scraping article links from {self.source}...")
        links = self.scrape_links()
        unique_links = list(set(links))
        print(f"Scraped {len(unique_links)} unique article links from {self.source}.")

        print(f"Stage 2: Scraping articles from {self.source}...")
        articles = self.scrape_articles(unique_links)

        if self.pdfs:
            print(f"Stage 3: Converting PDFs to text...")
            articles = pdf2text(self.pdf_path, self.articles_path)

        print(f"Saved {len(articles)} {self.source} articles to {self.articles_path}.")
        self.driver.quit()
        return articles


class AAFPScraper(Scraper):
    '''
    Source: AAFP - American Academy of Family Physicians (https://www.aafp.org/)
    '''
    def __init__(self, path, **kwargs):
        super().__init__('aafp', path, **kwargs)

    def _scrape_links(self):
        self.driver.get("https://www.aafp.org/family-physician/patient-care/clinical-recommendations/recommendations-by-topic.html")
        wait = WebDriverWait(self.driver, 10)
        elements = wait.until(EC.presence_of_all_elements_located((
            By.XPATH, "//a[(@class='adl-link') and (contains(@href, '/family-physician/patient-care/clinical-recommendations/recommendations-by-topic'))]")))
        topics = list(set([href.get_attribute("href") for href in elements]))
        links = []
        for topic in tqdm(topics):
            try:
                self.driver.get(topic)
                wait = WebDriverWait(self.driver, 10)
                elements = wait.until(EC.presence_of_all_elements_located((
                    By.XPATH, "//a[(@class='adl-link') and (contains(@href, '/family-physician/patient-care/clinical-recommendations/all-clinical-recommendations'))]")))
                links += [href.get_attribute("href") for href in elements]
            except:
                pass
        links = list(set(links))
        self.save_links(links)
        return links

    def scrape_articles(self, links):
        articles = []
        for link in tqdm(links):
            try: 
                self.driver.get(link)
                wait = WebDriverWait(self.driver, 10)
                element = wait.until(EC.presence_of_element_located((
                    By.XPATH, "//div[@class='aem-Grid aem-Grid--12 aem-Grid--default--12 ']")))
                content_text = element.text
                if content_text == '':
                    continue
                article = {"title": self.driver.title, 
                           "text": markdownify.markdownify(content_text),
                           "url": link}
                articles.append(article)
            except:
                pass
        self.save_articles(articles)
        return articles


class CCOScraper(Scraper):
    '''
    Source: CCO - Cancer Care Ontario
    '''
    def __init__(self, path, **kwargs):
        super().__init__('cco', path, pdfs=True, **kwargs)

    def _scrape_links(self):
        links = scrape_SPOR_links()
        links = [link for link in links if 'cancercareontario' in link]
        self.save_links(links)
        return links

    def scrape_articles(self, links):
        os.makedirs(self.pdf_path, exist_ok=True)
        for link in tqdm(links):
            try:
                self.driver.get(link)
                wait = WebDriverWait(self.driver, 10)
                button = wait.until(EC.element_to_be_clickable((
                    By.XPATH, '//span[@class="file"]/a[@class="jquery-once-2-processed btn purple icon pdfBTN"]')))
                self.reset_download_path()
                button.click()
            except:
                pass
        return []


class CDCScraper(Scraper):
    '''
    Source: CDC - Centers for Disease Control and Prevention (https://www.cdc.gov/)
    '''
    def __init__(self, path, **kwargs):
        super().__init__('cdc', path, pdfs=True, **kwargs)

    def _scrape_links(self):
        url = "https://stacks.cdc.gov/cbrowse?pid=cdc%3A100&parentId=cdc%3A100&maxResults=100&start=0"
        self.driver.get(url)
        links = []

        def next_page():
            try:
                continue_link = self.driver.find_element(By.PARTIAL_LINK_TEXT, 'Next')
                continue_link.click()
                time.sleep(5)
                return True

            except NoSuchElementException:
                return False
        search_expr = "//div[@class='search-result-row card']/div/div[1]/div/a"
        wait = WebDriverWait(self.driver, 5)
        hrefs = wait.until(EC.presence_of_all_elements_located((By.XPATH, search_expr)))
        links.extend([a.get_attribute("href") for a in hrefs])

        while next_page():
            try:
                wait = WebDriverWait(self.driver, 5)
                hrefs = wait.until(EC.presence_of_all_elements_located((By.XPATH, search_expr)))
                links.extend([a.get_attribute("href") for a in hrefs])
            except NoSuchElementException:
                pass

        self.save_links(links)
        return links

    def scrape_articles(self, links):
        articles = []
        for page in tqdm(links):
            self.driver.get(page)
            wait = WebDriverWait(self.driver, 5)
            try:
                button = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//button[@id='download-document-submit']")))
                button[0].click()
            except:
                pass
                #but = self.driver.find_element(By.XPATH, "//a[@id='not-link']")
                #articles.append(but.get_attribute("href"))

        self.save_articles(articles)
        return articles


class CMAScraper(Scraper):
    '''
    Source: CMA - Canadian Medical Association (https://joulecma.ca/cpg/homepage)
    '''

    def __init__(self, path, **kwargs):
        super().__init__('cma', path, **kwargs)

    def dismiss_popup(self):
        try:
            dismiss_button = self.driver.find_element(By.XPATH, "//button[@class='agree-button eu-cookie-compliance-default-button btn btn-primary']")
            dismiss_button.click()
            time.sleep(1)
        except NoSuchElementException:
            pass

    def _scrape_links(self):
        links = []
        errors = []
        ids = [488, 1038, 1000, 1001, 1003, 1005, 1007, 1009, 1028, 1012, 1014,
               1023, 1016, 1019, 1021, 1026, 1027, 1029, 1032, 1024, 1035, 1025,
               1043, 998, 999, 1030, 1002, 1004, 1006, 1008, 1010, 1037, 1013,
               1011, 1015, 1017, 1020, 1022, 1039, 1040, 1031, 1033, 1034, 1036,
               1041, 1042]
        for id in tqdm(ids): 
            try:
                self.driver.get(f"https://joulecma.ca/cpg/homepage/browse-by/category/specialties/id/{id}")
                self.dismiss_popup()  
                wait = WebDriverWait(self.driver, 5)
                multiple_page = wait.until(EC.presence_of_element_located((
                    By.XPATH, "//div[@class='pagination']")))
                if multiple_page is not None:
                    pages = self.driver.find_elements(By.XPATH, "//div[@class='pagination']/li/a")
                    for i, page in enumerate(pages):
                        wait = WebDriverWait(self.driver, 10)
                        self.driver.get(f"https://joulecma.ca/cpg/homepage/browse-by/category/specialties/id/{id}")
                        buttons = wait.until(EC.presence_of_all_elements_located((
                            By.XPATH, "//div[@class='pagination']/li/a")))
                        buttons[i].click()
                        time.sleep(3)
                        wait = WebDriverWait(self.driver, 10)
                        element = wait.until(EC.presence_of_all_elements_located((
                            By.XPATH, "//h5/a[@title='View fulltext guideline']")))
                        link = [a.get_attribute('href') for a in element]
                        links.extend(link)
                else:
                    element = wait.until(EC.presence_of_all_elements_located((
                        By.XPATH, "//h5/a[@title='View fulltext guideline']")))
                    link = [a.get_attribute('href') for a in element]
                    links.extend(link)
            except Exception as e:
                errors.append(id)
                pass
        links = list(set(links))
        self.save_links(links)
        return links

    def scrape_articles(self, links):
        '''
        Divide articles into [CMA, CMA PDFs, Choosing Wisely and other articles]
        Downloads only CMA and CMA PDFs.
        '''
        articles = []
        links = ([link.replace("\n", "") for link in links if not ("choosing" in link.lower()) and not ("choisir" in link.lower())])
        pdf_links = ([link for link in links if link.lower().endswith("pdf")])
        links = ([link for link in links if not link.lower().endswith("pdf")])
        cma_links = ([link for link in links if ("www.canada.ca" in link.lower())])

        print('Stage 2A: Downloading CMA PDFs...')
        self.driver.set_page_load_timeout(1)
        pdfs = []
        for pdf_link in tqdm(pdf_links):
            try: 
                self.driver.get(pdf_link)
                wait = WebDriverWait(self.driver, 10)
                element = wait.until(EC.presence_of_element_located((
                    By.XPATH, "//div[@id='main-content']")))
                inner_html = element.get_attribute('innerHTML')
                pdfs.append({"title": self.driver.title,
                             "text": markdownify.markdownify(inner_html),
                             "url": pdf_link})
            except:
                pass
        cma_pdfs_path = os.path.join(self.source_path, 'cma_pdfs.jsonl')
        self.save_articles(pdfs, path=cma_pdfs_path)
        articles.extend(pdfs)

        print('Stage 2B: Downloading HTML CMA articles...')
        cma = []
        for cma_link in tqdm(cma_links):
            try: 
                self.driver.get(cma_link)
                wait = WebDriverWait(self.driver, 10)
                element = wait.until(EC.presence_of_element_located((
                    By.XPATH, "//main[@property='mainContentOfPage']")))
                inner_html = element.get_attribute('innerHTML')
                cma.append({"title": self.driver.title,
                            "text": markdownify.markdownify(inner_html),
                            "url": cma_link})
            except:
                pass
        articles.extend(cma)
        self.save_articles(cma)
        return articles


class CPSScraper(Scraper):
    '''
    Source: CPS - Canadian Paediatric Society
    '''
    def __init__(self, path, **kwargs):
        super().__init__('cps', path, **kwargs)

    def _scrape_links(self):
        links = scrape_SPOR_links()
        links = [link for link in links if 'cps.ca' in link]
        self.save_links(links)
        return links

    def scrape_articles(self, links):
        articles = []
        for link in tqdm(links):
            try:
                self.driver.get(link)
                wait = WebDriverWait(self.driver, 2)
                button = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@class="cell main-body"]')))
                article = {
                    'title': self.driver.title,
                    'text': markdownify.markdownify(button.get_attribute("innerHTML")).replace("\n\n", "\n"),
                    'link': link
                }
                articles.append(article)
            except Exception as e:
                pass
        self.save_articles(articles)
        return articles


class DrugsScraper(Scraper):
    '''
    Source: Drugs.com (https://www.drugs.com/)
    '''
    def __init__(self, path, **kwargs):
        super().__init__('drugs', path, **kwargs)

    def _scrape_links(self):
        self.driver.get("https://www.drugs.com/dosage/")
        wait = WebDriverWait(self.driver, 10)
        elements = wait.until(EC.presence_of_all_elements_located((
            By.XPATH, "//ul/li/a[contains(@href, '/dosage-')]")))
        links = [el.get_attribute("href") for el in elements]
        self.save_links(links)
        return links

    def scrape_articles(self, links):
        articles = []
        for link in tqdm(links):
            try:
                self.driver.get(link)
                wait = WebDriverWait(self.driver, 10)
                elements_inner = wait.until(EC.presence_of_all_elements_located((
                    By.XPATH, "//ul/li/a[contains(@href, '/dosage/')]")))
                hrefs_art_clean = [el.get_attribute("href") for el in elements_inner]
                for href_inner in hrefs_art_clean:
                    try:
                        self.driver.get(href_inner)
                        wait = WebDriverWait(self.driver, 10)
                        div_content = wait.until(EC.presence_of_element_located((By.ID, "content")))
                        content = "<div>" + div_content.get_attribute('innerHTML') + "</div>"
                        article = {
                            "title": self.driver.title,
                            "text": markdownify.markdownify(content),
                            "url": link
                        }
                        articles.append(article)
                    except Exception as e:
                        print("Page-level: " + str(e))
            except Exception as e:
                print("Section-level: " + str(e))
        self.save_articles(articles)
        return articles


class GCScraper(Scraper):
    '''
    Source: GuidelineCentral (https://www.guidelinecentral.com/)
    '''
    def __init__(self, path, **kwargs):
        super().__init__('guidelinecentral', path, **kwargs)

    def _scrape_links(self):
        url = "https://www.guidelinecentral.com/guidelines/?t=guideline&f=%7B%22sort%22%3A%7B%22type%22%3A%22relevance%22%2C%22order%22%3A%22desc%22%7D%2C%22range%22%3A%7B%22max%22%3A0%2C%22start%22%3A0%2C%22limit%22%3A20%7D%2C%22filters%22%3A%5B%7B%22searchType%22%3A%22guideline%22%2C%22name%22%3A%22Within%2010%20Years%20(1766)%22%2C%22type%22%3A%22within10Years%22%2C%22syntax%22%3A%22docPubDate%3A%5BNOW-10YEAR%20TO%20NOW%5D%22%7D%5D%2C%22state%22%3A%22guidelines_landing_search%22%2C%22term%22%3A%22contentType%3ADOCUMENT%22%7D"
        self.driver.get(url)
        links = []
        count = 0
        while True:
            print(count)
            try:
                wait = WebDriverWait(self.driver, 30)
                hrefs = wait.until(EC.presence_of_all_elements_located((
                    By.XPATH, "//div[@class='result-meta']/h3/a")))
                link = [href.get_attribute("href") for href in hrefs]
                links.extend(link)
                next_page = wait.until(EC.presence_of_element_located((
                    By.XPATH, "//a[@class='page-link next-page-of-results']")))
                next_page.click()
                time.sleep(10)
                count += 1
            except:
                print("No more pages")
                break
        links = list(set(links))
        self.save_links(links)
        return links

    def scrape_articles(self, links):
        articles = []
        for link in tqdm(links):
            try:
                self.driver.get(link)
                wait = WebDriverWait(self.driver, 10)
                content = wait.until(EC.presence_of_element_located((By.XPATH, "//main[@class='site-main']")))
                try:
                    title = content.find_element(By.XPATH, "//div[@class='guideline-title text-center']/h1").text
                except NoSuchElementException:
                    print("No title found")
                    title = ""
                try:
                    main_content = content.find_element(By.ID, "summary-nav").text
                except NoSuchElementException:
                    print("No main content found")
                    main_content = ""
                if title and main_content:
                    articles.append({
                        "title": title,
                        "text": main_content
                    })
                time.sleep(1)
            except Exception as e:
                print(e)
                print("Skipping: " + link)
        self.save_articles(articles)
        return articles


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


class ICRCScraper(Scraper):
    '''
    Source: ICRC - International Committee of the Red Cross
    '''
    def __init__(self, path, **kwargs):
        super().__init__('icrc', path, pdfs=True, **kwargs)

    def _scrape_links(self):
        return []

    def scrape_articles(self, links):
        '''
        This one differs from other scrapers;
        it downloads a zip file of ICRC PDFs.
        '''
        assert [] == links

        def _check_file_on_disk(lf):
            expected_hash = '4369ebcfc324bd922f973d354d1f52a6aa134b335dc858311e9c69904ba3d93b'

            with open(lf, 'rb') as f:
                first_four_hundred = f.read(400)
            m = hashlib.sha256()
            m.update(first_four_hundred)

            assert m.hexdigest() == expected_hash, \
                "Downloaded file hash wrong"

        url = "https://www.idiap.ch/~kmatoba/icrc.tar.gz"

        local_dir = os.path.join(self.source_path, "pdfs")
        os.makedirs(local_dir, exist_ok=True)
        local_filename = os.path.join(local_dir, "icrc.tar.gz")

        if os.path.isfile(local_filename):
            print(f"{local_filename} exists, skipping download")
        else:
            download_file(url, local_filename)

        _check_file_on_disk(local_filename)
        with tarfile.open(local_filename, 'r') as f:
            f.extractall(path=self.pdf_path)

        files = os.listdir(self.pdf_path)
        return files


class IDSAScraper(Scraper):
    '''
    Source: IDSA - Infectious Diseases Society of America (https://www.idsociety.org/)
    '''
    def __init__(self, path, **kwargs):
        super().__init__('idsa', path, **kwargs)

    def _scrape_links(self):
        self.driver.get("https://www.idsociety.org/practice-guideline/alphabetical-guidelines/")
        wait = WebDriverWait(self.driver, 20)
        content = wait.until(EC.presence_of_element_located((
            By.CLASS_NAME, "rtln_body_content")))
        wait = WebDriverWait(content, 10)
        elements = wait.until(EC.presence_of_all_elements_located((
            By.TAG_NAME, "a")))
        links = [el.get_attribute("href") for el in elements]
        links = [link for link in links if 'idsociety' in link and 'pdf' not in link]
        self.save_links(links)
        return links

    def scrape_articles(self, links):
        articles = []
        for link in tqdm(links):
            print(link)
            try:
                self.driver.get(link)
                wait = WebDriverWait(self.driver, 10)
                content = wait.until(EC.presence_of_element_located((By.XPATH, "//article")))
                inner_html = content.get_attribute('innerHTML')
                text = markdownify.markdownify(inner_html)
                article = {
                    'link': link,
                    'text': text,
                    "url": link
                }
                articles.append(article)
            except Exception as e:
                continue

        self.save_articles(articles)
        return articles


class MAGICScraper(Scraper):
    '''
    Source: MAGIC - Making GRADE the Irresistible Choice (https://app.magicapp.org/)

    UNTESTED.
    '''

    def __init__(self, path, **kwargs):
        super().__init__('magic', path, **kwargs)

    def scroll_down(self):
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def scrape(self):
        ''' No links for MAGIC. '''
        articles = self.scrape_articles()
        print(f"Saved {len(articles)} {self.source} articles to {self.articles_path}.")
        self.driver.quit()

    def scrape_articles(self):
        articles = []
        self.driver.get("https://app.magicapp.org/#/guidelines")
        wait = WebDriverWait(self.driver, 10)
        element = wait.until(EC.presence_of_all_elements_located((
            By.XPATH, "//a[@class='contentItemDataTitle']")))

        total = len(element)
        ran = tqdm(range(total))
        for i in ran:
            try:
                self.scroll_down()
                self.driver.get(f"https://app.magicapp.org/#/guidelines")
                wait = WebDriverWait(self.driver, 10)
                element = wait.until(EC.presence_of_all_elements_located((
                    By.XPATH, "//a[@class='contentItemDataTitle']")))
                element[i].click()
                ran.set_description("Waiting for page click")
                time.sleep(7)
                try:
                    wait = WebDriverWait(self.driver, 10)
                    button_more = wait.until(EC.presence_of_all_elements_located((
                        By.XPATH, "//span[@class='sectionMoreLink']")))
                    for more in button_more:
                        more.click()
                    ran.set_description("Waiting for section click")
                    time.sleep(7)
                except:
                    pass
                try:
                    wait = WebDriverWait(self.driver, 10)
                    button_section = wait.until(EC.presence_of_all_elements_located((
                        By.XPATH, "//div[@class='ctrButtons']/button[@class='ctrBtn bkTextBtn']")))
                    for section in button_section:
                        section.click()
                    ran.set_description("Waiting for more click")
                    time.sleep(7)
                except:
                    pass
                wait = WebDriverWait(self.driver, 10)
                element = wait.until(EC.presence_of_element_located((
                    By.XPATH, "//main[@class='centerPanel noTabs']/div[@id='mainContent']//div[@class='guidelinePanel']")))
                inner_html = element.get_attribute('innerHTML')
                wait = WebDriverWait(self.driver, 10)
                strength = wait.until(EC.presence_of_all_elements_located((
                    By.XPATH, "//div[@class='recommendationWidget dojoDndItem']//div[@class='textComments']")))
                articles.append({"title": self.driver.title,
                            "strength": [markdownify.markdownify(stren.get_attribute("innerHTML")).replace("\n\n", "\n") for stren in strength],
                            "content": markdownify.markdownify(inner_html).replace("\n\n", "\n")})
            except:
                print(f"Error with {i}")
                try:
                    self.driver.find_element(By.XPATH, "//button[contains(text(),'OK')]").click()
                except:
                    pass
                pass
        self.save_articles(articles)
        return articles


class SPORScraper(Scraper):
    '''
    Source: SPOR - Strategy for Patient-Oriented Research
    '''
    def __init__(self, path, **kwargs):
        super().__init__('spor', path, pdfs=True, **kwargs)

    def _scrape_links(self):
        links = scrape_SPOR_links()
        pdf_links = [link for link in links if '.pdf' in link]
        self.save_links(pdf_links)
        return pdf_links

    def scrape_articles(self, links):
        self.driver.set_page_load_timeout(2)
        fail = 0
        for link in tqdm(links):
            try:
                self.driver.get(link)
                time.sleep(1)
            except:
                fail += 1
                pass
        if fail > 0:
            print(f"Failed to load {fail} links.")
        return []


class WHOScraper(Scraper):
    '''
    Source: WHO - World Health Organization (https://www.who.int/publications/guidelines/en/)

    Note: In November 2023, the WHO site is very slow. Play around with the sleep between iterations.
          Too long: complaints about timeouts, too fast and don't scrape the data.
    '''

    def __init__(self, path, **kwargs):
        super().__init__('who', path, pdfs=True, **kwargs)

    def _scrape_links(self):
        self.driver.get("https://www.who.int/publications/i?publishingoffices=c09761c0-ab8e-4cfa-9744-99509c4d306b")
        wait = WebDriverWait(self.driver, 10)
        pdf_urls = []
        # sleep_secs = 2.5
        sleep_secs = 1.5
        while True:
            # time.sleep(1)
            try:
                all_buttons = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//a[@class='download-url']")))
                links = [button.get_attribute("href") for button in all_buttons]
                pdf_urls.extend(links)
                print(f"Found {len(pdf_urls)} PDFs [{len(set(pdf_urls))} unique]")
                next_button_str_click = "//span[@class='k-icon k-i-arrow-60-right']"
                next_page_button_click = self.driver.find_element(By.XPATH, next_button_str_click)
                try:
                    next_page_button_click.click()
                    time.sleep(sleep_secs)
                except exceptions.ElementClickInterceptedException:
                    break
            except exceptions.StaleElementReferenceException:
                print("StaleElementReferenceException - Retrying...")
                continue
        self.save_links(pdf_urls)
        return pdf_urls

    def scrape_articles(self, links):
        os.makedirs(self.pdf_path, exist_ok=True)
        articles = []
        for i, pdf_url in tqdm(enumerate(links)):
            response = requests.get(pdf_url)
            if response.status_code != 200:
                print(f"Failed to get {pdf_url}")
                continue
            filename = os.path.join(self.pdf_path, f"{i}.pdf")
            with open(filename, "wb") as f:
                f.write(response.content)
        return articles


SCRAPERS = {
    'aafp': AAFPScraper,
    'cco': CCOScraper,
    'cdc': CDCScraper,
    'cma': CMAScraper,
    'cps': CPSScraper,
    'drugs': DrugsScraper,
    'guidelinecentral': GCScraper,
    'icrc': ICRCScraper,
    'idsa': IDSAScraper,
    'magic': MAGICScraper,
    'spor': SPORScraper,
    'who': WHOScraper
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        # default=os.path.join(os.getcwd(), 'raw'),
                        default=os.getcwd(),
                        help="Path to download scraped guidelines to.")
    parser.add_argument("--sources",
                        nargs="+",
                        default=list(SCRAPERS.keys()),
                        help="List of sources to scrape, formatted as: --sources source1 source2 ... Default: all sources.")
    parser.add_argument("--chrome_binary_location",
                        type=str,
                        default=None,
                        help="Path to Chrome binary.")
    parser.add_argument("--chrome_driver_location",
                        type=str,
                        default=None,
                        help="Path to Chrome driver.")
    catch_scrape_exceptions = False
    # catch_scrape_exceptions = True
    args = parser.parse_args()

    os.makedirs(args.path, exist_ok=True)
    print(f"Downloading guidelines to {args.path}.")

    scrapers_dict = {k: v for k, v in SCRAPERS.items() if k in args.sources} if args.sources else SCRAPERS
    print(f"Scraping {len(scrapers_dict)} sources: {list(scrapers_dict.keys())}")

    for i, (source, scraper_class) in enumerate(scrapers_dict.items()):
        print('\n' + '-' * 50 + f"Scraping {source} [{i+1}/{len(scrapers_dict)}]...")
        try:
            scrap_params = dict(path=args.path)
            if args.chrome_binary_location:
                scrap_params.update(chrome_binary_location=args.chrome_binary_location)
            if args.chrome_driver_location:
                scrap_params.update(chrome_driver_location=args.chrome_driver_location)
            scraper = scraper_class(**scrap_params)
        except Exception as e:
            print(f"Error while initializing {source} scraper: {e}")
            continue
        if catch_scrape_exceptions:
            try:
                scraper.scrape()
            except Exception as e:
                print(f"Error while scraping {source}: {e}")
                continue
        else:
            scraper.scrape()
