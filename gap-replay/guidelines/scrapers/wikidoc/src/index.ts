/* ==================== IMPORTS ==================== */

import type {Page, Browser} from 'puppeteer';
import { createCursor } from "ghost-cursor"

const puppeteer = require('puppeteer-extra')

// Stealth plugin (all tricks to hide puppeteer usage)
const StealthPlugin = require('puppeteer-extra-plugin-stealth')
puppeteer.use(StealthPlugin())

// Adblocker plugin to block all ads and trackers (saves bandwidth)
const AdblockerPlugin = require('puppeteer-extra-plugin-adblocker')
puppeteer.use(AdblockerPlugin({blockTrackers: true}))
const fs = require("fs");

/* ==================== GLOBAL VARIABLES ==================== */

const TOC_URL='https://www.wikidoc.org/index.php/Special:AllPages';
const VERBOSE = true;
const HEADLESS = true;
const TIMEOUT = 50;

const TITLE_SELECTOR:string = "#firstHeading"; // Title of the page
const NEXT_PAGE_SELECTOR:string = "div.mw-allpages-nav a"
const TOC_SELECTOR:string = "div.mw-allpages-body li a";                   
const CONTENT_SELECTOR:string = "div.mw-parser-output h1, div.mw-parser-output h2, div.mw-parser-output h3, div.mw-parser-output h4, div.mw-parser-output p, div.mw-parser-output li"; // Text content of the page

const OUTPUT_PATH: string = 'wikidoc_articles.jsonl'
const TOC_PATH = 'toc_pages.txt'
const SCRAPED_PATH = 'scraped_articles.txt'

var SCRAPED_URLs:string[] = [];

/* ==================== CLASSES ==================== */
class Article {
  readonly name:NonNullable<string>;
  readonly url:NonNullable<string>;
  text:NonNullable<string>;

  constructor(name:string,url:string,text:string){
    this.name=name;
    this.url=url;
    this.text=text;
  }
}

/* ==================== SCRAPER ==================== */

class PuppeteerRun{
  page: Page;
  browser: Browser;
  cursor:any;

  constructor(page: Page, browser: Browser,cursor:any){
    this.page=page;
    this.browser=browser;
    this.cursor=cursor;
  }

  /* ==================== HELPER FUNCTIONS ==================== */

  static async setup(headless_b:boolean):Promise<PuppeteerRun>{
    const headless= headless_b ? "new":headless_b;
    const browser=await puppeteer.launch({ headless: headless });
    const page = await browser.newPage();
    page.setViewport({ width: 800, height: 600 });
    const cursor = createCursor(page);
    await page.goto(TOC_URL);
    await page.waitForTimeout(TIMEOUT);
    if (VERBOSE){console.log("Reached table of contents at URL: ",TOC_URL,"\n");}
    return new PuppeteerRun(page,browser,cursor);
  }

  async check_sel (selector: string) {
    return await this.page.$eval(selector, () => true).catch(() => false);
  }
  async save_article(article:Article, path:string){
    await fs.appendFileSync(path, JSON.stringify(article, null, 0)+'\n');
  }
  async get_links (selector: string) {
    return await this.page.$$eval(selector, elements=>elements.map(a=>[a.textContent,(a as HTMLAnchorElement).href]));
  }

  async getAllPages (urls:string[]) : Promise<string[]> {
    // Collect all table of content page URLs
    if (await this.check_sel(NEXT_PAGE_SELECTOR)){
      let next_page_links = await this.get_links(NEXT_PAGE_SELECTOR);
      next_page_links = next_page_links.filter(x=>x[0]!.startsWith("Next"));
      if (next_page_links.length > 0 && next_page_links[0][1] != null){
        let next_url = next_page_links[0][1];
        urls.push(next_url);
        await this.page.goto(next_url);
        await this.page.waitForTimeout(100);
        return await this.getAllPages(urls);
      }
    }
    return urls;
  }

  /* ==================== ARTICLE EXTRACTOR ==================== */

  async formatContent(){
    let elements = (await this.page.$$(CONTENT_SELECTOR));
    let content = '';

    // If there is a TOC, skip all elements until first h2 after 'Contents'
    for (let el of elements) {
      let tag = (await this.page.evaluate(el => el.tagName, el));
      let text = (await this.page.evaluate(el => el.textContent, el))?.trim();

      // Skip children of table of contents or table
      let parent_path = await this.page.evaluate(el => {
        let path = '';
        let parent = el.parentElement;
        while (parent != null){
          path += parent.tagName+'.'+parent.id + ' '; 
          parent = parent.parentElement;
        }
        return path;
      }, el);
      if (parent_path.match(/DIV.toc/) || parent_path.match(/TABLE/)){
        continue;
      }
      //console.log('Tag: '+tag+'\nPath: '+parent_path+'\nText: '+text+'\n')
      if (tag == 'H1' || tag == 'H2' || tag == 'H3' || tag == 'H4') {
        if (text === 'References' || text?.startsWith('See also')){
          break;
        }
      }
      if (tag == 'H1'){
        content += '\n\n# ' + text;
      }
      else if (tag == 'H2'){
        content += '\n\n# ' + text;
      }
      else if (tag == 'H3'){
        content += '\n\n## ' + text;
      }
      else if (tag == 'H4'){
        content += '\n\n### ' + text;
      }
      else if (tag == 'P'){
        if (parent_path.match(/LI/)){
          continue;
        }
        text = text?.replace(/^\|.*$/gm, '');
        text = text?.replace(/^\{\{.*$/gm, '');
        text = text?.replace(/^\}\}.*$/gm, '');
        text = text?.replace(/^\s*[\r\n]/gm, '');
        if (text != null && text != ''){
          content += '\n' + text;
        }
      }
      else if (tag == 'LI'){
        content += '\n- ' + text;
      }
    }
    content = content.trim();
    //console.log('\n\n=========================\n\n'+content+'\n\n=========================\n\n')
    return content;
  }

  async scrapeArticle(url:string){
    await this.page.goto(url);
    await this.page.waitForTimeout(TIMEOUT);

    // Check for error 
    let title = await this.page.$eval(TITLE_SELECTOR, element=>element.textContent?.trim());
    if (title!.startsWith("Error")) {
      console.log(`\tSKIPPED.`);
      return;
    }
    // Extract article content
    let content = await this.formatContent();
    if (title == null){
      title = '';
    }
    if (title != null && title != ''){
      content = title + '\n\n' + content;
    }
    let article:Article = new Article(title,url,content);
    await this.save_article(article, OUTPUT_PATH);
  }

  /* ==================== SCRAPING FUNCTION ==================== */

  async scrape(toc_url:string){
    try {
      await this.page.goto(toc_url);
      await this.page.waitForTimeout(TIMEOUT);

      let article_links = await this.get_links(TOC_SELECTOR);
      for (let i = 0; i < article_links.length; i++) {
        const article_link = article_links[i];
        const article_name = article_link[0];
        const article_url = article_link[1];
        if (article_name == null || article_url! in SCRAPED_URLs) {continue;}
        console.log(`\n\tArticle (${i+1} / ${article_links.length}):\n\tName: ${article_name}\n\tURL: ${article_url}`)
        await this.scrapeArticle(article_url!);
        SCRAPED_URLs.push(article_url!);
        await fs.appendFileSync(SCRAPED_PATH, article_url+'\n');
      }
    }
    catch (e) {
      console.error(e);
    }
  }
}

/* ==================== MAIN ==================== */

async function run(){
  const run = await PuppeteerRun.setup(HEADLESS);

  // Get TOC pages
  var toc_urls:string[] = [];
  if (!fs.existsSync(TOC_PATH)) {
    console.log('Scraping all TOC pages...')
    toc_urls = await run.getAllPages([]);
    console.log('\nSaving TOC URLs to file...')
    for (let i = 0; i < toc_urls.length; i++) {
      const toc_url = toc_urls[i];
      await fs.appendFileSync(TOC_PATH, toc_url+'\n');
    }
  }
  else {
    toc_urls = fs.readFileSync(TOC_PATH, 'utf8').split('\n');
    console.log(`Loaded ${toc_urls.length} TOC pages`)
  }

  // Check for already scraped pages and articles
  if (fs.existsSync(SCRAPED_PATH) && fs.existsSync(OUTPUT_PATH)) {
    SCRAPED_URLs = fs.readFileSync(SCRAPED_PATH, 'utf8').split('\n');
    console.log(`Already scraped ${SCRAPED_URLs.length} articles`)
  }

  // Scrape all remaining articles
  console.log(`Scraping ${toc_urls.length} TOC pages.`)
  for (let i = 0; i < toc_urls.length; i++) {
    const toc_url = toc_urls[i];
    console.log(`Page (${i+1} / ${toc_urls.length}):`)
    await run.scrape(toc_url);
  }
}

run().then(()=>console.log("Done!")).catch(x=>console.error(x));
