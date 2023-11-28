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

const TOC_URL='https://www.nice.org.uk/guidance/published?ps=2500&ndt=Guidance';
const OUTPUT_PATH: string = 'nice_guidelines.json'
const VERBOSE = true;
const HEADLESS = true;
const TIMEOUT = 200;

const TOC_RESULT_ITEM_SELECTOR:string = "td a";                             // Top-level TOC items
const PAGE_SECTION_SELECTOR:string = "ul.nav-list li a";                    // Links to page sections
const CONTENT_SELECTOR:string = "div.chapter h3.title, div.chapter p, div.chapter h4.title"; // Text content of the page
const OVERVIEW_SELECTOR:string = "div.content p.lead"            // Overview

/* ==================== CLASSES ==================== */
interface Dictionary<T> {
  [Key: string]: T;
}
class Guideline {
  readonly name:NonNullable<string>;
  readonly url:NonNullable<string>;
  readonly overview:NonNullable<string>;
  content:NonNullable<Dictionary<string>>;

  constructor(name:string,url:string,overview:string,content:Dictionary<string>){
    this.name=name;
    this.url=url;
    this.overview=overview;
    this.content=content;
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

  async get_links (selector: string) {
    return await this.page.$$eval(selector, elements=>elements.map(a=>[a.textContent,(a as HTMLAnchorElement).href]));
  }
  async format_title(title:string|null):Promise<string|null>{
    if (title==null){return null;}
    const pattern = /\b(\d+(?:\.\d+){1,3})\b/g;
    return title.replace(pattern, '#');
  }
  async check_sel (selector: string) {
    return await this.page.$eval(selector, () => true).catch(() => false);
  }
  async get_by_sel(selector:string,cb?: (x:any)=>any) {
    if (cb!=null){
      return await this.page.$$eval(selector, (links) => links );
    }else{
      console.log("Trying to print objects");
      return await this.page.$$eval(selector, cb!);
    }
  }
  async save_guideline(guideline:Guideline, path:string){
    await fs.appendFileSync(path, JSON.stringify(guideline, null, 0)+'\n');
  }

  /* ==================== GUIDELINE EXTRACTOR ==================== */

  async getSectionContent(section_name:string, section_url:string){
    await this.page.goto(section_url);
    let section_content = '';
    let content_selector = CONTENT_SELECTOR;
    if (section_name == 'Context') {
      content_selector = 'div.chapter p'
    }
    const elements = (await this.page.$$(content_selector));
    for (let el of elements) {
      const tag = await this.page.evaluate(el => el.tagName, el);
      let text = await this.page.evaluate(el => el.textContent, el);
      let formatted_text = text?.trim().replace(/^\d+(\.\d+)*\s*/, '')
      if (tag == 'H3'){
        section_content += '\n\n# ' + formatted_text;
      }
      else if (tag == 'H4'){
        section_content += '\n\n## ' + formatted_text;
      }
      else if (tag == 'P'){
        section_content += '\n' + formatted_text;
      }
    }
    section_content = section_content.replace(/\[\d+\]/g, '').replace(/^\n+/, '');
    return section_content;
  }

  async getGuideline(name:string,url:string){
    await this.page.goto(url);
    await this.page.waitForTimeout(TIMEOUT);

    let content:Dictionary<string> = {};
    let overview = (await this.page.$$eval(OVERVIEW_SELECTOR, elements => elements.map(a => a.textContent))).join('');
    overview = overview.trim();

    let section_links = (await this.get_links(PAGE_SECTION_SELECTOR)).slice(1); // Remove overview from links
    for (let [sec_index, el] of section_links.entries()){
      const section_name = el[0]!;
      const section_url = el[1]!;
      if (VERBOSE){console.log(`\tSection ${sec_index} of ${section_links.length}:\n\tName: ${section_name}\n\tURL: ${section_url}\n`);}
      const section_content = await this.getSectionContent(section_name, section_url);
      content[section_name.replace(/^\s*\d+\s*/, '')] = section_content;
      //if (VERBOSE){console.log(`\tContent:\n${section_content}\n`);}
    }
    //if (VERBOSE){console.log(`\tOverview: ${overview}\n`);}
    let guideline = new Guideline(name, url, overview, content);
    await this.page.goto(TOC_URL);
    return guideline;
  }

  /* ==================== SCRAPING FUNCTION ==================== */

  async scrape(){
    var all_good = true;
    try {
      if (await this.check_sel(TOC_RESULT_ITEM_SELECTOR)){
        let toc_links = await this.get_links(TOC_RESULT_ITEM_SELECTOR);
        for (let [toc_index, el] of toc_links.entries()) {
          const page_name = el[0]!;
          const page_url = el[1]!;
          console.log(`\nGuideline ${toc_index} of ${toc_links.length}:\nName: ${page_name}\nURL: ${page_url}\n`);
          const guideline = await this.getGuideline(page_name, page_url);
          this.save_guideline(guideline, OUTPUT_PATH);
          this.page.waitForTimeout(TIMEOUT);
        }
      }
    }
    catch(e) {
      console.log(e);
      all_good=false;
    }
  return all_good;
  }
}

/* ==================== MAIN ==================== */

async function run(){
  const run = await PuppeteerRun.setup(HEADLESS);
  let all_good = await run.scrape();
  if (all_good){
    try {
      await run.browser.close();
    }
    catch(e) {
      console.log("Error while closing",e);
    }
  }
}

run().then(()=>console.log("Done!")).catch(x=>console.error(x));
