

/* ==================== IMPORTS ==================== */

import type {Page, Browser} from 'puppeteer';
import { createCursor } from "ghost-cursor"

const puppeteer = require('puppeteer-extra')

// Stealth plugin (all tricks to hide puppeteer usage)
const StealthPlugin = require('puppeteer-extra-plugin-stealth')
puppeteer.use(StealthPlugin())

// Adblocker plugin to block all ads and trackers (saves bandwidth)
const AdblockerPlugin = require('puppeteer-extra-plugin-adblocker')
puppeteer.use(AdblockerPlugin({ blockTrackers: true }))
const fs = require("fs");

/* ==================== GLOBAL VARIABLES ==================== */

const TOC_URL='https://www.rch.org.au/clinicalguide/about_rch_cpgs/welcome_to_the_clinical_practice_guidelines/';
const OUTPUT_PATH: string = 'royalAUS_guidelines.jsonl'
const HEADLESS = true;

const TOC_SELECTOR:string = "#tabnav-letter-blocks li a";    // Top-level TOC items
const CONTENT_SELECTOR:string = "div.widgetBody :not(table) p, div.widgetBody :not(table) li, div.widgetBody :not(table) h2, div.widgetBody :not(table) h3"

var scraped:string[] = [];

/* ==================== CLASSES ==================== */

class Guideline {
  readonly name:NonNullable<string>;
  readonly url:NonNullable<string>;
  content:NonNullable<string>;

  constructor(name:string,url:string,content:string){
    this.name=name;
    this.url=url;
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

  static async setup(headless_b:boolean,timeout:number):Promise<PuppeteerRun>{
    const headless= headless_b ? "new":headless_b;
    const browser=await puppeteer.launch({ headless: headless });
    const page = await browser.newPage();
    page.setViewport({ width: 800, height: 600 });
    const cursor = createCursor(page);
    await page.goto(TOC_URL);
    await page.waitForTimeout(timeout);
    return new PuppeteerRun(page,browser,cursor);
  }

  async get_links (selector: string) {
    return await this.page.$$eval(selector, (elements:any)=>elements.map((a:any)=>[a.textContent,(a as HTMLAnchorElement).href]));
  }

  async check_sel (selector: string) {
    return await this.page.$eval(selector, () => true).catch(() => false);
  }

  async save_guideline(guideline:Guideline, path:string){
    await fs.appendFileSync(path, JSON.stringify(guideline, null, 0)+'\n');
  }

  async getParentPath(el:any){
    return await this.page.evaluate((el:any) => {
      let path = '';
      let parent = el.parentElement;
      while (parent != null){
        path += parent.tagName+'.'+parent.className+' ';
        parent = parent.parentElement;
      }
      return path;
    }, el);
  }

  /* ==================== GUIDELINE EXTRACTOR ==================== */

  async getGuideline(name:string,url:string){

    await this.page.goto(url);
    await this.page.waitForTimeout(200);

    let content = '';
    let sections = await this.page.$$(CONTENT_SELECTOR);

    // If there is a section with id 'key-points', start from there
    const key_selector = 'div.widgetBody #key-points'
    let skip = await this.check_sel(key_selector);

    for (let el of sections) {
      const tag = await this.page.evaluate((el:any) => el.tagName, el);
      let text = await this.page.evaluate((el:any) => el.innerText.trim(), el);
      const id = await this.page.evaluate((el:any) => el.id, el);

      // Skip until key-points if present
      if (skip && id != 'key-points'){continue;}
      skip = false;

      // Skip if sub-contained in a table
      const parentPath = await this.getParentPath(el);
      if (parentPath.match(/TABLE\.table/)){continue;}

      // Replace double spaces (not newlines) by single space if present
      text = text.replace(/ {2}/g, " ");

      // Remove asterisks
      text = text.replace(/\*/g, '');

      // Skip empty text
      if (text == ''){continue;}

      // Sublists are matched twice, skip them
      if (parentPath.match(/LI.*LI/)){
        continue;
      }

      // If we reached Last updated or References or Additional notes, stop
      if (text.match(/^Last updated/)){break;}
      if (tag == 'H2' && text.match(/References/)){break;}
      if (tag == 'H2' && text.match(/Additional/)){break;}
      if (tag == 'H2' && text.match(/Parent information/)){break;}

      // Formatting text depending on tag
      if (tag == 'H2'){
        content += '\n\n# ' + text;
      }
      else if (tag == 'H3'){
        content += '\n\n## ' + text;
      }
      else if (tag == 'LI'){
        if (text.match(/\n/)){
          text = text.replace(/\n/g, '\n\t- ');
        }
        content += '\n- '+text;
      }
      else if (tag == 'P'){
        content += '\n' + text;
      }
    }
    content = content.trim();
    //console.log('\n\n\nContent:\n',content,'\n\n\n')
    let guideline = new Guideline(name, url, content);
    await this.page.goto(TOC_URL);
    return guideline;
  }

  /* ==================== SCRAPING FUNCTION ==================== */

  async scrape(){
    await this.page.waitForTimeout(500);
    let links = await this.get_links(TOC_SELECTOR);
    let num_links = links.length;

    for (let [idx, el] of links.entries()) {
      const name = el[0]!.split(' (see')[0];
      const url = el[1]!;
      if (name in scraped){continue;}

      console.log(`\nGuideline ${idx} of ${num_links}:\nName: ${name}\nURL: ${url}\n`);
      for (let i=0; i<3; i++){
        try{
          let guideline = await this.getGuideline(name, url);
          await this.save_guideline(guideline, OUTPUT_PATH);
          break;
        }
        catch(e){
          console.log(`Error: ${e}\nTrying again...`);
        }
      }
    }
  }
}

/* ==================== MAIN ==================== */

async function run(){
  
  const run = await PuppeteerRun.setup(HEADLESS, 200);
  await run.page.click(`a[href="#tab-All"]`);
  await run.scrape();
}

run().then(()=>console.log("Done!")).catch(x=>console.error(x));
