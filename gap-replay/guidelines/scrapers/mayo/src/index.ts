
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

const TOC_URL='https://www.mayoclinic.org/diseases-conditions/index';
const OUTPUT_PATH: string = 'mayo_guidelines.json'
const VERBOSE = true;
const HEADLESS = true;
const TIMEOUT = 200;

const TOC_RESULT_ITEM_SELECTOR:string = "div.cmp-back-to-top-container div.cmp-result-name div.cmp-link a";    // Top-level TOC items
const PAGE_SECTION_SELECTOR:string = "#access-nav a, div.cmp-tab-navigation-tabs a";        // Links to page sections
const CONTENT_SUBSELECTORS = [`li`, `h2`, `p`]; // Subselectors for content

/* ==================== CLASSES ==================== */
interface Dictionary<T> {
  [Key: string]: T;
}
class Guideline {
  readonly name:NonNullable<string>;
  readonly url:NonNullable<string>;
  content:NonNullable<Dictionary<string>>;

  constructor(name:string,url:string,content:Dictionary<string>){
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

  async save_guideline(guideline:Guideline, path:string){
    await fs.appendFileSync(path, JSON.stringify(guideline, null, 0)+'\n');
  }


  /* ==================== GUIDELINE EXTRACTOR ==================== */

  async getSectionContent(section_url:string){
    await this.page.goto(section_url);
    this.page.waitForTimeout(TIMEOUT);

    // Select content on page (try two different formats)
    let base_sel = '';
    if (await this.check_sel('div.content')){
      base_sel = 'div.content';
    }
    else {
      base_sel = 'div.aem-GridColumn section';
    }
    let selectors = [];
    for (let subsel of CONTENT_SUBSELECTORS){
      selectors.push(base_sel+' '+subsel);
    }
    if (selectors.length == 0){
      console.log("Couldn't find content selector for page")
      return '';
    }
    let content_selector = selectors.join(', ');
    //console.log('\tContent selector: ', content_selector)
    const sections = await this.page.$$(content_selector);
    
    // Construct section content
    let section_content = '';
    for (let [index, el] of sections.entries()) {
      const tag = await this.page.evaluate(el => el.tagName, el);
      const text = await this.page.evaluate(el => el.textContent?.trim(), el);

      // Filtering out ads and other stuff
      if (text == ''){continue;}
      const parentPath = await this.page.evaluate(el => {
        let path = '';
        let parent = el.parentElement;
        while (parent != null){
          path += parent.tagName+'.'+parent.className+' ';
          parent = parent.parentElement;
        }
        return path;
      }, el);
      if (parentPath.match(/DIV\.content.*DIV\.content/)){continue;}
      const toAvoid = ['references', 'acces-list-container', 'tableofcontents']
      if (toAvoid.some(x=>parentPath.match(x))){continue;}
      
      // Formatting text depending on tag
      //console.log('Tag: ',tag,'\nParent path: ', parentPath, '\nText: ', text,'\n')
      if (tag == 'H2'){
        section_content += '\n ' + text;
      }
      else if (tag == 'LI'){
        section_content += '\n- ' + text;
      }
      else if (tag == 'P'){
        section_content += '\n' + text;
      }
    }
    return section_content;
  }

  async getGuideline(name:string,url:string){
    let content:Dictionary<string> = {};
    let section_links = (await this.get_links(PAGE_SECTION_SELECTOR));
    
    for (let el of section_links){
      let section_name = el[0]!.replace(/&/g, '& ');
      const section_url = el[1]!;

      // Only keep Symptoms and Diagnosis sections
      if (!section_name.match(/Symptoms|Diagnosis/)){continue;}
      const section_content = await this.getSectionContent(section_url);
      if (VERBOSE){console.log(`\tSection: ${section_name}\n\tURL: ${section_url}\n`);}
      //if (VERBOSE){console.log(`\tContent:\n${section_content}\n`);}
      content[section_name] = section_content.trim();
    }
    let guideline = new Guideline(name, url, content);
    await this.page.goto(TOC_URL);
    return guideline;
  }

  /* ==================== SCRAPING FUNCTION ==================== */

  async scrape(){
    var all_good = true;
    try {
      // Iterate over each letter 
      for (let letter='A'; letter<='Z'; letter=String.fromCharCode(letter.charCodeAt(0)+1)){
        const letter_URL = TOC_URL+'?letter='+letter;
        await this.page.goto(letter_URL);
        await this.page.waitForTimeout(TIMEOUT);
        console.log(`\nLetter: ${letter}\nURL: ${letter_URL}\n`)

        // For each letter, get all links to guidelines starting with that letter
        if (await this.check_sel(TOC_RESULT_ITEM_SELECTOR)){
          let toc_links = await this.get_links(TOC_RESULT_ITEM_SELECTOR);
          for (let [toc_index, el] of toc_links.entries()) {
            const page_name = el[0]!;
            const page_url = el[1]!;
            console.log(`\nGuideline ${toc_index} of ${toc_links.length}:\nName: ${page_name}\nURL: ${page_url}\n`);

            // For each guideline, try to scrape it 3 times before giving up
            for (let i=0; i<3; i++){
              try {
                await this.page.goto(page_url);
                const guideline = await this.getGuideline(page_name, page_url);
                this.save_guideline(guideline, OUTPUT_PATH);
                console.log('\tSaved guideline!\n')
                break;
              }
              catch(e) {
                console.log('\tError while scraping guideline, retrying...\n', e)
                this.page.waitForTimeout(10000);
              }
            }
          }
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
