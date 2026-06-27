import "dotenv/config";
import express from "express";
import cors from "cors";
import fs from "fs";
import multer from "multer";
import { z } from "zod";

// ✅ pdfjs-dist — ESM native
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

import { ChatGroq } from "@langchain/groq";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

/* ─────────────────────────────────────────────
   EXPRESS SETUP
───────────────────────────────────────────── */
const app = express();
app.use(express.json());
app.use(cors());
const upload = multer({ dest: "uploads/" });

/* ─────────────────────────────────────────────
   IN-MEMORY CHAT HISTORY
───────────────────────────────────────────── */
const chatHistories = {};

/* ─────────────────────────────────────────────
   PDF TEXT EXTRACTION
───────────────────────────────────────────── */
async function extractTextFromPDF(filePath) {
  const data = new Uint8Array(fs.readFileSync(filePath));
  const pdf = await pdfjsLib.getDocument({ data, verbosity: 0 }).promise;
  console.log(`📖 PDF pages: ${pdf.numPages}`);
  let fullText = "";
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((item) => item.str)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    fullText += `\n\n[Page ${i}]\n${pageText}`;
  }
  if (!fullText.trim()) throw new Error("PDF appears empty or unreadable.");
  return fullText;
}

/* ─────────────────────────────────────────────
   TF-IDF VECTOR STORE
───────────────────────────────────────────── */
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
}

function tfidfScore(queryTokens, docTokens) {
  const docSet = new Set(docTokens);
  let hits = 0;
  for (const t of queryTokens) if (docSet.has(t)) hits++;
  return hits / (Math.sqrt(queryTokens.length) * Math.sqrt(docTokens.length) + 1);
}

class TFIDFVectorStore {
  constructor() {
    this.docs = [];
    this.tokenized = [];
  }
  static build(docs) {
    const s = new TFIDFVectorStore();
    s.docs = docs;
    s.tokenized = docs.map((d) => tokenize(d.pageContent));
    return s;
  }
  retrieve(query, k = 10) {
    const qTokens = tokenize(query);
    return this.docs
      .map((doc, i) => ({ doc, score: tfidfScore(qTokens, this.tokenized[i]) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((x) => x.doc);
  }
}

/* ─────────────────────────────────────────────
   RAG STATE
───────────────────────────────────────────── */
const rag = { store: null, ready: false, filename: null };

async function processPDF(filePath, name) {
  console.log(`\n📄 Processing: ${name}`);
  const text = await extractTextFromPDF(filePath);
  console.log(`📝 Extracted ${text.length} chars`);
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1500,
    chunkOverlap: 300,
  });
  const chunks = await splitter.createDocuments([text]);
  console.log(`✂️  ${chunks.length} chunks`);
  rag.store = TFIDFVectorStore.build(chunks);
  rag.ready = true;
  rag.filename = name;
  console.log(`✅ RAG ready: "${name}"\n`);
}

/* ─────────────────────────────────────────────
   COMPANY NAME → TICKER MAP
───────────────────────────────────────────── */
const COMPANY_TO_TICKER = {
  // Tech
  "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT",
  "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN",
  "meta": "META", "facebook": "META", "netflix": "NFLX",
  "nvidia": "NVDA", "intel": "INTC", "amd": "AMD",
  "applied materials": "AMAT", "qualcomm": "QCOM",
  "broadcom": "AVGO", "taiwan semiconductor": "TSM", "tsmc": "TSM",
  "samsung": "SSNLF", "sony": "SONY", "ibm": "IBM",
  "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE",
  "zoom": "ZM", "uber": "UBER", "lyft": "LYFT",
  "airbnb": "ABNB", "spotify": "SPOT",
  "snap": "SNAP", "pinterest": "PINS", "reddit": "RDDT",
  "amat": "AMAT",
  // Finance
  "jpmorgan": "JPM", "jp morgan": "JPM", "goldman sachs": "GS",
  "morgan stanley": "MS", "bank of america": "BAC",
  "wells fargo": "WFC", "citigroup": "C", "visa": "V",
  "mastercard": "MA", "paypal": "PYPL", "coinbase": "COIN",
  "blackrock": "BLK", "berkshire": "BRK-B",
  // EV / Auto
  "rivian": "RIVN", "lucid": "LCID", "ford": "F",
  "general motors": "GM", "toyota": "TM", "ferrari": "RACE",
  "volkswagen": "VWAGY", "bmw": "BMWYY",
  // Retail / Consumer
  "walmart": "WMT", "target": "TGT", "costco": "COST",
  "nike": "NKE", "starbucks": "SBUX", "mcdonalds": "MCD",
  "coca cola": "KO", "cocacola": "KO", "pepsi": "PEP",
  "pepsico": "PEP", "disney": "DIS",
  // Healthcare / Pharma
  "pfizer": "PFE", "moderna": "MRNA", "johnson": "JNJ",
  "abbvie": "ABBV", "eli lilly": "LLY", "lilly": "LLY",
  "unitedhealth": "UNH", "cvs": "CVS", "merck": "MRK",
  // Energy
  "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX",
  "bp": "BP", "shell": "SHEL", "aramco": "2222.SR",
  "conocophillips": "COP",
  // Semiconductors
  "micron": "MU", "arm": "ARM", "asml": "ASML",
  "lam research": "LRCX", "kla": "KLAC", "marvell": "MRVL",
  "skyworks": "SWKS", "analog devices": "ADI", "texas instruments": "TXN",
  // Cloud / SaaS
  "palantir": "PLTR", "snowflake": "SNOW", "shopify": "SHOP",
  "square": "SQ", "block": "SQ", "twilio": "TWLO",
  "datadog": "DDOG", "cloudflare": "NET", "crowdstrike": "CRWD",
  "palo alto": "PANW", "fortinet": "FTNT", "okta": "OKTA",
  "hubspot": "HUBS", "workday": "WDAY", "servicenow": "NOW",
  "mongodb": "MDB", "elastic": "ESTC", "confluent": "CFLT",
  // Others
  "spacex": "SPACEX", "astrazeneca": "AZN", "novartis": "NVS",
  "roche": "RHHBY", "siemens": "SIEGY", "sap": "SAP",
  "alibaba": "BABA", "baidu": "BIDU", "jd.com": "JD",
  "pinduoduo": "PDD", "tencent": "TCEHY", "nio": "NIO",
  "xpeng": "XPEV", "li auto": "LI",
};

/* ─────────────────────────────────────────────
   SKIP WORDS
───────────────────────────────────────────── */
const SKIP_WORDS = new Set([
  "what", "show", "give", "tell", "get", "find", "the", "for",
  "can", "you", "how", "much", "live", "now", "today", "its",
  "stock", "price", "share", "ticker", "current", "latest", "is",
  "are", "was", "has", "had", "and", "but", "with", "from",
  "news", "about", "who", "when", "where", "why", "this",
  "stcok", "prie", "stoock", "priice",
]);

/* ─────────────────────────────────────────────
   PATTERNS
───────────────────────────────────────────── */
const MATH_PATTERN      = /(\d+(?:\.\d+)?)\s*([\+\-\*\/x])\s*(\d+(?:\.\d+)?)/i;
const STOCK_TRIGGER     = /\b(stock|price|share|ticker)\b/i;
const DATE_TIME_PATTERN = /\b(today|current date|current time|what.*date|what.*time|what.*day|date today|time now|todays date)\b/i;

/* ─────────────────────────────────────────────
   NEWS KEYWORDS — always force web_search
───────────────────────────────────────────── */
const NEWS_KEYWORDS = [
  "news", "about", "latest", "update", "updates", "happening",
  "who is", "what is", "when is", "where is", "how is",
  "tell me", "explain", "weather", "score", "match", "game",
  "movie", "show", "song", "actor", "player", "team",
  "event", "tournament", "launch", "release", "announced",
  "rumor", "leak", "trailer", "review", "result", "winner",
  "how many", "mp", "minister", "party", "election", "government",
  "president", "prime minister", "country", "war", "policy",
];

/* ─────────────────────────────────────────────
   PDF KEYWORDS
───────────────────────────────────────────── */
const PDF_KEYWORDS = [
  "pdf", "document", "uploaded", "file", "report",
  "summarize", "summary", "according to", "in the doc",
  "from the file", "in the file", "page", "section",
  "oop", "oops", "explain the", "what does the",
];

/* ─────────────────────────────────────────────
   SMART TICKER EXTRACTOR
───────────────────────────────────────────── */
function extractTicker(query) {
  const lower = query.toLowerCase();

  const sortedNames = Object.keys(COMPANY_TO_TICKER)
    .sort((a, b) => b.length - a.length);
  for (const name of sortedNames) {
    if (lower.includes(name)) {
      console.log(`🏷️  Name match: "${name}" → ${COMPANY_TO_TICKER[name]}`);
      return COMPANY_TO_TICKER[name];
    }
  }

  const tickerMatches = query.toUpperCase().match(/\b([A-Z]{2,5})\b/g);
  if (tickerMatches) {
    for (const candidate of tickerMatches) {
      if (!SKIP_WORDS.has(candidate.toLowerCase())) {
        console.log(`🏷️  Symbol match: ${candidate}`);
        return candidate;
      }
    }
  }

  return null;
}

/* ─────────────────────────────────────────────
   INTENT DETECTOR
───────────────────────────────────────────── */
function detectIntent(message) {
  const lower = message.toLowerCase();

  if (rag.ready && PDF_KEYWORDS.some((kw) => lower.includes(kw))) return "rag";
  if (MATH_PATTERN.test(message)) return "math";
  if (DATE_TIME_PATTERN.test(lower)) return "datetime";

  const isNewsQuery = NEWS_KEYWORDS.some((kw) => lower.includes(kw));
  if (isNewsQuery && !STOCK_TRIGGER.test(message)) return "web_search";

  const ticker = extractTicker(message);
  if (ticker) return "stock";
  if (STOCK_TRIGGER.test(message)) return "stock";

  return "web_search";
}

/* ─────────────────────────────────────────────
   WEB SEARCH — Multi-source with fallbacks
───────────────────────────────────────────── */

// Source 1: DuckDuckGo Instant Answer API (no scraping)
async function searchDDGInstant(query) {
  try {
    const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1&t=chatbot`;
    const res = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; ChatBot/1.0)",
        "Accept": "application/json",
      },
      signal: AbortSignal.timeout(5000),
    });

    if (!res.ok) return "";
    const data = await res.json();
    let output = "";

    if (data.Answer) {
      output += `✅ Answer: ${data.Answer}\n\n`;
    }
    if (data.AbstractText) {
      output += `📌 ${data.AbstractText}\n`;
      if (data.AbstractURL) output += `Source: ${data.AbstractURL}\n`;
      output += "\n";
    }
    if (data.Definition) {
      output += `📖 Definition: ${data.Definition}\n`;
      if (data.DefinitionURL) output += `Source: ${data.DefinitionURL}\n`;
      output += "\n";
    }
    if (data.RelatedTopics?.length) {
      const topics = data.RelatedTopics
        .filter((t) => t.Text)
        .slice(0, 5)
        .map((t, i) => `[${i + 1}] ${t.Text}${t.FirstURL ? `\n    🔗 ${t.FirstURL}` : ""}`)
        .join("\n");
      if (topics) output += `🔗 Related:\n${topics}\n`;
    }

    return output.trim();
  } catch (err) {
    console.warn("⚠️ DDG Instant failed:", err.message);
    return "";
  }
}

// Source 2: Wikipedia Search API
async function searchWikipedia(query) {
  try {
    const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&format=json&srlimit=3&origin=*`;
    const searchRes = await fetch(searchUrl, {
      headers: { "User-Agent": "ChatBot/1.0 (educational project)" },
      signal: AbortSignal.timeout(5000),
    });

    if (!searchRes.ok) return "";
    const searchData = await searchRes.json();
    const results = searchData?.query?.search || [];
    if (!results.length) return "";

    // Get extract for top result
    const topTitle = results[0].title;
    const extractUrl = `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles=${encodeURIComponent(topTitle)}&format=json&origin=*`;
    const extractRes = await fetch(extractUrl, {
      headers: { "User-Agent": "ChatBot/1.0 (educational project)" },
      signal: AbortSignal.timeout(5000),
    });

    let output = "";
    if (extractRes.ok) {
      const extractData = await extractRes.json();
      const pages = extractData?.query?.pages || {};
      const page = Object.values(pages)[0];
      if (page?.extract) {
        const shortExtract = page.extract.slice(0, 800).trim();
        output += `📚 Wikipedia — ${topTitle}:\n${shortExtract}\n🔗 https://en.wikipedia.org/wiki/${encodeURIComponent(topTitle)}\n\n`;
      }
    }

    // Add other search results as references
    if (results.length > 1) {
      output += "🔍 Also relevant:\n";
      results.slice(1).forEach((r, i) => {
        const snippet = r.snippet.replace(/<[^>]+>/g, "").slice(0, 150);
        output += `[${i + 2}] ${r.title}: ${snippet}...\n    🔗 https://en.wikipedia.org/wiki/${encodeURIComponent(r.title)}\n`;
      });
    }

    return output.trim();
  } catch (err) {
    console.warn("⚠️ Wikipedia search failed:", err.message);
    return "";
  }
}

// Source 3: NewsAPI (requires free API key — set NEWS_API_KEY in .env)
async function searchNewsAPI(query) {
  if (!process.env.NEWS_API_KEY) return "";
  try {
    const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&sortBy=publishedAt&pageSize=5&language=en&apiKey=${process.env.NEWS_API_KEY}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return "";
    const data = await res.json();
    if (!data.articles?.length) return "";

    let output = "📰 Latest News:\n\n";
    data.articles.slice(0, 5).forEach((article, i) => {
      output += `[${i + 1}] ${article.title}\n`;
      if (article.description) output += `    ${article.description.slice(0, 150)}\n`;
      output += `    📅 ${new Date(article.publishedAt).toLocaleDateString()}\n`;
      output += `    🔗 ${article.url}\n\n`;
    });
    return output.trim();
  } catch (err) {
    console.warn("⚠️ NewsAPI failed:", err.message);
    return "";
  }
}

// Source 4: GNews API (requires free API key — set GNEWS_API_KEY in .env)
async function searchGNews(query) {
  if (!process.env.GNEWS_API_KEY) return "";
  try {
    const url = `https://gnews.io/api/v4/search?q=${encodeURIComponent(query)}&lang=en&max=5&apikey=${process.env.GNEWS_API_KEY}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return "";
    const data = await res.json();
    if (!data.articles?.length) return "";

    let output = "📰 GNews Results:\n\n";
    data.articles.slice(0, 5).forEach((article, i) => {
      output += `[${i + 1}] ${article.title}\n`;
      if (article.description) output += `    ${article.description.slice(0, 150)}\n`;
      output += `    📅 ${new Date(article.publishedAt).toLocaleDateString()}\n`;
      output += `    🔗 ${article.url}\n\n`;
    });
    return output.trim();
  } catch (err) {
    console.warn("⚠️ GNews failed:", err.message);
    return "";
  }
}

// Source 5: Brave Search API (requires free API key — set BRAVE_SEARCH_KEY in .env)
async function searchBrave(query) {
  if (!process.env.BRAVE_SEARCH_KEY) return "";
  try {
    const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=5`;
    const res = await fetch(url, {
      headers: {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": process.env.BRAVE_SEARCH_KEY,
      },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return "";
    const data = await res.json();
    const webResults = data?.web?.results || [];
    if (!webResults.length) return "";

    let output = "🔍 Brave Search Results:\n\n";
    webResults.slice(0, 5).forEach((r, i) => {
      output += `[${i + 1}] ${r.title}\n`;
      if (r.description) output += `    ${r.description.slice(0, 200)}\n`;
      output += `    🔗 ${r.url}\n\n`;
    });
    return output.trim();
  } catch (err) {
    console.warn("⚠️ Brave Search failed:", err.message);
    return "";
  }
}

// Source 6: Serper.dev (Google Search proxy — set SERPER_API_KEY in .env)
async function searchSerper(query) {
  if (!process.env.SERPER_API_KEY) return "";
  try {
    const res = await fetch("https://google.serper.dev/search", {
      method: "POST",
      headers: {
        "X-API-KEY": process.env.SERPER_API_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ q: query, num: 5 }),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return "";
    const data = await res.json();

    let output = "";
    if (data.answerBox?.answer) {
      output += `✅ Direct Answer: ${data.answerBox.answer}\n\n`;
    }
    if (data.answerBox?.snippet) {
      output += `📌 Featured: ${data.answerBox.snippet}\n\n`;
    }
    if (data.organic?.length) {
      output += "🔍 Google Search Results:\n\n";
      data.organic.slice(0, 5).forEach((r, i) => {
        output += `[${i + 1}] ${r.title}\n`;
        if (r.snippet) output += `    ${r.snippet.slice(0, 200)}\n`;
        output += `    🔗 ${r.link}\n\n`;
      });
    }
    if (data.news?.length) {
      output += "📰 News:\n\n";
      data.news.slice(0, 3).forEach((r, i) => {
        output += `[${i + 1}] ${r.title}\n`;
        if (r.snippet) output += `    ${r.snippet.slice(0, 150)}\n`;
        output += `    🔗 ${r.link}\n\n`;
      });
    }
    return output.trim();
  } catch (err) {
    console.warn("⚠️ Serper failed:", err.message);
    return "";
  }
}

/* ─────────────────────────────────────────────
   MAIN SEARCH FUNCTION — tries all sources
───────────────────────────────────────────── */
async function performWebSearch(query) {
  console.log(`🔍 Searching: "${query}"`);

  // Run all available sources in parallel
  const [serper, brave, newsapi, gnews, ddg, wiki] = await Promise.all([
    searchSerper(query),
    searchBrave(query),
    searchNewsAPI(query),
    searchGNews(query),
    searchDDGInstant(query),
    searchWikipedia(query),
  ]);

  // Combine results (priority: Serper > Brave > News > DDG > Wiki)
  const parts = [serper, brave, newsapi, gnews, ddg, wiki].filter(Boolean);

  if (!parts.length) {
    return `No search results found for "${query}". The assistant will answer from its training knowledge.`;
  }

  return parts.join("\n\n---\n\n");
}

/* ─────────────────────────────────────────────
   STOCK PRICE (Yahoo Finance + Alpha Vantage fallback)
───────────────────────────────────────────── */
const stockPrice = tool(
  async ({ symbol }) => {
    console.log(`📈 Stock lookup: ${symbol}`);
    try {
      const urls = [
        `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=1d`,
        `https://query2.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=1d`,
        `https://query1.finance.yahoo.com/v10/finance/quoteSummary/${symbol}?modules=price`,
        `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${symbol}?modules=price`,
      ];

      let data = null;
      for (const url of urls) {
        try {
          const res = await fetch(url, {
            headers: {
              "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
              Accept: "application/json",
              "Accept-Language": "en-US,en;q=0.9",
            },
          });
          if (res.ok) { data = await res.json(); break; }
        } catch {}
      }

      const result = data?.chart?.result?.[0];
      if (result) {
        const meta        = result.meta;
        const price       = meta.regularMarketPrice?.toFixed(2)       ?? "N/A";
        const prevClose   = meta.chartPreviousClose ?? meta.previousClose;
        const high        = meta.regularMarketDayHigh?.toFixed(2)     ?? "N/A";
        const low         = meta.regularMarketDayLow?.toFixed(2)      ?? "N/A";
        const volume      = meta.regularMarketVolume?.toLocaleString() ?? "N/A";
        const marketCap   = meta.marketCap ? `$${(meta.marketCap / 1e9).toFixed(2)}B` : "N/A";
        const currency    = meta.currency     || "USD";
        const name        = meta.longName     || meta.shortName || symbol;
        const exchange    = meta.exchangeName || "";
        const marketState = meta.marketState  || "";

        let change = "N/A", changePct = "N/A", changeEmoji = "➡️";
        if (price !== "N/A" && prevClose) {
          const diff  = parseFloat(price) - prevClose;
          const pct   = (diff / prevClose) * 100;
          change      = (diff >= 0 ? "+" : "") + diff.toFixed(2);
          changePct   = (pct  >= 0 ? "+" : "") + pct.toFixed(2) + "%";
          changeEmoji = diff >= 0 ? "🟢" : "🔴";
        }

        return (
          `📊 ${name} (${symbol}) — ${exchange}\n` +
          `💰 Price: ${currency} ${price}  ${changeEmoji} ${change} (${changePct})\n` +
          `🔺 Day High: ${currency} ${high}\n` +
          `🔻 Day Low:  ${currency} ${low}\n` +
          `📦 Volume:   ${volume}\n` +
          `🏢 Market Cap: ${marketCap}\n` +
          `🕐 Market State: ${marketState}`
        );
      }

      if (process.env.ALPHA_VANTAGE_KEY) {
        console.log(`⚠️ Yahoo failed for ${symbol}, trying Alpha Vantage...`);
        try {
          const avRes  = await fetch(
            `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${process.env.ALPHA_VANTAGE_KEY}`
          );
          const avData = await avRes.json();
          const q      = avData["Global Quote"];
          if (q && q["05. price"]) {
            const price     = parseFloat(q["05. price"]).toFixed(2);
            const change    = parseFloat(q["09. change"]).toFixed(2);
            const changePct = q["10. change percent"];
            const high      = parseFloat(q["03. high"]).toFixed(2);
            const low       = parseFloat(q["04. low"]).toFixed(2);
            const volume    = parseInt(q["06. volume"]).toLocaleString();
            const emoji     = parseFloat(change) >= 0 ? "🟢" : "🔴";
            return (
              `📊 ${symbol} (via Alpha Vantage)\n` +
              `💰 Price: USD ${price}  ${emoji} ${change >= 0 ? "+" : ""}${change} (${changePct})\n` +
              `🔺 Day High: USD ${high}\n` +
              `🔻 Day Low:  USD ${low}\n` +
              `📦 Volume:   ${volume}`
            );
          }
        } catch (avErr) {
          console.error("❌ Alpha Vantage failed:", avErr.message);
        }
      }

      return `Could not fetch stock price for ${symbol}. Try again later.`;
    } catch (err) {
      return `Stock lookup failed for ${symbol}: ${err.message}`;
    }
  },
  {
    name: "get_stock_price",
    description: "Get live stock price for any ticker or company name.",
    schema: z.object({
      symbol: z.string().describe("Stock ticker e.g. AAPL, TSLA, NVDA"),
    }),
  }
);

/* ─────────────────────────────────────────────
   LLM — Groq
───────────────────────────────────────────── */
const llm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama-3.3-70b-versatile",
  temperature: 0,
});

/* ─────────────────────────────────────────────
   ROUTES
───────────────────────────────────────────── */

// ── CHAT ──────────────────────────────────
app.post("/chat", async (req, res) => {
  const { message, thread_id } = req.body;
  if (!message || !thread_id)
    return res.status(400).json({ error: "message and thread_id required" });

  console.log(`\n💬 [${thread_id.slice(0, 8)}] ${message}`);

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  try {
    const intent = detectIntent(message);
    console.log(`🗺️  Intent: "${intent}"`);

    let toolResult = "";

    if (intent === "web_search") {
      toolResult = await performWebSearch(message);

    } else if (intent === "rag") {
      if (!rag.ready) {
        toolResult = "No PDF uploaded yet. Please upload a PDF first.";
      } else {
        const docs = rag.store.retrieve(message, 10);
        if (!docs.length) {
          toolResult = "No relevant content found in the PDF.";
        } else {
          toolResult = `Content from "${rag.filename}":\n\n` +
            docs.map((d, i) => `[Section ${i + 1}]:\n${d.pageContent}`).join("\n\n---\n\n");
        }
      }

    } else if (intent === "math") {
      const match = message.match(/(\d+(?:\.\d+)?)\s*([\+\-\*\/x])\s*(\d+(?:\.\d+)?)/i);
      if (match) {
        const a     = parseFloat(match[1]);
        const b     = parseFloat(match[3]);
        const opMap = { "+": "add", "-": "sub", "*": "mul", x: "mul", "/": "div" };
        const op    = opMap[match[2].toLowerCase()] || "add";
        const results = { add: a + b, sub: a - b, mul: a * b, div: b === 0 ? "Division by zero" : a / b };
        toolResult = `Result: ${results[op]}`;
      } else {
        toolResult = "Could not parse math. Try: '25 * 4' or '100 / 5'.";
      }

    } else if (intent === "datetime") {
      const now = new Date();
      toolResult =
        `📅 Today's Date: ${now.toLocaleDateString("en-US", {
          weekday: "long", year: "numeric", month: "long", day: "numeric",
        })}\n` +
        `🕐 Current Time: ${now.toLocaleTimeString("en-US", {
          hour: "2-digit", minute: "2-digit", second: "2-digit", timeZoneName: "short",
        })}`;

    } else if (intent === "stock") {
      const symbol = extractTicker(message);
      console.log(`🎯 Ticker: "${symbol}" from: "${message}"`);
      if (symbol) {
        toolResult = await stockPrice.invoke({ symbol });
      } else {
        toolResult = "Could not identify a stock ticker. Try: 'Apple stock price' or 'TSLA stock'.";
      }
    }

    console.log(`🔧 Tool result: ${String(toolResult).slice(0, 150)}`);

    const prompt =
      `User asked: "${message}"\n\nTool returned:\n${toolResult}\n\n` +
      `Write a complete, helpful, well-formatted answer. Do NOT say you have limitations.`;

    const response = await llm.invoke([
      new SystemMessage("You are a helpful assistant. Answer using the tool result clearly and completely."),
      new HumanMessage(prompt),
    ]);

    const reply =
      typeof response.content === "string"
        ? response.content
        : response.content?.[0]?.text || "";

    console.log(`🤖 ${reply.slice(0, 120)}`);

    if (!chatHistories[thread_id]) chatHistories[thread_id] = [];
    chatHistories[thread_id].push({ role: "user",      content: message });
    chatHistories[thread_id].push({ role: "assistant", content: reply   });
    if (chatHistories[thread_id].length > 20)
      chatHistories[thread_id] = chatHistories[thread_id].slice(-20);

    const words = reply.split(" ");
    for (let i = 0; i < words.length; i++) {
      res.write((i === 0 ? "" : " ") + words[i]);
      await new Promise((r) => setTimeout(r, 18));
    }

  } catch (err) {
    console.error("❌ Chat error:", err.message);
    res.write(`Error: ${err.message}`);
  } finally {
    res.end();
  }
});

// ── HISTORY ───────────────────────────────
app.get("/history/:thread_id", (req, res) => {
  const history = chatHistories[req.params.thread_id] || [];
  res.json({ messages: history });
});

// ── UPLOAD PDF ────────────────────────────
app.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file)
    return res.status(400).json({ error: "No file uploaded" });
  if (!req.file.originalname.toLowerCase().endsWith(".pdf"))
    return res.status(400).json({ error: "Only PDF files are supported" });
  try {
    await processPDF(req.file.path, req.file.originalname);
    res.json({ message: `✅ "${req.file.originalname}" ready! Ask me anything about it.` });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── RAG STATUS ────────────────────────────
app.get("/rag-status", (_req, res) =>
  res.json({ ready: rag.ready, filename: rag.filename })
);

// ── HEALTH ────────────────────────────────
app.get("/health", (_req, res) =>
  res.json({ status: "ok", time: new Date().toISOString() })
);

app.post("/clear-pdf", (_req, res) => {
  rag.store = null;
  rag.ready = false;
  rag.filename = null;
  res.json({ success: true, message: "PDF removed successfully" });
});

/* ─────────────────────────────────────────────
   START SERVER
───────────────────────────────────────────── */
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});