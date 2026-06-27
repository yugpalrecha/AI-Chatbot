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
   ENV REQUIRED:
   GROQ_API_KEY=...
   TAVILY_API_KEY=...   <-- get free key at https://tavily.com (1000 req/month free)
   ALPHA_VANTAGE_KEY=... (optional, stock fallback)
───────────────────────────────────────────── */

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
  "jpmorgan": "JPM", "jp morgan": "JPM", "goldman sachs": "GS",
  "morgan stanley": "MS", "bank of america": "BAC",
  "wells fargo": "WFC", "citigroup": "C", "visa": "V",
  "mastercard": "MA", "paypal": "PYPL", "coinbase": "COIN",
  "blackrock": "BLK", "berkshire": "BRK-B",
  "rivian": "RIVN", "lucid": "LCID", "ford": "F",
  "general motors": "GM", "toyota": "TM", "ferrari": "RACE",
  "volkswagen": "VWAGY", "bmw": "BMWYY",
  "walmart": "WMT", "target": "TGT", "costco": "COST",
  "nike": "NKE", "starbucks": "SBUX", "mcdonalds": "MCD",
  "coca cola": "KO", "cocacola": "KO", "pepsi": "PEP",
  "pepsico": "PEP", "disney": "DIS",
  "pfizer": "PFE", "moderna": "MRNA", "johnson": "JNJ",
  "abbvie": "ABBV", "eli lilly": "LLY", "lilly": "LLY",
  "unitedhealth": "UNH", "cvs": "CVS", "merck": "MRK",
  "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX",
  "bp": "BP", "shell": "SHEL", "aramco": "2222.SR",
  "conocophillips": "COP",
  "micron": "MU", "arm": "ARM", "asml": "ASML",
  "lam research": "LRCX", "kla": "KLAC", "marvell": "MRVL",
  "skyworks": "SWKS", "analog devices": "ADI", "texas instruments": "TXN",
  "palantir": "PLTR", "snowflake": "SNOW", "shopify": "SHOP",
  "square": "SQ", "block": "SQ", "twilio": "TWLO",
  "datadog": "DDOG", "cloudflare": "NET", "crowdstrike": "CRWD",
  "palo alto": "PANW", "fortinet": "FTNT", "okta": "OKTA",
  "hubspot": "HUBS", "workday": "WDAY", "servicenow": "NOW",
  "mongodb": "MDB", "elastic": "ESTC", "confluent": "CFLT",
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
   PRIMARY WEB SEARCH — Tavily API
───────────────────────────────────────────── */
async function tavilySearch(query) {
  if (!process.env.TAVILY_API_KEY) {
    throw new Error("TAVILY_API_KEY not set");
  }

  const res = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      api_key: process.env.TAVILY_API_KEY,
      query,
      search_depth: "basic",
      include_answer: true,
      max_results: 6,
    }),
  });

  if (!res.ok) {
    throw new Error(`Tavily API error: ${res.status} ${res.statusText}`);
  }

  const data = await res.json();
  let output = "";

  if (data.answer) {
    output += `📌 Summary: ${data.answer}\n\n`;
  }

  if (data.results?.length) {
    output += "🔍 Search Results:\n\n";
    data.results.forEach((r, i) => {
      output += `[${i + 1}] ${r.title}\n`;
      output += `    ${(r.content || "").slice(0, 400)}\n`;
      output += `    🔗 ${r.url}\n\n`;
    });
  }

  if (!output.trim()) {
    throw new Error("Tavily returned no results");
  }

  return output.trim();
}

/* ─────────────────────────────────────────────
   FALLBACK WEB SEARCH — DuckDuckGo (used only if Tavily fails)
───────────────────────────────────────────── */
async function duckduckgoSearchFallback(query) {
  const headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
  };

  let output = "";

  try {
    const iaRes = await fetch(
      `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`,
      { headers }
    );
    if (iaRes.ok) {
      const ia = await iaRes.json();
      if (ia.AbstractText) {
        output += `📌 Summary: ${ia.AbstractText}\n`;
        if (ia.AbstractURL) output += `Source: ${ia.AbstractURL}\n`;
        output += "\n";
      }
      if (ia.Answer) output += `✅ Direct Answer: ${ia.Answer}\n\n`;
      if (ia.RelatedTopics?.length) {
        const topics = ia.RelatedTopics
          .filter((t) => t.Text).slice(0, 4)
          .map((t, i) => `[${i + 1}] ${t.Text}${t.FirstURL ? `\n    ${t.FirstURL}` : ""}`)
          .join("\n");
        if (topics) output += `Related:\n${topics}\n\n`;
      }
    }
  } catch (err) {
    console.warn("⚠️ DDG Instant Answer failed:", err.message);
  }

  try {
    const htmlRes = await fetch(
      `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`,
      { headers }
    );
    if (htmlRes.ok) {
      const html = await htmlRes.text();
      const stripTags = (s) =>
        s.replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&nbsp;/g, " ")
          .replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/\s+/g, " ").trim();

      const titles = [], snippets = [], urls = [];
      let m;
      const titleRegex   = /<a[^>]+class="result__a"[^>]*>([\s\S]*?)<\/a>/gi;
      const snippetRegex = /<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/gi;
      const urlRegex     = /<a[^>]+class="result__url"[^>]*>([\s\S]*?)<\/a>/gi;

      while ((m = titleRegex.exec(html))   !== null) titles.push(stripTags(m[1]));
      while ((m = snippetRegex.exec(html)) !== null) snippets.push(stripTags(m[1]));
      while ((m = urlRegex.exec(html))     !== null) urls.push(stripTags(m[1]));

      const count = Math.min(titles.length, snippets.length, 6);
      if (count > 0) {
        output += "🔍 Search Results:\n\n";
        for (let i = 0; i < count; i++) {
          output += `[${i + 1}] ${titles[i] || "Result"}\n`;
          output += `    ${snippets[i]}\n`;
          if (urls[i]) output += `    🔗 ${urls[i]}\n`;
          output += "\n";
        }
      }
    }
  } catch (err) {
    console.warn("⚠️ DDG HTML scrape failed:", err.message);
  }

  if (!output.trim()) return `No search results found for: "${query}". Try rephrasing.`;
  return output.trim();
}

/* ─────────────────────────────────────────────
   WEB SEARCH WRAPPER — tries Tavily first, falls back to DDG
───────────────────────────────────────────── */
async function webSearch(query) {
  try {
    console.log(`🔎 Trying Tavily for: "${query}"`);
    const result = await tavilySearch(query);
    console.log(`✅ Tavily succeeded`);
    return result;
  } catch (err) {
    console.warn(`⚠️ Tavily failed (${err.message}), falling back to DuckDuckGo`);
    return await duckduckgoSearchFallback(query);
  }
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
      toolResult = await webSearch(message);

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

    res.json({
        success: true,
        message: "PDF removed successfully"
    });
});

/* ─────────────────────────────────────────────
   START SERVER
───────────────────────────────────────────── */
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`Server running on ${PORT}`);
});