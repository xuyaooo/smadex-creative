// Direct browser → OpenRouter calls. The API key is exposed to the bundle
// (Vite-injected env var), so this is a demo-only configuration.
//
// Set VITE_OPENROUTER_API_KEY in front/.env.local for it to work.
//
// Two callable shapes:
//   extractMetadataFromImage(file)       → fills the form from the image
//   analyzeCreative(file, metadata)      → returns strengths/weaknesses/recommendation

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
// Fast multimodal models tuned for real-time prompting:
//   FAST_MODEL    — used for metadata extraction + creative analysis (full pass)
//   LIVE_MODEL    — used for live drawing tips (smallest + lowest latency)
// Gemini 2.5 Flash Lite hits ~250-400 ms TTFT on small image inputs, and is
// what we'd consider "personalized VLM" surrogate at runtime.
const FAST_MODEL = "google/gemini-2.5-flash-lite";
const LIVE_MODEL = "google/gemini-2.5-flash-lite";
const DEFAULT_MODEL = FAST_MODEL;

export const ENV_KEY = (import.meta as any).env?.VITE_OPENROUTER_API_KEY ?? "";

export interface ExtractedMetadata {
  // categorical (vertical + format are now also VLM-inferred so the form is
  // ~98% auto-filled — only campaign performance numbers stay manual)
  vertical: string;
  format: string;
  dominant_color: string;
  emotional_tone: string;
  theme: string;
  hook_type: string;
  cta_text: string;
  // booleans
  has_price: boolean;
  has_discount_badge: boolean;
  has_gameplay: boolean;
  has_ugc_style: boolean;
  // counts / 0–1 scores
  faces_count: number;
  product_count: number;
  text_density: number;
  readability_score: number;
  brand_visibility_score: number;
  clutter_score: number;
  novelty_score: number;
  motion_score: number;
  copy_length_chars: number;
  duration_sec: number;
}

export interface ColorSuggestion {
  hex: string;       // e.g. "#f97316"
  label: string;     // e.g. "Hot orange · CTA button"
  why?: string;      // causal reason this color would help
}

export interface CausalSuggestion {
  change: string;    // imperative: "Move CTA above the fold"
  why: string;       // causal reason: "Your CTA is 65% down — top performers land it at 50%"
}

export interface CreativeAnalysis {
  performance_summary: string;
  visual_strengths: string[];
  visual_weaknesses: string[];
  fatigue_risk_reason: string;
  top_recommendation: string;
  // Each recommendation now includes WHY it would help, not just WHAT to do.
  color_recommendations?: ColorSuggestion[];
  layout_recommendations?: CausalSuggestion[];
  copy_recommendations?: CausalSuggestion[];
}

// ---------- Image → base64 ----------
export function fileToDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

import JSON5 from "json5";
import { jsonrepair } from "jsonrepair";

// Robust JSON parser. LLMs frequently emit:
//   - markdown code fences (```json … ```)
//   - trailing commas before } or ]
//   - single-quoted strings
//   - unquoted keys
//   - truncated output (missing closing braces)
//   - extra prose around the JSON
// We strip the wrapper, then attempt JSON.parse with a progressive series of
// repairs. Each repair is opt-in only if the previous step throws.
function parseJSON(text: string): any {
  let t = (text ?? "").trim();

  // Strip markdown fences
  if (t.startsWith("```")) {
    const fenceClose = t.lastIndexOf("```");
    if (fenceClose > 0 && fenceClose !== 0) {
      t = t.slice(t.indexOf("\n") + 1, fenceClose);
    } else {
      t = t.slice(t.indexOf("\n") + 1);
    }
    if (t.toLowerCase().startsWith("json")) t = t.slice(4);
  }

  const s = t.indexOf("{");
  const e = t.lastIndexOf("}") + 1;
  if (s < 0) throw new Error("No JSON object in response: " + text.slice(0, 120));
  let raw = t.slice(s, e);

  // Attempt strict parse first.
  try { return JSON.parse(raw); } catch { /* fall through */ }

  // Repair pass — apply common fixes one by one.
  let repaired = raw
    // remove trailing commas before } or ]
    .replace(/,\s*([}\]])/g, "$1")
    // remove BOM and zero-width spaces inside strings
    .replace(/[​-‍﻿]/g, "");
  try { return JSON.parse(repaired); } catch { /* keep going */ }

  // Replace single quotes around keys/strings with double quotes — only when
  // the file uses NO double quotes at all (avoids breaking apostrophes).
  if (!repaired.includes('"')) {
    const swapped = repaired.replace(/'/g, '"');
    try { return JSON.parse(swapped); } catch { /* fall through */ }
  }

  // Quote unquoted keys: foo: → "foo":
  let withQuotedKeys = repaired.replace(/([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)/g, '$1"$2"$3');
  try { return JSON.parse(withQuotedKeys); } catch { /* fall through */ }

  // Last resort #1: if the JSON appears truncated, close any unmatched brackets.
  let truncated = withQuotedKeys;
  if (truncated.match(/[{[]/)) {
    let opens = 0, closes = 0;
    for (const ch of truncated) {
      if (ch === "{" || ch === "[") opens++;
      else if (ch === "}" || ch === "]") closes++;
    }
    truncated = truncated.replace(/,\s*$/, "");
    while (closes < opens) { truncated += "}"; closes++; }
    try { return JSON.parse(truncated); } catch { /* fall through */ }
  }

  // Last resort #2: JSON5. Forgiving: single-quoted strings, multi-line
  // strings, trailing commas, unquoted keys, comments.
  try { return JSON5.parse(truncated); } catch { /* fall through */ }
  try { return JSON5.parse(raw); } catch { /* fall through */ }

  // Last resort #3: jsonrepair — dedicated library for malformed JSON. Fixes
  // unescaped newlines inside strings, unescaped quotes, dangling commas,
  // missing closing brackets, smart quotes, NDJSON streams, and more.
  try { return JSON.parse(jsonrepair(raw)); } catch { /* fall through */ }
  try { return JSON.parse(jsonrepair(truncated)); } catch { /* fall through */ }

  // eslint-disable-next-line no-console
  console.warn("[parseJSON] all repairs failed; raw response tail:", raw.slice(-300));
  throw new Error(
    "Could not parse model JSON after 8 repair attempts. Tail: " +
    raw.slice(Math.max(0, raw.length - 200)),
  );
}

async function callOpenRouter(opts: {
  apiKey: string;
  imageDataUrl: string;
  systemPrompt: string;
  userPrompt: string;
  model?: string;
  maxTokens?: number;
}): Promise<any> {
  const { apiKey, imageDataUrl, systemPrompt, userPrompt, model = DEFAULT_MODEL, maxTokens = 800 } = opts;

  const res = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://smadex-creative-intelligence",
      "X-Title": "Smadex Creative Intelligence",
    },
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      temperature: 0.1,
      messages: [
        { role: "system", content: systemPrompt },
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: imageDataUrl } },
            { type: "text", text: userPrompt },
          ],
        },
      ],
    }),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`OpenRouter ${res.status}: ${txt.slice(0, 200)}`);
  }
  const json = await res.json();
  const text: string = json?.choices?.[0]?.message?.content ?? "";
  return parseJSON(text);
}

// ---------- Metadata extraction ----------
const EXTRACT_SYSTEM = (
  "You are a senior ad-creative analyst. Inspect the image and return ONLY a JSON object " +
  "with the requested fields. Do not include any other text. Distribute scores across the full 0-1 range."
);

// Default vocab — overridden at call time with the dataset's actual values
// when the predictions metadata.json is loaded.
const DEFAULT_VOCAB = {
  verticals:       ["gaming", "travel", "fintech", "ecommerce", "food_delivery", "entertainment"],
  formats:         ["rewarded_video", "interstitial", "banner", "native", "playable"],
  dominant_colors: ["purple", "blue", "green", "yellow", "orange", "red", "pink", "white", "black"],
  emotional_tones: ["exciting", "urgent", "calm", "playful", "professional", "aspirational", "mysterious", "friendly"],
  themes:          ["adventure", "lifestyle", "achievement", "social", "transformation", "luxury", "value", "family"],
  hook_types:      ["demo", "testimonial", "problem_solution", "before_after", "character_intro", "gameplay", "statistic", "question"],
};

function buildExtractPrompt(vocab: typeof DEFAULT_VOCAB): string {
  return `Inspect this ad creative image and fill the JSON below. Infer everything you can directly from the image — do not leave anything blank.

CRITICAL: every categorical field MUST be EXACTLY one of the listed allowed values. Pick the closest match, never invent new values.

- vertical: one of [${vocab.verticals.join(", ")}]
- format: one of [${vocab.formats.join(", ")}]
- dominant_color: one of [${vocab.dominant_colors.join(", ")}]
- emotional_tone: one of [${vocab.emotional_tones.join(", ")}]
- theme: one of [${vocab.themes.join(", ")}]
- hook_type: one of [${vocab.hook_types.join(", ")}]
- cta_text: short text on the visible CTA button (e.g. "Play Now", "Get 50% off", "Install"). If no visible button, infer the implied CTA in <=4 words.
- has_price: true if the image shows a numeric price tag
- has_discount_badge: true if it shows a % off / SALE / discount badge
- has_gameplay: true if it shows gameplay or in-app footage
- has_ugc_style: true if it looks like user-generated content (handheld, unpolished)
- faces_count: integer count of clearly visible human faces
- product_count: integer count of distinct products shown
- text_density: 0..1 — fraction of frame covered by text
- readability_score: 0..1 — how easy the text is to read
- brand_visibility_score: 0..1 — how prominently the brand/logo is visible
- clutter_score: 0..1 — how visually busy the composition is
- novelty_score: 0..1 — how non-template / fresh the design feels
- motion_score: 0..1 — implied motion / energy of the composition (if static image: how dynamic the composition feels)
- copy_length_chars: integer — approximate total characters of visible copy
- duration_sec: integer — implied video length in seconds. For a static image use 0; for a still that strongly suggests a 15s/30s spot use that.

Respond ONLY with this JSON (replace placeholder values with your answers):
{
  "vertical": "gaming",
  "format": "rewarded_video",
  "dominant_color": "purple",
  "emotional_tone": "exciting",
  "theme": "adventure",
  "hook_type": "demo",
  "cta_text": "Play Now",
  "has_price": false,
  "has_discount_badge": false,
  "has_gameplay": false,
  "has_ugc_style": false,
  "faces_count": 0,
  "product_count": 0,
  "text_density": 0.3,
  "readability_score": 0.8,
  "brand_visibility_score": 0.5,
  "clutter_score": 0.3,
  "novelty_score": 0.5,
  "motion_score": 0.5,
  "copy_length_chars": 40,
  "duration_sec": 0
}`;
}

export type ExtractVocab = Partial<typeof DEFAULT_VOCAB>;

export async function extractMetadataFromImage(
  file: File,
  apiKey: string = ENV_KEY,
  vocabOverride?: ExtractVocab,
): Promise<ExtractedMetadata> {
  if (!apiKey) throw new Error("Missing OpenRouter API key");
  const dataUrl = await fileToDataURL(file);
  // Inject the dataset's actual vocabulary into the prompt so the VLM never
  // emits values like "celebrity" or "2-for-1" that the trained model can't
  // recognize. Anything the override omits falls back to DEFAULT_VOCAB.
  const vocab = { ...DEFAULT_VOCAB, ...(vocabOverride ?? {}) };
  const userPrompt = buildExtractPrompt(vocab);
  const raw = await callOpenRouter({
    apiKey,
    imageDataUrl: dataUrl,
    systemPrompt: EXTRACT_SYSTEM,
    userPrompt,
    model: FAST_MODEL,
    maxTokens: 320,
  });
  // Coerce types defensively
  return {
    vertical: String(raw.vertical ?? "gaming").toLowerCase(),
    format: String(raw.format ?? "rewarded_video").toLowerCase(),
    dominant_color: String(raw.dominant_color ?? "purple").toLowerCase(),
    emotional_tone: String(raw.emotional_tone ?? "exciting").toLowerCase(),
    theme: String(raw.theme ?? "adventure").toLowerCase(),
    hook_type: String(raw.hook_type ?? "demo").toLowerCase(),
    cta_text: String(raw.cta_text ?? "").slice(0, 32),
    has_price: Boolean(raw.has_price),
    has_discount_badge: Boolean(raw.has_discount_badge),
    has_gameplay: Boolean(raw.has_gameplay),
    has_ugc_style: Boolean(raw.has_ugc_style),
    faces_count: Math.max(0, Math.round(Number(raw.faces_count ?? 0))),
    product_count: Math.max(0, Math.round(Number(raw.product_count ?? 0))),
    text_density: clamp01(raw.text_density),
    readability_score: clamp01(raw.readability_score),
    brand_visibility_score: clamp01(raw.brand_visibility_score),
    clutter_score: clamp01(raw.clutter_score),
    novelty_score: clamp01(raw.novelty_score),
    motion_score: clamp01(raw.motion_score),
    copy_length_chars: Math.max(0, Math.round(Number(raw.copy_length_chars ?? 0))),
    duration_sec: Math.max(0, Math.round(Number(raw.duration_sec ?? 0))),
  };
}

function clamp01(x: any): number {
  const n = Number(x);
  if (!Number.isFinite(n)) return 0.5;
  return Math.max(0, Math.min(1, n));
}

// ---------- Creative analysis ----------
const ANALYSIS_SYSTEM = (
  "You are an expert ad creative performance analyst. " +
  "Given a creative image and its metadata, produce a precise JSON analysis. " +
  "Be specific about visual elements you can actually see in the image."
);

export interface FullPrediction {
  status: string;
  health_score: number;
  probabilities: Record<string, number>;
  action?: string;
  early?: { ctr?: number; impressions?: number; spend?: number; revenue?: number; duration_sec?: number };
  counterfactuals?: { feat: string; from: string; to: string; score: number }[];
}

export async function analyzeCreative(
  file: File,
  metadata: Record<string, any>,
  predicted: FullPrediction,
  apiKey: string = ENV_KEY,
): Promise<CreativeAnalysis> {
  if (!apiKey) throw new Error("Missing OpenRouter API key");
  const dataUrl = await fileToDataURL(file);

  const cfBlock = (predicted.counterfactuals && predicted.counterfactuals.length)
    ? "\nEnsemble counterfactual suggestions (single-feature changes that would lift the score):\n" +
      predicted.counterfactuals.map((c) => `- change ${c.feat} from "${c.from}" to "${c.to}" → score ≈ ${c.score.toFixed(0)}`).join("\n")
    : "";

  const earlyBlock = predicted.early
    ? `\nEarly-life signal (first 7 days):
- CTR: ${num((predicted.early.ctr ?? 0) * 100)}%
- Impressions: ${(predicted.early.impressions ?? 0).toLocaleString()}
- Spend: $${(predicted.early.spend ?? 0).toLocaleString()}
- Revenue: $${(predicted.early.revenue ?? 0).toLocaleString()}
- Duration: ${predicted.early.duration_sec ?? 0}s`
    : "";

  const prompt = `Analyze this ad creative.

Creative metadata:
- Vertical: ${metadata.vertical} | Format: ${metadata.format}
- Theme: ${metadata.theme} | Hook: ${metadata.hook_type}
- CTA: ${metadata.cta_text ?? "unknown"} | Tone: ${metadata.emotional_tone} | Color: ${metadata.dominant_color}
- Text density: ${num(metadata.text_density)} | Readability: ${num(metadata.readability_score)}
- Brand visibility: ${num(metadata.brand_visibility_score)} | Clutter: ${num(metadata.clutter_score)}
- Novelty: ${num(metadata.novelty_score)} | Motion: ${num(metadata.motion_score)}
- Faces: ${metadata.faces_count ?? 0} | Products: ${metadata.product_count ?? 0}
- Has price: ${!!metadata.has_price} | Discount badge: ${!!metadata.has_discount_badge}
- Has gameplay: ${!!metadata.has_gameplay} | UGC style: ${!!metadata.has_ugc_style}
- Duration: ${metadata.duration_sec ?? 0}s
${earlyBlock}

Tabular ensemble output (XGBoost 5-seed bag + LightGBM + CatBoost + HistGBM + LogReg, soft-vote):
- Predicted status: ${predicted.status}
- Recommended action: ${predicted.action ?? "(derive from health score)"}
- Health score: ${predicted.health_score.toFixed(0)}/100
- Class probabilities:
    top_performer:  ${(predicted.probabilities.top_performer * 100).toFixed(1)}%
    stable:         ${(predicted.probabilities.stable * 100).toFixed(1)}%
    fatigued:       ${(predicted.probabilities.fatigued * 100).toFixed(1)}%
    underperformer: ${(predicted.probabilities.underperformer * 100).toFixed(1)}%${cfBlock}

Use ALL of the above (image + metadata + ensemble output + counterfactuals + early signal) when reasoning.

Respond ONLY with this JSON (no markdown, no extra text):
{
  "performance_summary": "1-2 sentences explaining why this creative is predicted to perform as it does. Reference both what you see in the image AND the ensemble output.",
  "visual_strengths":  ["concrete strength 1", "concrete strength 2"],
  "visual_weaknesses": ["concrete weakness 1", "concrete weakness 2"],
  "fatigue_risk_reason": "why is or isn't this creative at risk of fatigue? Reference the early-life signal and class probabilities.",
  "top_recommendation": "single most impactful change to improve this creative. Cite the counterfactual if it agrees.",
  "color_recommendations": [
    { "hex": "#ff5722", "label": "Hot orange · CTA button", "why": "Saturated warm CTAs lift click-rate ~18% over muted ones in this vertical." },
    { "hex": "#0f172a", "label": "Deep ink · headline backdrop", "why": "Forces a high-contrast pairing with the orange so the headline parses in <1s." },
    { "hex": "#facc15", "label": "Vivid yellow · discount badge", "why": "Yellow scarcity badges boost urgency; top performers in your vertical use them on 64% of creatives." }
  ],
  "layout_recommendations": [
    { "change": "Move the CTA above the fold and increase its size ~30%.", "why": "Your CTA sits at 65% screen height; top performers anchor it at 45%, which lifts click rate." },
    { "change": "Reduce headline copy from 18 words to 9 — value prop only.", "why": "Text density is currently 0.62, well above the 0.35 sweet spot for top-performing rewarded video." }
  ],
  "copy_recommendations": [
    { "change": "Swap 'Best deal ever' for outcome-led 'Save 50% in 30 seconds'.", "why": "Outcome-led headlines beat superlative ones by ~0.3% absolute CTR in our benchmark." },
    { "change": "Add a discount badge ('-50%') in the upper-right corner.", "why": "Discount badges trigger has_discount_badge=true, a strong positive signal in the ensemble." }
  ]
}

CRITICAL: every entry in color_recommendations / layout_recommendations / copy_recommendations MUST include a CAUSAL "why" tied to the actual metadata, ensemble probabilities, counterfactuals, or your knowledge of ad creative best practices. Each "why" should reference a concrete metric, score, or observation — never a vague platitude. Under 30 words.

For color_recommendations: 3 to 5 valid CSS hex colors that would actually improve this creative. Use the dominant_color, vertical, and detected weaknesses to pick. label under 6 words.
For layout_recommendations and copy_recommendations: 2 to 4 entries each. The "change" is the imperative action; the "why" is the data-grounded reason.`;

  const raw = await callOpenRouter({
    apiKey,
    imageDataUrl: dataUrl,
    systemPrompt: ANALYSIS_SYSTEM,
    userPrompt: prompt,
    model: FAST_MODEL,
    maxTokens: 750,
  });

  // Defensive parse for color recommendations: ensure each entry has a valid hex.
  const colors: ColorSuggestion[] = Array.isArray(raw.color_recommendations)
    ? raw.color_recommendations
        .map((c: any) => ({
          hex: typeof c?.hex === "string" && /^#?[0-9a-fA-F]{3,8}$/.test(c.hex)
            ? (c.hex.startsWith("#") ? c.hex : `#${c.hex}`)
            : "",
          label: String(c?.label ?? "").slice(0, 80),
          why: c?.why ? String(c.why).slice(0, 200) : undefined,
        }))
        .filter((c: ColorSuggestion) => c.hex && c.label)
    : [];

  // Layout / copy recommendations may come back as strings (older shape) or
  // {change, why} objects. Normalize both into CausalSuggestion[].
  const toCausal = (xs: any): CausalSuggestion[] => {
    if (!Array.isArray(xs)) return [];
    return xs.map((entry) => {
      if (typeof entry === "string") return { change: entry, why: "" };
      return {
        change: String(entry?.change ?? entry?.text ?? "").slice(0, 200),
        why: String(entry?.why ?? entry?.reason ?? "").slice(0, 200),
      };
    }).filter((c) => c.change);
  };

  return {
    performance_summary: String(raw.performance_summary ?? ""),
    visual_strengths: Array.isArray(raw.visual_strengths) ? raw.visual_strengths.map(String) : [],
    visual_weaknesses: Array.isArray(raw.visual_weaknesses) ? raw.visual_weaknesses.map(String) : [],
    fatigue_risk_reason: String(raw.fatigue_risk_reason ?? ""),
    top_recommendation: String(raw.top_recommendation ?? ""),
    color_recommendations: colors,
    layout_recommendations: toCausal(raw.layout_recommendations),
    copy_recommendations:   toCausal(raw.copy_recommendations),
  };
}

function num(x: any): string {
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(2) : "0.00";
}

// ---------- Image edit (Nano Banana / Gemini 2.5 Flash Image) ----------
// Returns a base64 data URL of the edited image plus the model's accompanying text.
// We use the OpenAI-compat chat endpoint with `modalities: ["image", "text"]`
// — that's how OpenRouter exposes Gemini's native image editing.
// Nano Banana on OpenRouter. The "-preview" suffix returns 404 — the live ID
// is google/gemini-2.5-flash-image. (Nano Banana 2 = google/gemini-3.1-flash-image-preview,
// Nano Banana Pro = google/gemini-3-pro-image-preview.)
const IMAGE_EDIT_MODEL = "google/gemini-2.5-flash-image";

export interface ImageEditResult {
  imageDataUrl: string | null;
  caption: string;
  raw?: any;
}

export async function editCreativeImage(
  file: File,
  instructions: string,
  apiKey: string = ENV_KEY,
): Promise<ImageEditResult> {
  const dataUrl = await fileToDataURL(file);
  return editCreativeImageFromDataUrl(dataUrl, instructions, apiKey);
}

export async function editCreativeImageFromDataUrl(
  dataUrl: string,
  instructions: string,
  apiKey: string = ENV_KEY,
): Promise<ImageEditResult> {
  if (!apiKey) throw new Error("Missing OpenRouter API key");

  // Per OpenRouter docs the order is ["text","image"] for image-output models.
  // The prompt is written in second person and explicitly demands a returned
  // image — Gemini occasionally reverts to text-only if the prompt sounds
  // analytical instead of generative.
  const userText =
    "You are editing the image attached above. " +
    "Generate a new ad creative image that applies the changes below. " +
    "Output ONLY the edited image plus a single short caption describing what changed. " +
    "Preserve the brand, product, and overall subject — do not redraw from scratch.\n\n" +
    "Changes to apply:\n" + instructions;

  const res = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://creative-ai-companion",
      "X-Title": "Creative.AI",
    },
    body: JSON.stringify({
      model: IMAGE_EDIT_MODEL,
      modalities: ["text", "image"],
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: dataUrl } },
            { type: "text", text: userText },
          ],
        },
      ],
    }),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`AI edit ${res.status}: ${txt.slice(0, 200)}`);
  }
  const json = await res.json();
  // Surface the raw response to the browser console so you can see exactly
  // what the model returned when an image fails to render.
  // eslint-disable-next-line no-console
  console.debug("[AI edit] raw response", json);

  const msg = json?.choices?.[0]?.message ?? {};

  // The image can live in three different shapes depending on provider/version:
  //  1. message.images[i].image_url.url       (OpenRouter's documented shape)
  //  2. message.content[i].image_url.url      (multimodal content array)
  //  3. embedded as a `data:image/...;base64` substring in message.content     (string)
  let imgUrl: string | null = null;
  let captionText = "";

  // (1) Top-level images array
  if (Array.isArray(msg.images)) {
    for (const im of msg.images) {
      const u = im?.image_url?.url ?? im?.url ?? null;
      if (typeof u === "string" && u.startsWith("data:image")) { imgUrl = u; break; }
    }
  }

  // (2) Multimodal content array  (3) string content
  if (Array.isArray(msg.content)) {
    for (const part of msg.content) {
      if (!part) continue;
      if (part.type === "image_url" && typeof part.image_url?.url === "string" && part.image_url.url.startsWith("data:image")) {
        if (!imgUrl) imgUrl = part.image_url.url;
      } else if (part.type === "text" && typeof part.text === "string") {
        captionText += (captionText ? " " : "") + part.text;
      }
    }
  } else if (typeof msg.content === "string") {
    captionText = msg.content;
    if (!imgUrl) {
      const m = msg.content.match(/data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+/);
      if (m) imgUrl = m[0];
    }
  }

  return { imageDataUrl: imgUrl, caption: captionText.trim(), raw: json };
}

// ---------- Live drawing recommendation ----------
// Used in draw mode — returns a single short tip from the latest canvas
// snapshot. `recentTips` is a list of prior tips the model has already given,
// so we can tell it not to repeat itself.
export async function liveDrawingTip(
  dataUrl: string,
  recentTips: string[] = [],
  apiKey: string = ENV_KEY,
): Promise<string> {
  if (!apiKey) throw new Error("Missing OpenRouter API key");

  const historyBlock = recentTips.length
    ? "\n\nYOU HAVE ALREADY GIVEN THESE TIPS — do NOT repeat them, do NOT paraphrase them. Keep momentum forward, address something NEW you can see now:\n" +
      recentTips.map((t, i) => `${recentTips.length - i}. "${t}"`).join("\n")
    : "";

  const raw = await callOpenRouter({
    apiKey,
    imageDataUrl: dataUrl,
    systemPrompt:
      "You are Maya — a senior ad-creative director with 15 years at Wieden+Kennedy and Smadex. " +
      "The designer just CIRCLED a region of an ad creative to ask your opinion on that specific area. " +
      "Look at the circled region first. Decide: is it fine as-is, or does it need a fix? " +
      "If fine: say so warmly in one short sentence (\"the brand mark sits perfectly there — leave it\"). " +
      "If it needs work: name the element they circled and give ONE concrete, sensory fix. " +
      "Vary phrasing across tips. Use sensory verbs (\"pop\", \"breathe\", \"cut\", \"land\", \"warm up\") not vague adjectives. " +
      "Speak like a human collaborator, not corporate AI. Output JSON only.",
    userPrompt:
      "The designer drew a closed loop around a specific element of this ad. " +
      "Identify what they circled (CTA, headline, product, badge, brand mark, background, face, copy block, etc.) and either approve it or give one concrete fix." +
      historyBlock +
      '\n\nReturn ONLY: { "tip": "one warm sentence (≤ 22 words) that names the element they circled, then either approves or gives one fix" }',
    model: LIVE_MODEL,
    maxTokens: 100,
  });
  return String(raw.tip ?? "Looks solid — that area's pulling its weight. Try circling another element to keep iterating.");
}
