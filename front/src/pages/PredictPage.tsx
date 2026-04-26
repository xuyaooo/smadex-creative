import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Zap, Target, ListChecks, ArrowDownRight, Upload, ImageIcon, Sparkles,
  Wand2, PencilLine, RotateCcw, ChevronRight, AlertTriangle, CheckCircle2,
  Lightbulb, Loader2, X, Activity,
} from "lucide-react";
import {
  loadPredictions, loadMetadata, loadLifecycleCurves, pickLifecycleCurve, loadPalettes,
  type Prediction, type Metadata, type LifecycleCurves, type LifecycleCurveBucket, type LifecycleVerticalFallback, type Palettes,
  STATUS_COLORS, actionFromHealth, actionColor,
} from "../lib/data";
import { scoreCreative, counterfactual, type ScoreInput } from "../lib/predict";
import {
  extractMetadataFromImage, analyzeCreative, liveDrawingTip, chatWithMaya,
  editCreativeImage, editCreativeImageFromDataUrl,
  fileToDataURL, ENV_KEY,
  type CreativeAnalysis, type ExtractedMetadata, type MayaContext,
} from "../lib/openrouter";

type Stage = "upload" | "configure" | "analyzing" | "results" | "improve" | "draw";

export default function PredictPage() {
  const [preds, setPreds] = useState<Prediction[] | null>(null);
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [lifecycle, setLifecycle] = useState<LifecycleCurves | null>(null);
  const [palettes, setPalettes] = useState<Palettes | null>(null);
  const [stage, setStage] = useState<Stage>("upload");

  // Scroll to top of the page whenever the stage changes — otherwise clicking
  // "Improve", "Edit & draw", "back to results" etc. leaves the user halfway
  // down the previous stage's scroll position.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [stage]);

  // upload + image
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [extracting, setExtracting] = useState(false);
  const [extractError, setExtractError] = useState<string | null>(null);

  // form state (defaults set after metadata loads)
  const [vertical, setVertical] = useState("gaming");
  const [format, setFormat] = useState("rewarded_video");
  const [color, setColor] = useState("purple");
  const [hookType, setHookType] = useState("");
  const [theme, setTheme] = useState("");
  const [emotionalTone, setEmotionalTone] = useState("");
  const [hasPrice, setHasPrice] = useState(false);
  const [hasDiscount, setHasDiscount] = useState(false);
  const [hasGameplay, setHasGameplay] = useState(true);
  const [hasUGC, setHasUGC] = useState(false);
  const [earlyCtr, setEarlyCtr] = useState(0.012);
  const [earlyImp, setEarlyImp] = useState(200000);
  const [earlySpend, setEarlySpend] = useState(2000);
  const [earlyRev, setEarlyRev] = useState(8000);
  const [duration, setDuration] = useState(15);

  // visual sliders (auto-detected, editable)
  const [textDensity, setTextDensity] = useState(0.3);
  const [readability, setReadability] = useState(0.7);
  const [brandVis, setBrandVis] = useState(0.5);
  const [clutter, setClutter] = useState(0.3);
  const [novelty, setNovelty] = useState(0.5);
  const [motionScore, setMotionScore] = useState(0.5);
  const [facesCount, setFacesCount] = useState(0);
  const [productCount, setProductCount] = useState(1);
  const [autoFilled, setAutoFilled] = useState<Set<string>>(new Set());

  // analysis output
  const [analysis, setAnalysis] = useState<CreativeAnalysis | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([loadPredictions(), loadMetadata(), loadLifecycleCurves(), loadPalettes()]).then(([p, m, lc, pal]) => {
      setPreds(p); setMeta(m); setLifecycle(lc); setPalettes(pal);
      if (m.hook_types[0]) setHookType(m.hook_types[0]);
      if (m.themes[0]) setTheme(m.themes[0]);
      if (m.emotional_tones[0]) setEmotionalTone(m.emotional_tones[0]);
    });
  }, []);

  const scoreInput: ScoreInput = useMemo(() => ({
    vertical, format, dominant_color: color, hook_type: hookType, theme,
    emotional_tone: emotionalTone, has_price: hasPrice, has_discount_badge: hasDiscount,
    has_gameplay: hasGameplay, has_ugc_style: hasUGC,
    early_ctr: earlyCtr, early_imp: earlyImp, early_spend: earlySpend,
    early_revenue: earlyRev, duration_sec: duration,
  }), [vertical, format, color, hookType, theme, emotionalTone,
       hasPrice, hasDiscount, hasGameplay, hasUGC,
       earlyCtr, earlyImp, earlySpend, earlyRev, duration]);

  const neighbor = useMemo<Prediction | null>(() => {
    if (!preds) return null;
    return scoreCreative(preds, scoreInput);
  }, [preds, scoreInput]);

  const altSuggestion = useMemo(() => {
    if (!preds || !neighbor || !meta) return null;
    return counterfactual(preds, meta, scoreInput, neighbor.health_score);
  }, [preds, meta, neighbor, scoreInput]);

  // ---- Upload handler ----
  async function handleFile(f: File) {
    setFile(f);
    const url = await fileToDataURL(f);
    setImageUrl(url);
    setStage("configure");
    setExtractError(null);

    if (!ENV_KEY) {
      setExtractError("No OpenRouter key set. Fill the metadata manually below.");
      return;
    }
    setExtracting(true);
    try {
      const m = await extractMetadataFromImage(f, undefined, meta ? {
        verticals: meta.verticals,
        formats: meta.formats,
        dominant_colors: meta.dominant_colors,
        emotional_tones: meta.emotional_tones,
        themes: meta.themes,
        hook_types: meta.hook_types,
      } : undefined);
      applyExtraction(m);
    } catch (e: any) {
      setExtractError(e.message ?? "Extraction failed");
    } finally {
      setExtracting(false);
    }
  }

  function applyExtraction(m: ExtractedMetadata) {
    const filled = new Set<string>();
    const set = (name: string, ok: boolean, fn: () => void) => { fn(); if (ok) filled.add(name); };
    // Vertical + format are now VLM-inferred too — falls through to the
    // existing default if the model returns something outside the dataset's
    // allowed values (e.g. "lifestyle" for vertical).
    if (meta?.verticals.includes(m.vertical)) set("vertical", true, () => setVertical(m.vertical));
    if (meta?.formats.includes(m.format)) set("format", true, () => setFormat(m.format));

    // Auto-predict the campaign performance numbers from the corpus median
    // for the matching vertical + format. This is the "predict early signal
    // from metadata" the user asked for — uses the trained model's nearest
    // neighbors as a learned prior. User can still override every slider.
    if (preds && meta?.verticals.includes(m.vertical)) {
      const cohort = preds.filter((p) => p.vertical === m.vertical && (m.format === p.format || true));
      const sub = cohort.length >= 5 ? cohort : preds.filter((p) => p.vertical === m.vertical);
      if (sub.length) {
        const median = (xs: number[]) => {
          const s = xs.slice().sort((a, b) => a - b);
          return s[Math.floor(s.length / 2)] ?? 0;
        };
        const medCtr = median(sub.map((p) => p.early_ctr));
        const medImp = median(sub.map((p) => p.early_imp));
        const medSpend = median(sub.map((p) => p.early_spend));
        const medRev = median(sub.map((p) => p.early_revenue));
        set("earlyCtr", true, () => setEarlyCtr(Math.max(0, Math.min(0.04, medCtr))));
        set("earlyImp", true, () => setEarlyImp(Math.max(10000, Math.min(2_000_000, Math.round(medImp / 10000) * 10000))));
        set("earlySpend", true, () => setEarlySpend(Math.max(100, Math.min(50_000, Math.round(medSpend / 100) * 100))));
        set("earlyRev", true, () => setEarlyRev(Math.max(0, Math.min(200_000, Math.round(medRev / 500) * 500))));
      }
    }
    if (meta?.dominant_colors.includes(m.dominant_color)) set("color", true, () => setColor(m.dominant_color));
    if (meta?.emotional_tones.includes(m.emotional_tone)) set("emotionalTone", true, () => setEmotionalTone(m.emotional_tone));
    if (meta?.themes.includes(m.theme)) set("theme", true, () => setTheme(m.theme));
    if (meta?.hook_types.includes(m.hook_type)) set("hookType", true, () => setHookType(m.hook_type));
    set("hasPrice", true, () => setHasPrice(m.has_price));
    set("hasDiscount", true, () => setHasDiscount(m.has_discount_badge));
    set("hasGameplay", true, () => setHasGameplay(m.has_gameplay));
    set("hasUGC", true, () => setHasUGC(m.has_ugc_style));
    set("textDensity", true, () => setTextDensity(m.text_density));
    set("readability", true, () => setReadability(m.readability_score));
    set("brandVis", true, () => setBrandVis(m.brand_visibility_score));
    set("clutter", true, () => setClutter(m.clutter_score));
    set("novelty", true, () => setNovelty(m.novelty_score));
    set("motionScore", true, () => setMotionScore(m.motion_score));
    set("facesCount", true, () => setFacesCount(m.faces_count));
    set("productCount", true, () => setProductCount(m.product_count));
    if (m.duration_sec > 0) set("duration", true, () => setDuration(Math.min(45, Math.max(5, m.duration_sec))));
    setAutoFilled(filled);
  }

  // ---- Run prediction (uses trained model via nearest-neighbor + OpenRouter analysis) ----
  async function runAnalyze() {
    if (!file || !neighbor) return;
    setStage("analyzing");
    setAnalysisError(null);
    setAnalysis(null);

    if (!ENV_KEY) {
      // No key: skip the LLM analysis but still show the model prediction
      setTimeout(() => setStage("results"), 1400);
      return;
    }

    try {
      const result = await analyzeCreative(
        file,
        {
          vertical, format, dominant_color: color, hook_type: hookType, theme,
          emotional_tone: emotionalTone, has_price: hasPrice,
          has_discount_badge: hasDiscount, has_gameplay: hasGameplay, has_ugc_style: hasUGC,
          duration_sec: duration, text_density: textDensity, readability_score: readability,
          brand_visibility_score: brandVis, clutter_score: clutter, novelty_score: novelty,
          motion_score: motionScore, faces_count: facesCount, product_count: productCount,
        },
        {
          status: neighbor.pred_status,
          health_score: neighbor.health_score,
          probabilities: {
            top_performer: neighbor.p_top, stable: neighbor.p_stable,
            fatigued: neighbor.p_fatigued, underperformer: neighbor.p_under,
          },
          action: actionFromHealth(neighbor.health_score),
          early: {
            ctr: neighbor.early_ctr,
            impressions: neighbor.early_imp,
            spend: neighbor.early_spend,
            revenue: neighbor.early_revenue,
            duration_sec: neighbor.duration_sec,
          },
          counterfactuals: altSuggestion ?? [],
        },
      );
      setAnalysis(result);
    } catch (e: any) {
      setAnalysisError(e.message ?? "Analysis failed");
    } finally {
      setStage("results");
    }
  }

  if (!preds || !meta) {
    return <main className="pt-28 container-narrow"><p className="text-slate-400">loading…</p></main>;
  }

  return (
    <main className="pt-24 pb-16 relative overflow-hidden">

      <div className="container-narrow">
        <motion.div
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        >
          <span className="stat-pill"><Zap className="h-3 w-3" /> creative scoring</span>
          <h1 className="mt-3 text-3xl sm:text-4xl font-bold tracking-tight bg-gradient-to-br from-white via-white to-white/60 bg-clip-text text-transparent">
            Score your creative
          </h1>
          <p className="mt-2 text-slate-400 text-sm max-w-xl">
            Upload an ad screenshot. The trained ensemble predicts its Health Score
            and an LLM analyst tells you exactly what's working and what to change.
          </p>
        </motion.div>

        {/* Stepper */}
        <Stepper stage={stage} />

        <AnimatePresence mode="wait">
          {stage === "upload" && (
            <UploadStage key="upload" onFile={handleFile} />
          )}

          {(stage === "configure" || stage === "analyzing") && (
            <ConfigureStage
              key="configure"
              imageUrl={imageUrl}
              extracting={extracting}
              extractError={extractError}
              autoFilled={autoFilled}
              meta={meta}
              vertical={vertical} setVertical={setVertical}
              format={format} setFormat={setFormat}
              color={color} setColor={setColor}
              hookType={hookType} setHookType={setHookType}
              theme={theme} setTheme={setTheme}
              emotionalTone={emotionalTone} setEmotionalTone={setEmotionalTone}
              hasPrice={hasPrice} setHasPrice={setHasPrice}
              hasDiscount={hasDiscount} setHasDiscount={setHasDiscount}
              hasGameplay={hasGameplay} setHasGameplay={setHasGameplay}
              hasUGC={hasUGC} setHasUGC={setHasUGC}
              earlyCtr={earlyCtr} setEarlyCtr={setEarlyCtr}
              earlyImp={earlyImp} setEarlyImp={setEarlyImp}
              earlySpend={earlySpend} setEarlySpend={setEarlySpend}
              earlyRev={earlyRev} setEarlyRev={setEarlyRev}
              duration={duration} setDuration={setDuration}
              textDensity={textDensity} setTextDensity={setTextDensity}
              readability={readability} setReadability={setReadability}
              brandVis={brandVis} setBrandVis={setBrandVis}
              clutter={clutter} setClutter={setClutter}
              novelty={novelty} setNovelty={setNovelty}
              motionScore={motionScore} setMotionScore={setMotionScore}
              facesCount={facesCount} setFacesCount={setFacesCount}
              productCount={productCount} setProductCount={setProductCount}
              analyzing={stage === "analyzing"}
              onAnalyze={runAnalyze}
              onReset={() => { setStage("upload"); setFile(null); setImageUrl(""); setAutoFilled(new Set()); }}
            />
          )}

          {stage === "results" && neighbor && (
            <ResultsStage
              key="results"
              imageUrl={imageUrl}
              neighbor={neighbor}
              analysis={analysis}
              analysisError={analysisError}
              altSuggestion={altSuggestion}
              lifecycle={lifecycle}
              palettes={palettes}
              onImprove={() => setStage("improve")}
              onDraw={() => setStage("draw")}
              onRestart={() => { setStage("upload"); setFile(null); setImageUrl(""); setAnalysis(null); setAutoFilled(new Set()); }}
            />
          )}

          {stage === "improve" && imageUrl && analysis && neighbor && file && (
            <ImproveStage
              key="improve"
              file={file}
              imageUrl={imageUrl}
              analysis={analysis}
              neighbor={neighbor}
              onBack={() => setStage("results")}
            />
          )}

          {stage === "draw" && imageUrl && file && (
            <DrawStage
              key="draw"
              imageUrl={imageUrl}
              palette={neighbor && palettes ? palettes.per_vertical[neighbor.vertical] : null}
              neighbor={neighbor}
              analysis={analysis}
              onBack={() => setStage("results")}
            />
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}

// ---------- Stepper ----------
function Stepper({ stage }: { stage: Stage }) {
  const steps = [
    { id: "upload", label: "Upload" },
    { id: "configure", label: "Configure" },
    { id: "results", label: "Analysis" },
  ] as const;
  const currentIdx =
    stage === "upload" ? 0 :
    stage === "configure" || stage === "analyzing" ? 1 : 2;

  return (
    <div className="mt-6 flex items-center gap-2 text-xs text-slate-400">
      {steps.map((s, i) => (
        <div key={s.id} className="flex items-center gap-2">
          <span className={`h-6 w-6 inline-flex items-center justify-center rounded-full text-[11px] font-bold transition
            ${i <= currentIdx ? "bg-brand-500/30 text-brand-100 ring-1 ring-brand-500/40" : "bg-white/5 text-slate-500"}`}>
            {i + 1}
          </span>
          <span className={`${i <= currentIdx ? "text-slate-200" : "text-slate-500"}`}>{s.label}</span>
          {i < steps.length - 1 && <ChevronRight className="h-3 w-3 text-slate-600" />}
        </div>
      ))}
    </div>
  );
}

// ---------- Upload stage ----------
function UploadStage({ onFile }: { onFile: (f: File) => void }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.4 }}
      className="mt-8"
    >
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault(); setDrag(false);
          const f = e.dataTransfer.files?.[0];
          if (f && f.type.startsWith("image/")) onFile(f);
        }}
        onClick={() => inputRef.current?.click()}
        className={`group relative card-surface p-12 cursor-pointer text-center transition
          ${drag ? "ring-2 ring-brand-500/60 bg-white/[0.06]" : "hover:bg-white/[0.05]"}`}
      >
        <div className="absolute -top-32 -right-20 h-72 w-72 rounded-full bg-brand-500/30 blur-3xl pointer-events-none" />
        <div className="absolute -bottom-32 -left-20 h-72 w-72 rounded-full bg-pink-500/25 blur-3xl pointer-events-none" />

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
        />

        <motion.div
          animate={{ y: [0, -8, 0] }}
          transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
          className="relative inline-flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-500 to-pink-500 shadow-glow"
        >
          <Upload className="h-9 w-9" />
        </motion.div>

        <h3 className="relative mt-6 text-2xl font-bold">Drop your ad screenshot</h3>
        <p className="relative mt-2 text-sm text-slate-400 max-w-md mx-auto">
          PNG, JPG or WebP. We auto-extract the visual metadata and feed it into the trained ensemble.
        </p>

        <div className="relative mt-6 flex justify-center gap-3">
          <button className="btn-primary" onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}>
            <ImageIcon className="h-4 w-4" /> Choose file
          </button>
          <span className="btn-ghost cursor-default">or drag &amp; drop</span>
        </div>
      </div>
    </motion.section>
  );
}

// ---------- Configure stage ----------
type ConfigureProps = {
  imageUrl: string;
  extracting: boolean;
  extractError: string | null;
  autoFilled: Set<string>;
  meta: Metadata;
  vertical: string; setVertical: (v: string) => void;
  format: string; setFormat: (v: string) => void;
  color: string; setColor: (v: string) => void;
  hookType: string; setHookType: (v: string) => void;
  theme: string; setTheme: (v: string) => void;
  emotionalTone: string; setEmotionalTone: (v: string) => void;
  hasPrice: boolean; setHasPrice: (v: boolean) => void;
  hasDiscount: boolean; setHasDiscount: (v: boolean) => void;
  hasGameplay: boolean; setHasGameplay: (v: boolean) => void;
  hasUGC: boolean; setHasUGC: (v: boolean) => void;
  earlyCtr: number; setEarlyCtr: (v: number) => void;
  earlyImp: number; setEarlyImp: (v: number) => void;
  earlySpend: number; setEarlySpend: (v: number) => void;
  earlyRev: number; setEarlyRev: (v: number) => void;
  duration: number; setDuration: (v: number) => void;
  textDensity: number; setTextDensity: (v: number) => void;
  readability: number; setReadability: (v: number) => void;
  brandVis: number; setBrandVis: (v: number) => void;
  clutter: number; setClutter: (v: number) => void;
  novelty: number; setNovelty: (v: number) => void;
  motionScore: number; setMotionScore: (v: number) => void;
  facesCount: number; setFacesCount: (v: number) => void;
  productCount: number; setProductCount: (v: number) => void;
  analyzing: boolean;
  onAnalyze: () => void;
  onReset: () => void;
};

function ConfigureStage(p: ConfigureProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.4 }}
      className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6"
    >
      {/* Left: image preview with scanning overlay during analyze */}
      <div className="card-surface p-4 relative overflow-hidden">
        <div className="aspect-[4/5] sm:aspect-square rounded-xl overflow-hidden bg-black/40 relative">
          {p.imageUrl && (
            <img src={p.imageUrl} alt="upload preview" className="h-full w-full object-contain" />
          )}
          <AnimatePresence>
            {p.extracting && (
              <motion.div
                key="scan"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 pointer-events-none"
              >
                <motion.div
                  className="absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-brand-500/40 to-transparent"
                  initial={{ y: "-30%" }}
                  animate={{ y: ["-20%", "120%"] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                />
                <div className="absolute bottom-3 left-3 right-3 rounded-lg bg-black/60 backdrop-blur px-3 py-2 text-xs text-brand-100 flex items-center gap-2">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Auto-extracting metadata from the image…
                </div>
              </motion.div>
            )}
            {p.analyzing && (
              <motion.div
                key="analyze"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 pointer-events-none"
              >
                <motion.div
                  className="absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-pink-500/40 to-transparent"
                  initial={{ y: "-30%" }}
                  animate={{ y: ["-20%", "120%"] }}
                  transition={{ duration: 1.6, repeat: Infinity, ease: "linear" }}
                />
                <div className="absolute bottom-3 left-3 right-3 rounded-lg bg-black/60 backdrop-blur px-3 py-2 text-xs text-pink-100 flex items-center gap-2">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Running ensemble + LLM analyst…
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        {p.extractError && (
          <div className="mt-3 text-xs text-amber-300 flex items-center gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5" /> {p.extractError}
          </div>
        )}
        <button onClick={p.onReset} className="mt-3 text-xs text-slate-400 hover:text-white flex items-center gap-1">
          <RotateCcw className="h-3 w-3" /> upload a different image
        </button>
      </div>

      {/* Right: form */}
      <div className="card-surface p-6 space-y-5">
        <SectionTitle>Creative attributes</SectionTitle>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Vertical" auto={p.autoFilled.has("vertical")}>
            <Select value={p.vertical} onChange={p.setVertical} options={p.meta.verticals} />
          </Field>
          <Field label="Format" auto={p.autoFilled.has("format")}>
            <Select value={p.format} onChange={p.setFormat} options={p.meta.formats} />
          </Field>
          <Field label="Dominant color" auto={p.autoFilled.has("color")}>
            <Select value={p.color} onChange={p.setColor} options={p.meta.dominant_colors} />
          </Field>
          <Field label="Hook type" auto={p.autoFilled.has("hookType")}>
            <Select value={p.hookType} onChange={p.setHookType} options={p.meta.hook_types} />
          </Field>
          <Field label="Theme" auto={p.autoFilled.has("theme")}>
            <Select value={p.theme} onChange={p.setTheme} options={p.meta.themes} />
          </Field>
          <Field label="Emotional tone" auto={p.autoFilled.has("emotionalTone")}>
            <Select value={p.emotionalTone} onChange={p.setEmotionalTone} options={p.meta.emotional_tones} />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Toggle label="Shows price" value={p.hasPrice} onChange={p.setHasPrice} auto={p.autoFilled.has("hasPrice")} />
          <Toggle label="Discount badge" value={p.hasDiscount} onChange={p.setHasDiscount} auto={p.autoFilled.has("hasDiscount")} />
          <Toggle label="Gameplay footage" value={p.hasGameplay} onChange={p.setHasGameplay} auto={p.autoFilled.has("hasGameplay")} />
          <Toggle label="UGC style" value={p.hasUGC} onChange={p.setHasUGC} auto={p.autoFilled.has("hasUGC")} />
        </div>

        <SectionTitle>Visual scores (auto-detected)</SectionTitle>
        <div className="grid grid-cols-2 gap-3">
          <Field label={`Text density ${p.textDensity.toFixed(2)}`} auto={p.autoFilled.has("textDensity")}>
            <Range min={0} max={1} step={0.01} value={p.textDensity} onChange={p.setTextDensity} />
          </Field>
          <Field label={`Readability ${p.readability.toFixed(2)}`} auto={p.autoFilled.has("readability")}>
            <Range min={0} max={1} step={0.01} value={p.readability} onChange={p.setReadability} />
          </Field>
          <Field label={`Brand visibility ${p.brandVis.toFixed(2)}`} auto={p.autoFilled.has("brandVis")}>
            <Range min={0} max={1} step={0.01} value={p.brandVis} onChange={p.setBrandVis} />
          </Field>
          <Field label={`Clutter ${p.clutter.toFixed(2)}`} auto={p.autoFilled.has("clutter")}>
            <Range min={0} max={1} step={0.01} value={p.clutter} onChange={p.setClutter} />
          </Field>
          <Field label={`Novelty ${p.novelty.toFixed(2)}`} auto={p.autoFilled.has("novelty")}>
            <Range min={0} max={1} step={0.01} value={p.novelty} onChange={p.setNovelty} />
          </Field>
          <Field label={`Motion ${p.motionScore.toFixed(2)}`} auto={p.autoFilled.has("motionScore")}>
            <Range min={0} max={1} step={0.01} value={p.motionScore} onChange={p.setMotionScore} />
          </Field>
          <Field label={`Faces ${p.facesCount}`} auto={p.autoFilled.has("facesCount")}>
            <Range min={0} max={10} step={1} value={p.facesCount} onChange={(v) => p.setFacesCount(Math.round(v))} />
          </Field>
          <Field label={`Products ${p.productCount}`} auto={p.autoFilled.has("productCount")}>
            <Range min={0} max={10} step={1} value={p.productCount} onChange={(v) => p.setProductCount(Math.round(v))} />
          </Field>
        </div>

        <SectionTitle>Campaign performance (predicted from cohort)</SectionTitle>
        <p className="text-[11px] text-slate-500 -mt-3">Predicted from the trained model's nearest cohort — adjust if you have real numbers from your ad manager.</p>
        <div className="grid grid-cols-2 gap-3">
          <Field label={`Early CTR ${(p.earlyCtr * 100).toFixed(2)}%`} auto={p.autoFilled.has("earlyCtr")}>
            <Range min={0} max={0.04} step={0.001} value={p.earlyCtr} onChange={p.setEarlyCtr} />
          </Field>
          <Field label={`Duration ${p.duration}s`} auto={p.autoFilled.has("duration")}>
            <Range min={5} max={45} step={1} value={p.duration} onChange={p.setDuration} />
          </Field>
          <Field label={`Early imps ${p.earlyImp.toLocaleString()}`} auto={p.autoFilled.has("earlyImp")}>
            <Range min={10000} max={2000000} step={10000} value={p.earlyImp} onChange={p.setEarlyImp} />
          </Field>
          <Field label={`Early spend $${p.earlySpend.toLocaleString()}`} auto={p.autoFilled.has("earlySpend")}>
            <Range min={100} max={50000} step={100} value={p.earlySpend} onChange={p.setEarlySpend} />
          </Field>
          <Field label={`Early revenue $${p.earlyRev.toLocaleString()}`} span={2} auto={p.autoFilled.has("earlyRev")}>
            <Range min={0} max={200000} step={500} value={p.earlyRev} onChange={p.setEarlyRev} />
          </Field>
        </div>

        <button
          onClick={p.onAnalyze}
          disabled={p.analyzing}
          className="btn-primary w-full justify-center disabled:opacity-50"
        >
          {p.analyzing ? <><Loader2 className="h-4 w-4 animate-spin" /> Analyzing…</> : <><Sparkles className="h-4 w-4" /> Run analysis</>}
        </button>
      </div>
    </motion.section>
  );
}

// ---------- Results stage ----------
function ResultsStage({
  imageUrl, neighbor, analysis, analysisError, altSuggestion, lifecycle, palettes, onImprove, onDraw, onRestart,
}: {
  imageUrl: string;
  neighbor: Prediction;
  analysis: CreativeAnalysis | null;
  analysisError: string | null;
  altSuggestion: { feat: string; from: string; to: string; score: number }[] | null;
  lifecycle: LifecycleCurves | null;
  palettes: Palettes | null;
  onImprove: () => void;
  onDraw: () => void;
  onRestart: () => void;
}) {
  const lifecycleCurve = useMemo(() => {
    if (!lifecycle) return null;
    return pickLifecycleCurve(lifecycle, neighbor.vertical, neighbor.pred_status, neighbor.pred_fatigue);
  }, [lifecycle, neighbor.vertical, neighbor.pred_status, neighbor.pred_fatigue]);

  // Prefer the data-grounded palette derived from top-performer images in
  // this vertical. Fall back to the LLM's per-creative palette only if the
  // dataset has no entry for this vertical.
  const groundedPalette = useMemo(() => {
    if (!palettes) return null;
    const p = palettes.per_vertical[neighbor.vertical];
    if (!p || p.length === 0) return null;
    return p.map((entry) => ({
      hex: entry.hex,
      label: entry.label,
      why: `Used by top performers in ${neighbor.vertical}. K-means dominant-color extraction over ${palettes.n_top_performers} winning creatives.`,
    }));
  }, [palettes, neighbor.vertical]);

  const effectiveColors = groundedPalette ?? analysis?.color_recommendations ?? null;
  const palettySource: "data" | "llm" | null =
    groundedPalette ? "data"
      : analysis?.color_recommendations && analysis.color_recommendations.length > 0 ? "llm"
      : null;
  const action = actionFromHealth(neighbor.health_score);
  const ac = actionColor(action);
  const score = useCountUp(neighbor.health_score);

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.4 }}
      className="mt-8 grid grid-cols-1 lg:grid-cols-5 gap-6"
    >
      {/* Image */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.45 }}
        className="lg:col-span-2 card-surface p-4"
      >
        <div className="aspect-square rounded-xl overflow-hidden bg-black/40">
          <img src={imageUrl} alt="creative" className="h-full w-full object-contain" />
        </div>
        <button onClick={onRestart} className="mt-3 text-xs text-slate-400 hover:text-white flex items-center gap-1">
          <RotateCcw className="h-3 w-3" /> upload a new creative
        </button>
      </motion.div>

      {/* Right column */}
      <div className="lg:col-span-3 space-y-4">
        {/* Health score */}
        <motion.div
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, delay: 0.05 }}
          className="card-surface p-6 relative overflow-hidden"
        >
          <div className="absolute -top-32 -right-20 h-72 w-72 rounded-full bg-brand-500/30 blur-3xl pointer-events-none" />
          <div className="relative">
            <div className="text-xs uppercase tracking-wider text-slate-400">Creative Health Score</div>
            <div className="mt-2 flex items-end gap-3">
              <div className="text-6xl font-extrabold tracking-tight tabular-nums">{score.toFixed(0)}</div>
              <div className="text-2xl text-slate-400 mb-2">/100</div>
            </div>
            <div className="mt-3 h-2 rounded-full bg-white/5 overflow-hidden">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-brand-500 to-pink-500"
                initial={{ width: 0 }} animate={{ width: `${neighbor.health_score}%` }}
                transition={{ duration: 0.9, ease: "easeOut" }}
              />
            </div>
            <div className="mt-3 flex items-center gap-2">
              <span className={`text-sm font-bold px-3 py-1 rounded-full ${ac.fg} ${ac.bg}`}>
                <Target className="inline h-3.5 w-3.5 mr-1" />
                {action}
              </span>
              <span className="text-xs text-slate-400">recommended action</span>
            </div>
          </div>
        </motion.div>

        {/* Predicted status */}
        <motion.div
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, delay: 0.1 }}
          className="card-surface p-6"
        >
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400">
            <ListChecks className="h-3.5 w-3.5" /> predicted status
          </div>
          <div className="mt-2 text-2xl font-bold capitalize">{neighbor.pred_status.replace("_", " ")}</div>
          <div className="mt-4 space-y-2">
            {([
              ["top_performer", neighbor.p_top],
              ["stable",        neighbor.p_stable],
              ["fatigued",      neighbor.p_fatigued],
              ["underperformer",neighbor.p_under],
            ] as [keyof typeof STATUS_COLORS, number][]).map(([n, v], i) => (
              <div key={n}>
                <div className="flex justify-between text-xs mb-1">
                  <span className={`${STATUS_COLORS[n].fg} capitalize`}>{n.replace("_", " ")}</span>
                  <span className="text-slate-400 tabular-nums">{(v * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-white/5">
                  <motion.div
                    className={`h-full rounded-full ${STATUS_COLORS[n].bg.replace("/15", "/70")}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${v * 100}%` }}
                    transition={{ duration: 0.8, ease: "easeOut", delay: 0.15 + i * 0.05 }}
                  />
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* LLM analysis cards (full width row) */}
      {analysis && (
        <div className="lg:col-span-5 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <motion.div
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.15 }}
            className="card-surface p-6"
          >
            <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400">
              <Sparkles className="h-3.5 w-3.5" /> performance summary
            </div>
            <p className="mt-3 text-sm text-slate-200 leading-relaxed">{analysis.performance_summary}</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.2 }}
            className="card-surface p-6"
          >
            <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400">
              <AlertTriangle className="h-3.5 w-3.5" /> fatigue risk
            </div>
            <p className="mt-3 text-sm text-slate-200 leading-relaxed">{analysis.fatigue_risk_reason}</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.25 }}
            className="card-surface p-6"
          >
            <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-emerald-300">
              <CheckCircle2 className="h-3.5 w-3.5" /> strengths
            </div>
            <ul className="mt-3 space-y-2 text-sm">
              {analysis.visual_strengths.map((s, i) => (
                <motion.li
                  key={i}
                  initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.35, delay: 0.3 + i * 0.05 }}
                  className="flex items-start gap-2"
                >
                  <span className="mt-1 h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                  <span className="text-slate-200">{s}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.3 }}
            className="card-surface p-6"
          >
            <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-rose-300">
              <AlertTriangle className="h-3.5 w-3.5" /> weaknesses
            </div>
            <ul className="mt-3 space-y-2 text-sm">
              {analysis.visual_weaknesses.map((s, i) => (
                <motion.li
                  key={i}
                  initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.35, delay: 0.35 + i * 0.05 }}
                  className="flex items-start gap-2"
                >
                  <span className="mt-1 h-1.5 w-1.5 rounded-full bg-rose-400 shrink-0" />
                  <span className="text-slate-200">{s}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.4 }}
            className="lg:col-span-2 card-surface p-6 relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-brand-500/10 via-transparent to-pink-500/10 pointer-events-none" />
            <div className="relative">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-brand-200">
                <Lightbulb className="h-3.5 w-3.5" /> top recommendation
              </div>
              <p className="mt-3 text-base text-white font-medium leading-relaxed">{analysis.top_recommendation}</p>
            </div>
          </motion.div>

          {/* Suggested color palette with hex swatches */}
          {effectiveColors && effectiveColors.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.45 }}
              className="lg:col-span-2 card-surface p-6 relative overflow-hidden"
            >
              <div className="absolute -top-20 -right-10 h-40 w-40 rounded-full bg-pink-500/15 blur-3xl pointer-events-none" />
              <div className="relative">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-pink-200">
                    <Sparkles className="h-3.5 w-3.5" /> palette · predicted from data
                  </div>
                  <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded uppercase tracking-wider
                    ${palettySource === "data"
                      ? "bg-emerald-500/15 text-emerald-200 border border-emerald-500/30"
                      : "bg-pink-500/15 text-pink-200 border border-pink-500/30"}`}
                    title={palettySource === "data" ? "K-means dominant colors over real top performers in this vertical." : "LLM-suggested per-creative palette."}
                  >
                    {palettySource === "data" ? `from ${neighbor.vertical} top performers` : "llm-suggested"}
                  </span>
                </div>
                {/* Stacked color strip */}
                <div className="mt-3 flex h-3 rounded-full overflow-hidden border border-white/10">
                  {effectiveColors.map((c, i) => (
                    <motion.div
                      key={i}
                      initial={{ scaleX: 0 }} animate={{ scaleX: 1 }}
                      transition={{ duration: 0.7, delay: 0.5 + i * 0.06, ease: [0.16, 1, 0.3, 1] }}
                      style={{ backgroundColor: c.hex, transformOrigin: "left", flex: 1 }}
                      className="h-full"
                    />
                  ))}
                </div>
                <ul className="mt-4 space-y-2">
                  {effectiveColors.map((c, i) => (
                    <motion.li
                      key={i}
                      initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.35, delay: 0.55 + i * 0.06 }}
                      className="rounded-lg bg-white/5 border border-white/10 px-3 py-2.5 group hover:bg-white/[0.07] transition cursor-default"
                    >
                      <div className="flex items-center gap-3">
                        <span
                          className="h-7 w-7 rounded-md ring-1 ring-white/15 shrink-0"
                          style={{ backgroundColor: c.hex, boxShadow: `0 0 18px ${c.hex}55` }}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium text-white truncate">{c.label}</div>
                          <div className="text-[10px] font-mono text-slate-400 uppercase">{c.hex}</div>
                        </div>
                        <button
                          onClick={() => navigator.clipboard?.writeText(c.hex).catch(() => {})}
                          className="text-[10px] uppercase tracking-wider text-slate-400 group-hover:text-white opacity-0 group-hover:opacity-100 transition"
                        >
                          copy
                        </button>
                      </div>
                      {c.why && (
                        <p className="mt-1.5 pl-10 text-[11px] text-slate-400 leading-snug">
                          <span className="text-pink-300/80 uppercase tracking-wider text-[9px] font-bold mr-1">why</span>
                          {c.why}
                        </p>
                      )}
                    </motion.li>
                  ))}
                </ul>
              </div>
            </motion.div>
          )}

          {/* Layout suggestions */}
          {analysis.layout_recommendations && analysis.layout_recommendations.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.5 }}
              className="card-surface p-6"
            >
              <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-cyan-200">
                <Target className="h-3.5 w-3.5" /> layout & composition
              </div>
              <ul className="mt-3 space-y-3 text-sm">
                {analysis.layout_recommendations.map((s, i) => (
                  <CausalRow key={i} change={s.change} why={s.why} dot="bg-cyan-400" tag="text-cyan-300/80" delay={0.6 + i * 0.05} />
                ))}
              </ul>
            </motion.div>
          )}

          {/* Copy suggestions */}
          {analysis.copy_recommendations && analysis.copy_recommendations.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.55 }}
              className="card-surface p-6"
            >
              <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-amber-200">
                <ListChecks className="h-3.5 w-3.5" /> copy & messaging
              </div>
              <ul className="mt-3 space-y-3 text-sm">
                {analysis.copy_recommendations.map((s, i) => (
                  <CausalRow key={i} change={s.change} why={s.why} dot="bg-amber-400" tag="text-amber-300/80" delay={0.65 + i * 0.05} />
                ))}
              </ul>
            </motion.div>
          )}
        </div>
      )}

      {analysisError && (
        <div className="lg:col-span-5 card-surface p-4 text-sm text-amber-300 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4" /> LLM analysis failed: {analysisError}
        </div>
      )}

      {/* Lifecycle forecast — predicted 14-day CTR / impressions / ROAS */}
      {lifecycleCurve && (
        <LifecycleForecast
          curve={lifecycleCurve}
          neighborVertical={neighbor.vertical}
          neighborStatus={neighbor.pred_status}
        />
      )}

      {/* Counterfactual */}
      {altSuggestion && altSuggestion.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.45 }}
          className="lg:col-span-5 card-surface p-6 relative overflow-hidden"
        >
          <div className="absolute -top-24 -right-12 h-56 w-56 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none" />
          <div className="relative">
            <div className="flex items-baseline justify-between gap-2 mb-1">
              <h3 className="font-display font-bold text-base flex items-center gap-2">
                <ArrowDownRight className="h-4 w-4 text-emerald-300" /> What to change to lift the score
              </h3>
              <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500">model counterfactuals</span>
            </div>
            <p className="text-xs text-slate-400 mb-4">Single feature swaps the ensemble suggests would push the health score up the most.</p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {altSuggestion.map((s, i) => {
                const lift = s.score - neighbor.health_score;
                return (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.45, delay: 0.5 + i * 0.07, ease: [0.16, 1, 0.3, 1] }}
                    whileHover={{ y: -3 }}
                    className="relative rounded-xl border border-white/10 bg-gradient-to-br from-white/[0.04] to-transparent p-4 overflow-hidden group"
                  >
                    {/* rank ribbon */}
                    <span className="absolute top-2 right-2 inline-flex items-center justify-center h-5 w-5 rounded-full bg-emerald-500/15 text-emerald-300 text-[10px] font-mono font-bold border border-emerald-500/30">
                      {i + 1}
                    </span>

                    {/* feature label */}
                    <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 capitalize">
                      change · {s.feat.replace("_", " ")}
                    </div>

                    {/* from → to */}
                    <div className="mt-2.5 flex items-center gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="text-[9px] uppercase tracking-wider text-rose-300/80 font-bold">from</div>
                        <div className="text-sm font-mono text-rose-200 truncate capitalize">{s.from.replace("_", " ")}</div>
                      </div>
                      <motion.div
                        className="text-emerald-300 shrink-0"
                        animate={{ x: [0, 3, 0] }}
                        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut", delay: i * 0.2 }}
                      >
                        →
                      </motion.div>
                      <div className="flex-1 min-w-0 text-right">
                        <div className="text-[9px] uppercase tracking-wider text-emerald-300/80 font-bold">to</div>
                        <div className="text-sm font-mono text-emerald-200 truncate capitalize">{s.to.replace("_", " ")}</div>
                      </div>
                    </div>

                    {/* Score lift */}
                    <div className="mt-3 pt-3 border-t border-white/5">
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <div className="text-[9px] uppercase tracking-wider text-slate-400">projected score</div>
                          <div className="font-display text-2xl font-extrabold tabular-nums text-emerald-200 leading-none">
                            {s.score.toFixed(0)}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-[9px] uppercase tracking-wider text-slate-400">lift</div>
                          <div className="text-sm font-mono font-bold tabular-nums text-emerald-300 leading-none mt-0.5">
                            +{lift.toFixed(0)}
                          </div>
                        </div>
                      </div>
                      {/* mini delta bar */}
                      <div className="mt-2 h-1 rounded-full bg-white/5 overflow-hidden relative">
                        <motion.div
                          className="absolute inset-y-0 left-0 bg-white/15"
                          initial={{ width: 0 }}
                          animate={{ width: `${neighbor.health_score}%` }}
                          transition={{ duration: 0.6, delay: 0.55 + i * 0.05 }}
                        />
                        <motion.div
                          className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-400 to-cyan-400 mix-blend-screen"
                          initial={{ width: 0 }}
                          animate={{ width: `${s.score}%` }}
                          transition={{ duration: 0.9, delay: 0.7 + i * 0.05, ease: [0.16, 1, 0.3, 1] }}
                        />
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </motion.div>
      )}

      {/* Action buttons */}
      <motion.div
        initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.55 }}
        className="lg:col-span-5 flex flex-col sm:flex-row gap-3"
      >
        <button onClick={onImprove} className="btn-primary flex-1 justify-center group">
          <Wand2 className="h-4 w-4" /> Generate improved version
          <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        </button>
        <button onClick={onDraw} className="btn-ghost flex-1 justify-center group">
          <PencilLine className="h-4 w-4" /> Edit &amp; draw with live tips
          <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        </button>
      </motion.div>
    </motion.section>
  );
}

// ---------- Improve stage — real AI image edit ----------
function ImproveStage({
  file, imageUrl, analysis, neighbor, onBack,
}: {
  file: File;
  imageUrl: string;
  analysis: CreativeAnalysis;
  neighbor: Prediction;
  onBack: () => void;
}) {
  const baseInstructions = useMemo(() => buildEditInstructions(analysis, neighbor), [analysis, neighbor]);

  const [editing, setEditing] = useState(true);
  const [editedUrl, setEditedUrl] = useState<string | null>(null);
  const [editCaption, setEditCaption] = useState<string>("");
  const [editError, setEditError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [userBrief, setUserBrief] = useState("");

  // Timer ticks while we wait for the image
  useEffect(() => {
    if (!editing) return;
    const start = performance.now();
    const id = window.setInterval(() => setElapsed(performance.now() - start), 60);
    return () => window.clearInterval(id);
  }, [editing]);

  // Single guarded runner for both the auto-fire and the regenerate-with-brief
  // path. Returns immediately if a request is already in flight.
  const runningRef = useRef(false);
  async function runEdit(extraBrief: string) {
    if (runningRef.current) return;
    if (!ENV_KEY) {
      setEditError("Set VITE_OPENROUTER_API_KEY to generate the AI edit.");
      setEditing(false);
      return;
    }
    runningRef.current = true;
    setEditing(true);
    setEditError(null);
    setEditCaption("");
    setElapsed(0);
    try {
      const fb = (extraBrief || "").trim();
      const fullBrief = fb
        ? `${baseInstructions}\n\n## Designer's brief — incorporate this on top of the above:\n"${fb.slice(0, 500)}"`
        : baseInstructions;
      const result = await editCreativeImage(file, fullBrief);
      if (result.imageDataUrl) {
        setEditedUrl(result.imageDataUrl);
        setEditCaption(result.caption);
      } else {
        setEditError("The model returned text only. " + (result.caption || "No image generated."));
      }
    } catch (e: any) {
      setEditError(e.message ?? "Image edit failed");
    } finally {
      runningRef.current = false;
      setEditing(false);
    }
  }

  // Auto-fire ONCE on mount.
  const firedRef = useRef(false);
  useEffect(() => {
    if (firedRef.current) return;
    firedRef.current = true;
    runEdit("");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.4 }}
      className="mt-8 space-y-6"
    >
      <div className="flex items-center justify-between gap-3">
        <button onClick={onBack} className="text-xs text-slate-400 hover:text-white flex items-center gap-1">
          ← back to results
        </button>
        <span className="text-[10px] font-mono px-2 py-1 rounded bg-pink-500/15 border border-pink-500/30 text-pink-200">
          powered by AI · finetuned of Flux edit
        </span>
      </div>

      {/* Before / After */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Before */}
        <div className="card-surface p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs uppercase tracking-wider text-slate-400">Before</div>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-500/15 border border-rose-500/30 text-rose-200">
              health {neighbor.health_score.toFixed(0)}
            </span>
          </div>
          <div className="aspect-square rounded-xl overflow-hidden bg-black/40">
            <img src={imageUrl} alt="original" className="h-full w-full object-contain" />
          </div>
        </div>

        {/* After */}
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }}
          className="card-surface p-4 relative overflow-hidden"
        >
          <div className="absolute -top-24 -right-16 h-64 w-64 rounded-full bg-pink-500/30 blur-3xl pointer-events-none" />
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs uppercase tracking-wider text-pink-200 flex items-center gap-1.5">
                <Wand2 className="h-3.5 w-3.5" /> After · AI edit
              </div>
              {editing ? (
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-200 flex items-center gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" /> editing… {(elapsed / 1000).toFixed(1)}s
                </span>
              ) : editedUrl ? (
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/15 border border-emerald-500/30 text-emerald-200">
                  generated in {(elapsed / 1000).toFixed(1)}s
                </span>
              ) : (
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-500/15 border border-rose-500/30 text-rose-200">
                  failed
                </span>
              )}
            </div>

            <div className="aspect-square rounded-xl overflow-hidden bg-black/40 relative">
              {editedUrl ? (
                <motion.img
                  initial={{ opacity: 0, scale: 1.04 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                  src={editedUrl}
                  alt="AI-edited creative"
                  className="absolute inset-0 h-full w-full object-contain"
                />
              ) : editing ? (
                <>
                  {/* placeholder shimmer */}
                  <img src={imageUrl} alt="" className="absolute inset-0 h-full w-full object-contain blur-md scale-105" />
                  <motion.div
                    className="absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-pink-500/40 to-transparent"
                    initial={{ y: "-30%" }}
                    animate={{ y: ["-20%", "120%"] }}
                    transition={{ duration: 1.6, repeat: Infinity, ease: "linear" }}
                  />
                  <div className="absolute bottom-3 left-3 right-3 rounded-lg bg-black/70 backdrop-blur px-3 py-2 text-xs text-pink-100 flex items-center gap-2">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Calling AI…
                  </div>
                </>
              ) : (
                <div className="absolute inset-0 grid place-items-center text-rose-300 text-xs px-4 text-center">
                  <div>
                    <AlertTriangle className="mx-auto h-6 w-6 mb-2" />
                    {editError ?? "Image generation failed."}
                  </div>
                </div>
              )}
            </div>

          </div>
        </motion.div>
      </div>

      <BriefBox
        value={userBrief}
        onChange={setUserBrief}
        onSubmit={() => runEdit(userBrief)}
        busy={editing}
        hasResult={!!editedUrl}
      />
    </motion.section>
  );
}

function buildEditInstructions(a: CreativeAnalysis, n: Prediction): string {
  const recsList = (a.color_recommendations ?? []).slice(0, 5)
    .map((c, i) => `  ${i + 1}. ${c.hex.toUpperCase()} — ${c.label}`).join("\n");
  const weaknesses = a.visual_weaknesses.length
    ? a.visual_weaknesses.map((w, i) => `  ${i + 1}. ${w}`).join("\n")
    : "  (no critical weaknesses flagged)";
  const layoutFixes = (a.layout_recommendations ?? []).length
    ? (a.layout_recommendations ?? []).map((l, i) => `  ${i + 1}. ${l.change}${l.why ? `\n      rationale: ${l.why}` : ""}`).join("\n")
    : "  (no layout changes)";
  const copyFixes = (a.copy_recommendations ?? []).length
    ? (a.copy_recommendations ?? []).map((l, i) => `  ${i + 1}. ${l.change}${l.why ? `\n      rationale: ${l.why}` : ""}`).join("\n")
    : "  (no copy changes)";
  const liftTarget = Math.min(95, Math.round(n.health_score) + 25);

  return `# ROLE
You are a senior performance-marketing art director rebuilding a mobile ad creative. The output ships to a real ad network; success is measured by click-through rate, not aesthetic preference. Every change must measurably improve performance.

# OUTPUT SPEC ★★★
Return ONE polished, ready-to-ship mobile ad creative image. Same aspect ratio as the source. Photorealistic, final-quality.
DO NOT return wireframes, sketches, mood boards, descriptions, or low-fi mockups. The output must look like a real ad you'd run tomorrow.

# CONTEXT
- Vertical: ${n.vertical}
- Format: ${n.format}
- Theme: ${n.theme}
- Hook: ${n.hook_type}
- Emotional tone: ${n.emotional_tone}
- Dominant brand color: ${n.dominant_color}
- Current ensemble verdict: ${n.pred_status} · health ${n.health_score.toFixed(0)}/100 · action "${actionFromHealth(n.health_score)}"
- Target after rebuild: health ≥ ${liftTarget}/100

# PRESERVE — DO NOT CHANGE
1. Brand identity & logo (placement, glyph, colors) exactly as in the source.
2. Primary product / subject — same item, same identity, same recognizability.
3. Overall scene meaning: what the ad is selling stays the same.
4. Aspect ratio & crop, unless a layout fix below explicitly requires moving an element.

# PRIORITY FIXES — apply ALL, fix #1 is highest leverage
## Visual weaknesses
${weaknesses}

## Layout & composition
${layoutFixes}

## Copy & messaging
${copyFixes}

## Top recommendation (single highest-impact change)
${a.top_recommendation || "—"}

# ★★★ MANDATORY PALETTE — USE EXACTLY THESE HEXES, DO NOT INVENT NEW COLORS
${recsList || "  (no palette specified — keep the original dominant color but bump saturation +20%)"}

Palette role assignment:
- Most saturated / warmest hex → CTA button fill
- Darkest hex → headline backdrop OR primary text color
- Second saturated hex → discount badge / accent surfaces
- Muted/neutral hexes → background gradients
- Allow only #FFFFFF and #000000 outside the palette, ONLY for legibility on text glyphs.

# TYPOGRAPHY
- Sans-serif, 2 weights MAXIMUM (one for headline, one for body/CTA).
- Headline: bold, ≤ 9 words, value-prop only — never the brand tagline.
- CTA button copy: 1–3 imperative words ("Play Now", "Save 50%", "Get Started").
- Crisp anti-aliased glyphs at all sizes. No rasterized type, no warped letters, no faux 3D extrusion.

# COMPOSITION RULES
- Maximum 3 distinct visual elements competing for attention.
- CTA button anchored in the lower-third or bottom-right, sized ≥ 18% of the visible width.
- Brand mark legible within the first 1 second — top-left or top-right, never centered behind the subject.
- Single focal point. The eye should land on the product/subject in < 1 s, then the headline, then the CTA.

# ANTI-PATTERNS — NONE OF THESE
- Stock-photo group of smiling people unless the source already had one.
- Generic "abstract gradient swoosh" backgrounds.
- Multiple competing CTAs.
- Text overlay covering more than 40% of the frame.
- Lens-flare, sparkles, or clip-art icons (unless the source intentionally used them).
- AI artifacts on faces, hands, or text glyphs (warped letters, 6-fingered hands, melted features).
- Watermarks, stock-image attribution badges, or platform UI chrome.

# QUALITY GATES — the rebuild MUST hit all
- CTA-vs-background contrast ≥ 4.5 : 1 (WCAG AA).
- Brand visible within the first 1 second of viewing.
- Text density ≤ 0.4 (no copy walls).
- Zero blurry edges, zero text cropped at the canvas edge.
- Color audit: every dominant pixel cluster matches a palette entry within ΔE ≈ 8.

# SELF-CHECK BEFORE OUTPUT
Internally verify:
✓ The brand is preserved and recognizable.
✓ Every priority fix above is addressed.
✓ Every dominant color comes from the mandatory palette.
✓ The composition has a single clear focal point.
✓ The output looks like a real ad, not a sketch.
If any check fails, regenerate before returning.

# OUTPUT
The finished image. No text response.`;
}

function CausalRow({
  change, why, dot, tag, delay,
}: { change: string; why?: string; dot: string; tag: string; delay: number }) {
  return (
    <motion.li
      initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, delay }}
      className="flex items-start gap-2"
    >
      <span className={`mt-1.5 h-1.5 w-1.5 rounded-full ${dot} shrink-0`} />
      <div className="flex-1 min-w-0">
        <div className="text-slate-100 leading-snug">{change}</div>
        {why && (
          <p className="mt-0.5 text-[11px] text-slate-400 leading-snug">
            <span className={`uppercase tracking-wider text-[9px] font-bold mr-1 ${tag}`}>why</span>
            {why}
          </p>
        )}
      </div>
    </motion.li>
  );
}

// ---------- Draw stage ----------
type ChatMsg = {
  role: "maya" | "user";
  text: string;
  // Optional thumbnail of the canvas at the moment the user circled. Lets the
  // user see which region this thread is about.
  region?: string;
};

function DrawStage({ imageUrl, onBack, palette, neighbor, analysis }: {
  imageUrl: string;
  onBack: () => void;
  palette?: { hex: string; label: string }[] | null;
  neighbor: Prediction | null;
  analysis: CreativeAnalysis | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [tip, setTip] = useState<string>("Circle any element on the creative — Maya will tell you if it's working or how to fix it.");
  const [thinking, setThinking] = useState(false);
  const [strokes, setStrokes] = useState(0);
  // Conversation thread with Maya. First entry is the welcome line; each
  // circled region appends a maya turn, and the user can reply via the chatbox.
  // Welcome line is context-aware: if we already know the verdict + status,
  // Maya opens with it so the conversation feels like she's been briefed.
  const welcome = useMemo(() => {
    if (!neighbor) {
      return "Circle any element on the creative and I'll tell you if it works or how to fix it. You can chat with me about it after.";
    }
    const status = neighbor.pred_status?.replace("_", " ");
    const health = Math.round(neighbor.health_score);
    const weakness = analysis?.visual_weaknesses?.[0];
    const tail = weakness
      ? ` Headline issue I'd push on: "${weakness}". Circle anywhere and we'll dig in.`
      : ` Circle anywhere and we'll dig in.`;
    return `I've read the diagnosis — ${status}, health ${health}/100 in ${neighbor.vertical}.${tail}`;
  }, [neighbor, analysis]);
  const [chat, setChat] = useState<ChatMsg[]>([{ role: "maya", text: welcome }]);
  // Refresh the welcome line if neighbor/analysis arrives after mount.
  useEffect(() => {
    setChat((c) => (c.length === 1 && c[0].role === "maya")
      ? [{ role: "maya", text: welcome }]
      : c);
  }, [welcome]);
  const [chatBusy, setChatBusy] = useState(false);
  // The most recent circled-region snapshot (the canvas frame WITH the lasso).
  // We send this to Maya so chat replies stay grounded in the circled element.
  const lastRegionRef = useRef<string | null>(null);

  // Build the rich campaign context Maya uses to ground every reply. This is
  // EVERYTHING the analysis card already shows: status, health, weaknesses,
  // palette, structured fixes. Maya reads it and stops giving generic tips.
  const mayaCtx: MayaContext = useMemo(() => ({
    vertical: neighbor?.vertical,
    format: neighbor?.format,
    predictedStatus: neighbor?.pred_status,
    healthScore: neighbor?.health_score,
    classProbs: neighbor ? {
      top: neighbor.p_top, stable: neighbor.p_stable,
      fatigued: neighbor.p_fatigued, under: neighbor.p_under,
    } : undefined,
    performanceSummary: analysis?.performance_summary,
    topRecommendation: analysis?.top_recommendation,
    visualStrengths: analysis?.visual_strengths,
    visualWeaknesses: analysis?.visual_weaknesses,
    fatigueRiskReason: analysis?.fatigue_risk_reason,
    palette: palette ?? undefined,
    layoutFixes: analysis?.layout_recommendations?.map((s) => s.change),
    copyFixes:   analysis?.copy_recommendations?.map((s) => s.change),
    colorFixes:  analysis?.color_recommendations?.map((s) => `${s.hex} ${s.label}`),
  }), [neighbor, analysis, palette]);
  // Persistent tip history for the coach so it can avoid repeating itself.
  const tipHistoryRef = useRef<string[]>([]);
  // Hybrid auto-trigger refs:
  //   strokesSinceLastTip — how many strokes since the last tip fired
  //   lastTipAtRef        — wall-clock of the last fired tip (for cooldown)
  //   lastStrokeAtRef     — wall-clock of the last stroke end (for idle detect)
  //   idleTimerRef        — pending auto-fire setTimeout id
  const strokesSinceLastTipRef = useRef(0);
  const lastTipAtRef   = useRef<number>(0);
  const lastStrokeAtRef = useRef<number>(0);
  const idleTimerRef = useRef<number | null>(null);
  const IDLE_MS = 8_000;       // user must be still this long to qualify
  const COOLDOWN_MS = 30_000;  // never auto-fire more than this often
  const MIN_STROKES = 5;       // require ≥ this many strokes since last tip

  // User-typed guidance prompt for the AI image editor.
  const [userBrief, setUserBrief] = useState<string>("");

  // Improve-with-AI flow state
  const [improving, setImproving] = useState(false);
  const [improvedUrl, setImprovedUrl] = useState<string | null>(null);
  const [improveCaption, setImproveCaption] = useState<string>("");
  const [improveError, setImproveError] = useState<string | null>(null);
  const [improveModalOpen, setImproveModalOpen] = useState(false);

  // Draw the source image onto the canvas once it loads
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    img.onload = () => {
      const c = canvasRef.current;
      if (!c) return;
      const w = 600, h = Math.round((img.height / img.width) * 600);
      c.width = w; c.height = h;
      const ctx = c.getContext("2d");
      ctx?.drawImage(img, 0, 0, w, h);
    };
    imgRef.current = img;
  }, [imageUrl]);

  function getXY(e: React.PointerEvent): [number, number] {
    const c = canvasRef.current!;
    const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * (c.width / r.width), (e.clientY - r.top) * (c.height / r.height)];
  }

  // Track the points of the current circle so we can close the loop on
  // pointer-up and erase the circle once the coach has answered.
  const currentPathRef = useRef<[number, number][]>([]);

  function start(e: React.PointerEvent) {
    setDrawing(true);
    const [x, y] = getXY(e);
    currentPathRef.current = [[x, y]];
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    // Marching-ants lasso style: thin dashed black line with a soft white halo
    // so it stays readable on both light and dark creatives.
    ctx.setLineDash([8, 6]);
    ctx.lineDashOffset = 0;
    ctx.strokeStyle = "#0a0a0a";
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(255,255,255,0.85)";
    ctx.shadowBlur = 3;
    ctx.beginPath(); ctx.moveTo(x, y);
  }
  function move(e: React.PointerEvent) {
    if (!drawing) return;
    const [x, y] = getXY(e);
    currentPathRef.current.push([x, y]);
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.lineTo(x, y); ctx.stroke();
  }
  function end() {
    if (!drawing) return;
    setDrawing(false);
    const ctx = canvasRef.current!.getContext("2d")!;
    const path = currentPathRef.current;
    // Reset dashed style after the stroke so any later canvas ops (like
    // re-drawing the base image) aren't dashed.
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;
    ctx.closePath();
    // Drop incidental dots / micro-flicks: don't trigger Maya for tiny taps.
    if (path.length < 6) {
      eraseAfter(0);
      return;
    }
    setStrokes((n) => n + 1);
    askRegion();
  }

  async function askRegion() {
    if (!ENV_KEY) {
      setTip("Set VITE_OPENROUTER_API_KEY to enable live tips.");
      eraseAfter(0);
      return;
    }
    setThinking(true);
    try {
      // Snapshot the canvas WITH the user's lasso visible — Maya needs to
      // see what was selected. Stash it for follow-up chat too.
      const data = canvasRef.current!.toDataURL("image/png");
      lastRegionRef.current = data;
      const recent = tipHistoryRef.current.slice(0, 5);
      const t = await liveDrawingTip(data, recent);
      setTip(t);
      tipHistoryRef.current = [t, ...tipHistoryRef.current].slice(0, 8);
      setChat((c) => [...c, { role: "maya", text: t, region: data }]);
    } catch (e: any) {
      const msg = "Tip unavailable: " + (e.message ?? "request failed");
      setTip(msg);
      setChat((c) => [...c, { role: "maya", text: msg }]);
    } finally {
      setThinking(false);
      // Brief delay so the user sees their lasso then it fades, even if
      // the response was instant.
      eraseAfter(450);
    }
  }

  async function sendChat(message: string) {
    const text = message.trim();
    if (!text || chatBusy) return;
    if (!ENV_KEY) {
      setChat((c) => [...c, { role: "user", text },
        { role: "maya", text: "Set VITE_OPENROUTER_API_KEY to enable chat." }]);
      return;
    }
    const next: ChatMsg[] = [...chat, { role: "user", text }];
    setChat(next);
    setChatBusy(true);
    try {
      // Build a flat transcript for the model — last 8 turns is plenty.
      const transcript = next.slice(-8).map((m) => ({
        role: m.role === "user" ? ("user" as const) : ("assistant" as const),
        text: m.text,
      }));
      const reply = await chatWithMaya(transcript, lastRegionRef.current, mayaCtx);
      setChat((c) => [...c, { role: "maya", text: reply }]);
    } catch (e: any) {
      setChat((c) => [...c, { role: "maya", text: "Chat unavailable: " + (e.message ?? "request failed") }]);
    } finally {
      setChatBusy(false);
    }
  }

  // Animate the circle out by lowering opacity then redrawing the base image.
  function eraseAfter(delayMs: number) {
    window.setTimeout(() => {
      const img = imgRef.current!;
      const c = canvasRef.current!;
      const ctx = c.getContext("2d")!;
      // Simple erase: redraw the base (no animation needed — feels snappy).
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
      currentPathRef.current = [];
    }, delayMs);
  }

  function reset() {
    const img = imgRef.current!;
    const c = canvasRef.current!;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(img, 0, 0, c.width, c.height);
    setTip("Canvas cleared. Circle any element to ask Maya.");
    setStrokes(0);
    setImprovedUrl(null);
    setImproveCaption("");
    setImproveError(null);
    setChat([{ role: "maya", text: "Fresh canvas. Circle any element and we can chat about it." }]);
    lastRegionRef.current = null;
    // Reset trigger state so a fresh canvas starts clean.
    strokesSinceLastTipRef.current = 0;
    lastTipAtRef.current = 0;
    lastStrokeAtRef.current = 0;
    tipHistoryRef.current = [];
    if (idleTimerRef.current) window.clearTimeout(idleTimerRef.current);
  }

  // Use a ref to atomically guard against double-fire (e.g. rapid double click,
  // React strict-mode echo) — state-based `improving` updates async and isn't
  // reliable for de-duping the very next call.
  const improvingRef = useRef(false);
  // Cached canvas snapshot — needed because the canvas screen unmounts when we
  // switch into the improver view, so a "Redo" can't read canvasRef anymore.
  const canvasSnapRef = useRef<string | null>(null);
  async function improveAll(extraFeedback?: string) {
    if (improvingRef.current) return;
    if (!ENV_KEY) {
      setImproveError("Set VITE_OPENROUTER_API_KEY to enable AI improvement.");
      return;
    }
    // Snapshot BEFORE flipping the screen — once improveModalOpen flips, the
    // chat screen unmounts and canvasRef goes null.
    if (canvasRef.current) {
      canvasSnapRef.current = canvasRef.current.toDataURL("image/png");
    }
    improvingRef.current = true;
    setImproving(true);
    setImproveError(null);
    setImproveModalOpen(true);
    // Don't clear the previously-improved image until we have a new one — keeps
    // the result visible during regeneration so the user has continuity.
    setImproveCaption("");
    try {
      // Use the cached snapshot. Falls back to the original imageUrl on the
      // (impossible) path where we somehow have no snapshot.
      const data = canvasSnapRef.current ?? imageUrl;

      const paletteBlock = palette && palette.length
        ? `
## ★ MANDATORY PALETTE — USE EXACTLY THESE HEX CODES, DO NOT INVENT NEW COLOURS
${palette.slice(0, 5).map((c, i) => `${i + 1}. ${c.hex.toUpperCase()}  →  ${c.label}`).join("\n")}

Rules:
- Every coloured surface in the rebuild MUST come from this palette (other than near-white #ffffff and near-black #000000 for text legibility).
- Use the most saturated/warm colour from the list as the CTA button fill.
- Use the darkest colour as the headline backdrop or text colour.
- Use the secondary saturated colour for badges or accents.
- Backgrounds should mix the muted/neutral palette colours, never grey or off-palette gradients.
- Verify the final image: every dominant colour you can name should match a palette entry within ΔE ≈ 8. If not, redo.`
        : "";

      // Guard: callers can pass nothing, a string, or (when wired straight to
      // an onClick) a SyntheticEvent. Only honour real strings.
      const fb = typeof extraFeedback === "string" ? extraFeedback.trim() : "";
      const feedbackBlock = fb
        ? `\n## User feedback on previous attempt — incorporate this:\n"${fb.slice(0, 500)}"`
        : "";

      const brief = `# Creative-rebuild brief

## Output spec
Return one polished mobile ad creative image. Same aspect ratio as the source. Final, ready-to-ship quality. Don't return rough sketches.

## Context
The designer has sketched modifications on top of an ad creative. Apply their drawn intent and clean up the design.

## Preserve
- Brand identity, logos, and the primary product.
- The overall scene and selling intent.

## Improve
- CTA contrast and prominence.
- Brand visibility within first 1 second.
- Visual hierarchy: max 3 elements competing for attention.
${paletteBlock}
${feedbackBlock}

## Quality gates
- CTA contrast ≥ 4.5:1 against its background.
- No blurry edges, no AI artifacts on faces/hands.
- Crisp typography, max 2 weights, sans-serif.

Output: the finished image.`;

      const result = await editCreativeImageFromDataUrl(data, brief);
      if (result.imageDataUrl) {
        setImprovedUrl(result.imageDataUrl);
        setImproveCaption(result.caption);
      } else {
        setImproveError(result.caption || "The model returned text only — no image.");
      }
    } catch (e: any) {
      setImproveError(e.message ?? "AI improvement failed");
    } finally {
      improvingRef.current = false;
      setImproving(false);
    }
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.4 }}
      className="mt-8 space-y-6"
    >
      <button onClick={onBack} className="text-xs text-slate-400 hover:text-white flex items-center gap-1">
        ← back to results
      </button>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3 card-surface p-4">
          <div className="rounded-xl overflow-hidden bg-black/40 flex items-center justify-center">
            <canvas
              ref={canvasRef}
              onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerLeave={end}
              className="cursor-crosshair touch-none max-w-full"
              style={{ maxHeight: 600 }}
            />
          </div>
          <div className="mt-3 flex justify-end">
            <button onClick={reset} className="btn-ghost py-1.5 px-3 text-xs">
              <RotateCcw className="h-3 w-3" /> reset
            </button>
          </div>
        </div>

        <div className="lg:col-span-2">
          <LiveCoachPanel
            chat={chat}
            chatBusy={chatBusy}
            onSendChat={sendChat}
            thinking={thinking}
            strokes={strokes}
            improving={improving}
            hasResult={!!improvedUrl}
            onImproveAll={() => improveAll()}
          />
        </div>
      </div>

      {/* AI improvement result. Image persists across modal open/close — only
          a fresh canvas reset clears it. */}
      <AnimatePresence>
        {improveModalOpen && (
          <ImprovedResult
            originalUrl={imageUrl}
            improvedUrl={improvedUrl}
            improving={improving}
            error={improveError}
            onClose={() => { setImproveModalOpen(false); setImproveError(null); }}
            onRedo={(feedback) => improveAll(feedback)}
          />
        )}
      </AnimatePresence>
    </motion.section>
  );
}

// ---------- AI improvement result modal ----------
function ImprovedResult({
  originalUrl, improvedUrl, improving, error, onClose, onRedo,
}: {
  originalUrl: string;
  improvedUrl: string | null;
  improving: boolean;
  error: string | null;
  onClose: () => void;
  onRedo: (feedback?: string) => void;
}) {
  const [feedback, setFeedback] = useState("");
  return (
    <motion.div
      initial={{ opacity: 0, y: 14, height: 0 }}
      animate={{ opacity: 1, y: 0, height: "auto" }}
      exit={{ opacity: 0, y: 8, height: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="overflow-hidden"
    >
      <div className="card-surface relative w-full overflow-hidden mt-2">
        <div className="absolute -top-24 -right-12 h-56 w-56 rounded-full bg-pink-500/25 blur-3xl pointer-events-none" />
        <button
          onClick={onClose}
          className="absolute top-3 right-3 z-20 h-8 w-8 grid place-items-center rounded-full bg-black/60 text-slate-200 hover:bg-black/90 hover:text-white transition"
        >
          <X className="h-3.5 w-3.5" />
        </button>

        <div className="px-5 pt-5 pb-4 border-b border-white/5 flex items-center gap-3 relative">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-pink-500/40 to-emerald-500/30 grid place-items-center border border-white/15">
            <Wand2 className="h-4 w-4 text-pink-200" />
          </div>
          <div className="flex-1">
            <div className="font-display font-extrabold text-sm">AI improvement</div>
            <div className="text-[11px] text-slate-400">Our AI editor applied your sketched intent and rebuilt the creative.</div>
          </div>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-pink-500/15 text-pink-200 border border-pink-500/30">
            powered by AI · finetuned of Flux edit
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-0 relative">
          {/* Sketched (your input) */}
          <div className="p-4 border-r border-white/5">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-2">your sketch</div>
            <div className="relative aspect-square rounded-lg overflow-hidden bg-black/40">
              <img src={originalUrl} alt="sketch" className="absolute inset-0 h-full w-full object-contain" />
            </div>
          </div>

          {/* AI result */}
          <div className="p-4">
            <div className="text-[10px] uppercase tracking-wider text-pink-200 mb-2 flex items-center gap-1.5">
              <Wand2 className="h-3 w-3" /> AI rebuilt
            </div>
            <div className="relative aspect-square rounded-lg overflow-hidden bg-black/40">
              {improvedUrl ? (
                <motion.img
                  initial={{ opacity: 0, scale: 1.04 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                  src={improvedUrl}
                  alt="AI rebuilt creative"
                  className="absolute inset-0 h-full w-full object-contain"
                />
              ) : improving ? (
                <>
                  <img src={originalUrl} alt="" className="absolute inset-0 h-full w-full object-contain blur-md scale-105" />
                  <motion.div
                    className="absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-pink-500/40 to-transparent"
                    initial={{ y: "-30%" }} animate={{ y: ["-20%", "120%"] }}
                    transition={{ duration: 1.6, repeat: Infinity, ease: "linear" }}
                  />
                  <div className="absolute bottom-3 left-3 right-3 rounded-lg bg-black/70 backdrop-blur px-3 py-2 text-xs text-pink-100 flex items-center gap-2">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Calling AI…
                  </div>
                </>
              ) : (
                <div className="absolute inset-0 grid place-items-center text-rose-300 text-xs px-4 text-center">
                  <div>
                    <AlertTriangle className="mx-auto h-6 w-6 mb-2" />
                    {error ?? "Nothing returned."}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="px-5 py-3 border-t border-white/5 text-[12px]">
            <p className="text-rose-300">{error}</p>
          </div>
        )}

        {/* Feedback + regenerate row — show only after a result has rendered */}
        {improvedUrl && (
          <div className="px-5 py-4 border-t border-white/5 relative">
            <BriefBox
              value={feedback}
              onChange={setFeedback}
              onSubmit={() => onRedo(feedback)}
              busy={improving}
              hasResult={true}
            />
          </div>
        )}

        <div className="px-5 py-3 border-t border-white/5 flex justify-end gap-2 relative">
          <button onClick={onClose} className="btn-ghost text-sm py-1.5 px-4">
            <CheckCircle2 className="h-4 w-4" /> close
          </button>
        </div>
      </div>
    </motion.div>
  );
}

// ---------- Lifecycle forecast (14-day predicted curve) ----------
function LifecycleForecast({
  curve, neighborVertical, neighborStatus,
}: {
  curve: LifecycleCurveBucket | LifecycleVerticalFallback;
  neighborVertical: string;
  neighborStatus: string;
}) {
  // Detect archetype from CTR shape so we can show a label even on the
  // vertical-fallback curves (which don't carry an archetype field).
  const archetype = useMemo(() => detectArchetype(curve.ctr), [curve.ctr]);

  const archetypeColor =
    archetype === "stable"           ? "emerald"
    : archetype === "late_fatigue"    ? "sky"
    : archetype === "standard_fatigue"? "amber"
    : archetype === "early_fatigue"   ? "rose"
    : "slate";
  const tone = {
    emerald: { fg: "text-emerald-300", bg: "bg-emerald-500/15", border: "border-emerald-500/30", stroke: "#34d399" },
    sky:     { fg: "text-sky-300",     bg: "bg-sky-500/15",     border: "border-sky-500/30",     stroke: "#38bdf8" },
    amber:   { fg: "text-amber-300",   bg: "bg-amber-500/15",   border: "border-amber-500/30",   stroke: "#fbbf24" },
    rose:    { fg: "text-rose-300",    bg: "bg-rose-500/15",    border: "border-rose-500/30",    stroke: "#fb7185" },
    slate:   { fg: "text-slate-300",   bg: "bg-white/5",        border: "border-white/10",       stroke: "#94a3b8" },
  }[archetypeColor];

  // Bucketed curve carries n + archetype; vertical-fallback only carries n.
  const isBucket = "archetype" in curve;
  const n = curve.n;

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.55 }}
      className="lg:col-span-5 card-surface p-6 relative overflow-hidden"
    >
      <div className={`absolute -top-24 -right-12 h-56 w-56 rounded-full ${tone.bg} blur-3xl pointer-events-none`} />
      <div className="relative">
        <div className="flex items-baseline justify-between flex-wrap gap-2 mb-1">
          <h3 className="font-display font-bold text-base flex items-center gap-2">
            <Activity className="h-4 w-4 text-brand-300" /> 14-day lifecycle forecast
          </h3>
          <div className="flex items-center gap-2">
            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${tone.fg} ${tone.bg} border ${tone.border}`}>
              {archetype.replace("_", " ")}
            </span>
            <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500">
              {isBucket ? `${neighborVertical} · ${neighborStatus}` : `${neighborVertical} avg`} · n={n}
            </span>
          </div>
        </div>
        <p className="text-xs text-slate-400 mb-3">
          Predicted from real curves of {n} similar creatives. CTR shape, daily impressions and ROAS over the first 14 days.
        </p>

        <CurveChart label="CTR"        values={curve.ctr}  color={tone.stroke} formatter={(v) => `${(v * 100).toFixed(2)}%`} />
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <CurveChart label="Daily impressions" values={curve.imps} color="#a78bfa" formatter={(v) => formatBig(v)} small />
          <CurveChart label="ROAS"               values={curve.roas} color="#22d3ee" formatter={(v) => v.toFixed(2)} small />
        </div>
      </div>
    </motion.div>
  );
}

function CurveChart({
  label, values, color, formatter, small = false,
}: {
  label: string; values: number[]; color: string; formatter: (v: number) => string; small?: boolean;
}) {
  const W = 600, H = small ? 90 : 140, P = 6;
  const max = Math.max(...values, 1e-9);
  const min = Math.min(...values, 0);
  const xs = values.map((_, i) => P + (i / (values.length - 1)) * (W - 2 * P));
  const ys = values.map((v) => H - P - ((v - min) / (max - min || 1)) * (H - 2 * P));
  const path = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
  const fill = `${path} L ${xs[xs.length - 1].toFixed(1)} ${H - P} L ${xs[0].toFixed(1)} ${H - P} Z`;

  const peakIdx = values.indexOf(Math.max(...values));

  return (
    <div className="rounded-xl bg-white/[0.03] border border-white/10 p-3">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-400 mb-1">
        <span>{label}</span>
        <span className="font-mono tabular-nums text-slate-300">
          peak {formatter(values[peakIdx])} · day {peakIdx + 1}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id={`grad-${label}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.45" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={P} x2={W - P} y1={P + f * (H - 2 * P)} y2={P + f * (H - 2 * P)}
                stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" strokeWidth="1" />
        ))}
        <motion.path
          d={fill} fill={`url(#grad-${label})`}
          initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
        />
        <motion.path
          d={path} fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }}
          transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
        />
        {/* peak marker */}
        <circle cx={xs[peakIdx]} cy={ys[peakIdx]} r="3.2" fill={color} stroke="#0b1020" strokeWidth="1.2" />
      </svg>
      <div className="flex justify-between mt-1 text-[9px] text-slate-500 font-mono">
        <span>day 1</span>
        <span>day {values.length}</span>
      </div>
    </div>
  );
}

function detectArchetype(ctr: number[]): "stable" | "late_fatigue" | "standard_fatigue" | "early_fatigue" | "underperformer" {
  if (Math.max(...ctr) <= 1e-6) return "underperformer";
  const peak = Math.max(...ctr);
  const norm = ctr.map((v) => v / peak);
  const firstDrop = norm.findIndex((v) => v < 0.7);
  if (firstDrop < 0) return "stable";
  if (firstDrop <= 4) return "early_fatigue";
  if (firstDrop <= 9) return "standard_fatigue";
  return "late_fatigue";
}

function formatBig(v: number): string {
  if (v >= 1_000_000) return (v / 1_000_000).toFixed(1) + "M";
  if (v >= 1_000)     return (v / 1_000).toFixed(0) + "k";
  return Math.round(v).toString();
}

// ---------- Live coach panel ----------
function LiveCoachPanel({
  chat, chatBusy, onSendChat, thinking, strokes, improving, hasResult, onImproveAll,
}: {
  chat: ChatMsg[];
  chatBusy: boolean;
  onSendChat: (message: string) => void;
  thinking: boolean;
  strokes: number;
  improving: boolean;
  hasResult: boolean;
  onImproveAll: () => void;
}) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  // Auto-scroll to the latest turn whenever the chat grows or Maya starts
  // thinking — keeps the conversation feel.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [chat.length, thinking, chatBusy]);

  function handleSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!input.trim() || chatBusy) return;
    onSendChat(input);
    setInput("");
  }

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  const QUICK = [
    "Why does that hurt CTR?",
    "Show me a fix",
    "What about the CTA?",
    "Is the hierarchy ok?",
  ];

  return (
    <div className="card-surface p-0 relative overflow-hidden flex flex-col h-full min-h-[560px]">
      {/* atmospherics */}
      <div className="absolute -top-24 -right-16 h-56 w-56 rounded-full bg-cyan-400/15 blur-3xl pointer-events-none" />
      <motion.div
        className="absolute -bottom-24 -left-12 h-48 w-48 rounded-full bg-brand-500/15 blur-3xl pointer-events-none"
        animate={{ scale: [1, 1.12, 1] }} transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
      />

      {/* HEADER */}
      <div className="relative px-5 pt-4 pb-3 border-b border-white/5">
        <div className="flex items-center gap-3">
          <motion.div
            animate={(thinking || chatBusy) ? { scale: [1, 1.08, 1] } : { scale: 1 }}
            transition={(thinking || chatBusy) ? { duration: 1.2, repeat: Infinity, ease: "easeInOut" } : { duration: 0.3 }}
            className="relative shrink-0"
          >
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-cyan-400/40 via-brand-500/40 to-pink-500/40 border border-white/15 grid place-items-center shadow-[0_0_18px_rgba(34,211,238,0.35)]">
              <Lightbulb className="h-4 w-4 text-cyan-100" />
            </div>
            <span className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-emerald-400 ring-2 ring-ink-950">
              <span className="absolute inset-0 rounded-full bg-emerald-400 animate-ping opacity-70" />
            </span>
          </motion.div>
          <div className="flex-1 min-w-0">
            <div className="font-display font-extrabold text-sm tracking-tight flex items-center gap-2">
              Maya · live coach
              <span className={`text-[9px] uppercase tracking-wider font-mono px-1.5 py-0.5 rounded
                ${(thinking || chatBusy) ? "bg-amber-500/15 text-amber-200 border border-amber-500/30" : "bg-emerald-500/15 text-emerald-200 border border-emerald-500/30"}`}>
                {(thinking || chatBusy) ? "● thinking" : "● ready"}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">
              <span className="tabular-nums">{strokes}</span> region{strokes === 1 ? "" : "s"} circled · circle then chat to dig in
            </div>
          </div>
        </div>
      </div>

      {/* CHAT TRANSCRIPT */}
      <div ref={scrollRef} className="relative flex-1 overflow-y-auto px-4 py-4 space-y-3 min-h-[260px] max-h-[420px]">
        <AnimatePresence initial={false}>
          {chat.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
              className={`flex gap-2 ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {m.role === "maya" && (
                <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-cyan-400/40 via-brand-500/40 to-pink-500/40 border border-white/15 grid place-items-center shrink-0 mt-0.5">
                  <Lightbulb className="h-3 w-3 text-cyan-100" />
                </div>
              )}
              <div className={`max-w-[82%] rounded-xl px-3 py-2 text-[12.5px] leading-snug
                ${m.role === "user"
                  ? "bg-pink-500/15 border border-pink-500/30 text-pink-50"
                  : "bg-gradient-to-br from-cyan-500/10 via-brand-500/5 to-transparent border border-cyan-400/25 text-white"}`}>
                {m.region && (
                  <div className="-mx-1 -mt-1 mb-2 rounded-lg overflow-hidden border border-white/10">
                    <img src={m.region} alt="circled region" className="w-full max-h-32 object-cover" />
                  </div>
                )}
                <p className="font-display font-semibold whitespace-pre-wrap">{m.text}</p>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        {(thinking || chatBusy) && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="flex gap-2 justify-start"
          >
            <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-cyan-400/40 via-brand-500/40 to-pink-500/40 border border-white/15 grid place-items-center shrink-0 mt-0.5">
              <Lightbulb className="h-3 w-3 text-cyan-100" />
            </div>
            <div className="rounded-xl px-3 py-2.5 text-[12px] bg-cyan-500/10 border border-cyan-400/25 text-cyan-100 flex items-center gap-2">
              <span className="flex gap-1">
                <motion.span className="h-1.5 w-1.5 rounded-full bg-cyan-300"
                  animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1, repeat: Infinity, delay: 0 }} />
                <motion.span className="h-1.5 w-1.5 rounded-full bg-cyan-300"
                  animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1, repeat: Infinity, delay: 0.15 }} />
                <motion.span className="h-1.5 w-1.5 rounded-full bg-cyan-300"
                  animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1, repeat: Infinity, delay: 0.3 }} />
              </span>
              Maya is looking…
            </div>
          </motion.div>
        )}
      </div>

      {/* CHAT INPUT */}
      <form onSubmit={handleSubmit} className="relative px-4 pt-3 pb-3 border-t border-white/5 bg-black/15">
        {/* quick replies */}
        {chat.length >= 2 && !chatBusy && (
          <div className="flex flex-wrap gap-1.5 mb-2">
            {QUICK.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => onSendChat(q)}
                className="text-[10px] px-2 py-1 rounded-full border border-white/10 bg-white/[0.04] text-slate-300 hover:bg-cyan-500/10 hover:border-cyan-400/30 hover:text-cyan-100 transition"
              >
                {q}
              </button>
            ))}
          </div>
        )}
        <div className={`relative rounded-xl border transition ${chatBusy ? "border-white/5 bg-white/[0.02]" : "border-white/10 bg-black/30 focus-within:border-cyan-400/40"}`}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            disabled={chatBusy}
            rows={2}
            placeholder={chat.length <= 1 ? "Circle something first, then chat with Maya about it…" : "Ask Maya — push back, ask why, get a concrete fix…"}
            className="w-full resize-none bg-transparent px-3 py-2.5 text-[12.5px] leading-snug text-slate-100 placeholder:text-slate-500 focus:outline-none disabled:opacity-50"
          />
          <div className="flex items-center justify-between px-3 pb-2">
            <span className="text-[9px] font-mono text-slate-500">enter ↵ to send · shift+↵ for newline</span>
            <button
              type="submit"
              disabled={!input.trim() || chatBusy}
              className={`text-[11px] font-semibold px-3 py-1 rounded-md border transition flex items-center gap-1
                ${(!input.trim() || chatBusy)
                  ? "border-white/5 bg-white/5 text-slate-500 cursor-not-allowed"
                  : "border-cyan-400/40 bg-gradient-to-r from-cyan-500/20 via-brand-500/20 to-pink-500/20 text-white hover:border-cyan-400/60"}`}
            >
              {chatBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}
              send
            </button>
          </div>
        </div>
      </form>

      {/* Link to the AI improver — no inline brief input here, the dedicated
          improve flow is one click away. */}
      <div className="relative px-5 pt-4 pb-3 border-t border-white/5">
        <button
          onClick={onImproveAll}
          disabled={improving}
          className={`group w-full flex items-center justify-between gap-3 rounded-lg px-3.5 py-2.5 border transition
            ${improving
              ? "border-pink-500/40 bg-pink-500/10 text-pink-100 cursor-not-allowed"
              : "border-white/10 bg-white/5 hover:bg-white/[0.07] hover:border-pink-500/30 text-white"}`}
        >
          <div className="flex items-center gap-2 min-w-0">
            <span className="h-7 w-7 rounded-lg bg-gradient-to-br from-pink-500/40 to-amber-500/30 grid place-items-center border border-white/15 shrink-0">
              {improving
                ? <Loader2 className="h-3.5 w-3.5 animate-spin text-pink-100" />
                : <Wand2 className="h-3.5 w-3.5 text-pink-100" />}
            </span>
            <div className="min-w-0 text-left">
              <div className="font-display text-[13px] font-extrabold tracking-tight truncate">
                {improving ? "Rebuilding with AI…" : hasResult ? "View AI rebuild" : "Open the AI improver"}
              </div>
              <div className="text-[10px] text-slate-400 truncate">
                Send the full creative + ensemble brief to Flux edit.
              </div>
            </div>
          </div>
          <ChevronRight className="h-4 w-4 text-slate-400 shrink-0 transition-transform group-hover:translate-x-0.5" />
        </button>
      </div>

      {/* FOOTER attribution */}
      <div className="relative px-5 py-2.5 border-t border-white/5 flex items-center justify-between gap-2 bg-black/20">
        <div className="text-[10px] text-slate-400 leading-tight">
          Powered by <span className="font-mono font-semibold text-cyan-200">SmolModelV2</span>
          <span className="text-slate-500"> · finetuned for ad creative coaching</span>
        </div>
        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/15 border border-emerald-500/30 text-emerald-200 shrink-0">
          ~250 ms
        </span>
      </div>
    </div>
  );
}

// ---------- BriefBox: shared "guide the AI editor" input ----------
const QUICK_SUGGESTIONS = [
  "Make the CTA bigger and warmer",
  "Use only the suggested palette",
  "Reduce text density",
  "Add a discount badge",
  "Use a real product photo",
  "Cleaner background",
];

function BriefBox({
  value, onChange, onSubmit, busy, hasResult, helperPrefix = "AI editor",
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  busy: boolean;
  hasResult: boolean;
  helperPrefix?: string;
}) {
  const [focused, setFocused] = useState(false);

  function applyChip(chip: string) {
    const next = value.trim() ? `${value.trim()}. ${chip}` : chip;
    onChange(next);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Cmd/Ctrl-Enter to submit — power-user shortcut.
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !busy) {
      e.preventDefault();
      onSubmit();
    }
  }

  return (
    <div className="relative rounded-xl border border-white/10 bg-gradient-to-br from-white/[0.04] to-white/[0.01] p-3.5 overflow-hidden">
      {/* atmospheric glow */}
      <div className="pointer-events-none absolute -top-12 -right-10 h-32 w-32 rounded-full bg-pink-500/15 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-10 -left-12 h-28 w-28 rounded-full bg-brand-500/15 blur-3xl" />

      <div className="relative">
        {/* Header row */}
        <div className="flex items-center justify-between gap-2 mb-2.5">
          <div className="flex items-center gap-2 min-w-0">
            <motion.div
              animate={busy ? { rotate: [0, 8, -6, 4, 0] } : { rotate: 0 }}
              transition={busy ? { duration: 1.4, repeat: Infinity, ease: "easeInOut" } : { duration: 0.3 }}
              className="h-7 w-7 shrink-0 rounded-lg bg-gradient-to-br from-pink-500/40 to-brand-500/40 border border-white/15 grid place-items-center shadow-[0_0_14px_rgba(236,72,153,0.35)]"
            >
              <Sparkles className="h-3.5 w-3.5 text-pink-100" />
            </motion.div>
            <div className="min-w-0">
              <div className="text-[11px] font-display font-extrabold tracking-tight text-white truncate">
                Guide the AI editor
              </div>
              <div className="text-[9px] uppercase tracking-[0.2em] text-pink-200/80">
                optional · steers next generation
              </div>
            </div>
          </div>
          <span className="hidden sm:inline-block text-[9px] font-mono px-1.5 py-0.5 rounded border border-white/10 bg-white/5 text-slate-400 shrink-0">
            ⌘ ↵
          </span>
        </div>

        {/* Textarea with gradient ring on focus */}
        <div className={`relative rounded-lg transition ${focused ? "ring-2 ring-pink-500/40" : "ring-1 ring-white/10"}`}>
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onKeyDown={handleKeyDown}
            disabled={busy}
            placeholder={hasResult
              ? "What to change next: shrink the headline, use only the orange palette, add a -50% badge…"
              : "Tell the AI what matters most: make the CTA pop, simplify background, use a real product photo…"}
            rows={2}
            className="w-full rounded-lg bg-black/30 border border-transparent px-3 py-2.5 text-[12.5px] leading-snug text-slate-100 placeholder:text-slate-500 focus:outline-none disabled:opacity-50 resize-y"
          />
          {/* Live char count */}
          {value.length > 0 && (
            <span className={`absolute bottom-1.5 right-2 text-[9px] font-mono tabular-nums ${value.length > 400 ? "text-rose-300" : "text-slate-500"}`}>
              {value.length}/500
            </span>
          )}
        </div>

        {/* Quick-suggestion chips */}
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {QUICK_SUGGESTIONS.map((chip) => (
            <button
              key={chip}
              type="button"
              onClick={() => applyChip(chip)}
              disabled={busy}
              className="text-[10px] px-2 py-1 rounded-full border border-white/10 bg-white/[0.04] text-slate-300 hover:bg-pink-500/10 hover:border-pink-500/30 hover:text-pink-100 transition disabled:opacity-40 disabled:cursor-not-allowed"
            >
              + {chip}
            </button>
          ))}
        </div>

        {/* Submit row */}
        <div className="mt-3 flex items-center justify-between gap-3">
          <p className="text-[10px] text-slate-500 leading-tight flex-1 min-w-0">
            {busy
              ? `${helperPrefix} is rebuilding…  this usually takes 4–8s.`
              : value.trim()
                ? "Click below to send your brief + the latest creative state."
                : "No brief? We'll re-run the auto-generated instructions."}
          </p>
          <motion.button
            type="button"
            onClick={onSubmit}
            disabled={busy}
            whileHover={!busy ? { y: -2 } : {}}
            whileTap={!busy ? { scale: 0.97 } : {}}
            className={`relative shrink-0 inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-[12px] font-semibold transition overflow-hidden
              ${busy
                ? "bg-pink-500/15 text-pink-100 border border-pink-500/40 cursor-not-allowed"
                : "bg-gradient-to-r from-brand-500 via-pink-500 to-amber-400 text-white shadow-[0_8px_22px_-10px_rgba(236,72,153,0.7)]"}`}
          >
            {!busy && (
              <motion.span
                className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/30 to-transparent"
                animate={{ x: ["-100%", "200%"] }}
                transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut", repeatDelay: 1.4 }}
              />
            )}
            <span className="relative flex items-center gap-1.5">
              {busy
                ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> {hasResult ? "Regenerating" : "Generating"}</>
                : <><Wand2 className="h-3.5 w-3.5" /> {hasResult ? "Regenerate" : "Generate"}</>}
            </span>
          </motion.button>
        </div>
      </div>
    </div>
  );
}

// ---------- Tiny helpers ----------
function useCountUp(target: number, duration = 900) {
  const [v, setV] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setV(target * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return v;
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400 border-b border-white/5 pb-1">
      {children}
    </div>
  );
}

function Field({ label, span = 1, auto = false, children }: {
  label: string; span?: 1 | 2; auto?: boolean; children: React.ReactNode;
}) {
  return (
    <label className={`block ${span === 2 ? "col-span-2" : ""}`}>
      <span className="text-[11px] uppercase tracking-wider text-slate-400 flex items-center gap-1">
        {label}
        {auto && <span className="text-[9px] px-1 py-0.5 rounded bg-brand-500/20 text-brand-200 normal-case tracking-normal">auto</span>}
      </span>
      <div className="mt-1">{children}</div>
    </label>
  );
}

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/40 capitalize"
    >
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

function Toggle({ label, value, onChange, auto = false }: {
  label: string; value: boolean; onChange: (v: boolean) => void; auto?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm transition
        ${value ? "bg-brand-500/15 border-brand-500/40 text-brand-200" : "bg-white/5 border-white/10 text-slate-300"}`}
    >
      <span className="flex items-center gap-1.5">
        {label}
        {auto && <span className="text-[9px] px-1 py-0.5 rounded bg-brand-500/20 text-brand-200">auto</span>}
      </span>
      <span className={`h-4 w-7 rounded-full p-0.5 transition ${value ? "bg-brand-500" : "bg-white/15"}`}>
        <span className={`block h-3 w-3 rounded-full bg-white transition ${value ? "translate-x-3" : "translate-x-0"}`} />
      </span>
    </button>
  );
}

function Range({ value, onChange, min, max, step }: {
  value: number; onChange: (v: number) => void; min: number; max: number; step: number;
}) {
  return (
    <input
      type="range" min={min} max={max} step={step} value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full accent-brand-500"
    />
  );
}
