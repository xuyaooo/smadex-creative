import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  loadPredictions, loadMetadata, type Prediction, type Metadata,
  STATUS_COLORS, VERTICAL_COLORS, actionFromHealth, actionColor,
} from "../lib/data";
import { Filter, X, Search, ChevronLeft, ChevronRight, ImageOff, Target, Sparkles, LayoutGrid, Rows3 } from "lucide-react";

const PAGE = 24;

const ASSET = (cid: number) => `/assets/creative_${cid}.png`;

function StatusBadge({ s }: { s: keyof typeof STATUS_COLORS }) {
  const c = STATUS_COLORS[s];
  return (
    <span className={`px-2 py-0.5 rounded-md text-[11px] font-medium capitalize ring-1 ${c.fg} ${c.bg} ${c.ring}`}>
      {s.replace("_", " ")}
    </span>
  );
}

function CreativeImage({ cid, className = "" }: { cid: number; className?: string }) {
  const [error, setError] = useState(false);
  if (error) {
    return (
      <div className={`grid place-items-center bg-gradient-to-br from-brand-500/15 via-pink-500/10 to-cyan-500/10 ${className}`}>
        <ImageOff className="h-6 w-6 text-slate-500" />
      </div>
    );
  }
  return (
    <img
      src={ASSET(cid)} alt={`creative ${cid}`}
      onError={() => setError(true)}
      className={`object-cover ${className}`}
      loading="lazy"
    />
  );
}

export default function ExplorerPage() {
  const [preds, setPreds] = useState<Prediction[] | null>(null);
  const [meta, setMeta] = useState<Metadata | null>(null);

  const [search, setSearch] = useState("");
  const [splitFilter, setSplitFilter] = useState<"all" | "train" | "val" | "test">("test");
  const [verticalFilter, setVerticalFilter] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [actionFilter, setActionFilter] = useState<string | null>(null);
  const [view, setView] = useState<"gallery" | "compact">("gallery");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Prediction | null>(null);

  useEffect(() => {
    Promise.all([loadPredictions(), loadMetadata()]).then(([p, m]) => {
      setPreds(p); setMeta(m);
    });
  }, []);

  const filtered = useMemo(() => {
    if (!preds) return [];
    const term = search.trim().toLowerCase();
    return preds.filter((p) => {
      if (splitFilter !== "all" && p.split !== splitFilter) return false;
      if (verticalFilter && p.vertical !== verticalFilter) return false;
      if (statusFilter && p.true_status !== statusFilter) return false;
      if (actionFilter && actionFromHealth(p.health_score) !== actionFilter) return false;
      if (term && !String(p.creative_id).includes(term) &&
                  !p.vertical.toLowerCase().includes(term) &&
                  !p.format.toLowerCase().includes(term)) return false;
      return true;
    });
  }, [preds, search, splitFilter, verticalFilter, statusFilter, actionFilter]);

  // Counts for the filter chips so we don't have to re-compute
  const verticalCounts = useMemo(() => {
    const m: Record<string, number> = {};
    if (!preds) return m;
    for (const p of preds.filter((p) => splitFilter === "all" || p.split === splitFilter)) {
      m[p.vertical] = (m[p.vertical] ?? 0) + 1;
    }
    return m;
  }, [preds, splitFilter]);

  const statusCounts = useMemo(() => {
    const m: Record<string, number> = {};
    if (!preds) return m;
    for (const p of preds.filter((p) => splitFilter === "all" || p.split === splitFilter)) {
      m[p.true_status] = (m[p.true_status] ?? 0) + 1;
    }
    return m;
  }, [preds, splitFilter]);

  const pageItems = filtered.slice(page * PAGE, (page + 1) * PAGE);
  const totalPages = Math.ceil(filtered.length / PAGE);

  function clearAll() {
    setVerticalFilter(null); setStatusFilter(null); setActionFilter(null); setSearch(""); setPage(0);
  }
  const hasFilters = !!(verticalFilter || statusFilter || actionFilter || search);

  if (!preds || !meta) {
    return <main className="pt-28 container-narrow"><p className="text-slate-400">loading…</p></main>;
  }

  return (
    <main className="pt-24 pb-20 relative">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-24 -left-20 h-[28rem] w-[28rem] rounded-full bg-brand-500/15 blur-3xl animate-blob" />
        <div className="absolute top-40 -right-24 h-[22rem] w-[22rem] rounded-full bg-pink-500/10 blur-3xl animate-blob [animation-delay:-6s]" />
      </div>

      <div className="container-narrow">
        <motion.div
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        >
          <span className="stat-pill"><Filter className="h-3 w-3" /> creative explorer</span>
          <h1 className="mt-3 text-3xl sm:text-4xl font-bold tracking-tight bg-gradient-to-br from-white via-white to-white/60 bg-clip-text text-transparent">
            Browse all 1,076 creatives
          </h1>
          <p className="mt-2 text-slate-400 text-sm max-w-xl">
            Filter by vertical, status, or action. Click any creative for the full per-image breakdown.
          </p>
        </motion.div>

        {/* Search + view toggle */}
        <motion.div
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.05 }}
          className="mt-7 flex flex-wrap items-center gap-3"
        >
          <div className="relative flex-1 min-w-[220px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              placeholder="search id, vertical, format…"
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(0); }}
              className="w-full rounded-xl bg-white/5 border border-white/10 pl-10 pr-3 py-2.5 text-sm placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500/40"
            />
          </div>
          <div className="flex items-center rounded-xl border border-white/10 bg-white/5 p-1 text-xs">
            {([
              ["gallery", LayoutGrid],
              ["compact", Rows3],
            ] as const).map(([v, Icon]) => (
              <button
                key={v}
                onClick={() => setView(v as any)}
                className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition ${view === v ? "bg-brand-500/20 text-brand-100" : "text-slate-400 hover:text-white"}`}
              >
                <Icon className="h-3.5 w-3.5" /> {v}
              </button>
            ))}
          </div>
          {hasFilters && (
            <button onClick={clearAll} className="text-xs text-slate-400 hover:text-white inline-flex items-center gap-1">
              <X className="h-3.5 w-3.5" /> clear
            </button>
          )}
        </motion.div>

        {/* Filter chips */}
        <motion.div
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}
          className="mt-5 space-y-3"
        >
          <ChipRow label="split">
            {(["all", "train", "val", "test"] as const).map((s) => (
              <Chip key={s} active={splitFilter === s} onClick={() => { setSplitFilter(s); setPage(0); }}>
                {s}
              </Chip>
            ))}
          </ChipRow>

          <ChipRow label="vertical">
            <Chip active={verticalFilter === null} onClick={() => { setVerticalFilter(null); setPage(0); }}>all</Chip>
            {meta.verticals.map((v) => (
              <Chip
                key={v}
                active={verticalFilter === v}
                onClick={() => { setVerticalFilter(verticalFilter === v ? null : v); setPage(0); }}
                dotColor={VERTICAL_COLORS[v]}
                count={verticalCounts[v]}
              >
                {v}
              </Chip>
            ))}
          </ChipRow>

          <ChipRow label="status">
            <Chip active={statusFilter === null} onClick={() => { setStatusFilter(null); setPage(0); }}>all</Chip>
            {meta.class_names.map((c) => (
              <Chip
                key={c}
                active={statusFilter === c}
                onClick={() => { setStatusFilter(statusFilter === c ? null : c); setPage(0); }}
                tone={c as any}
                count={statusCounts[c]}
              >
                {c.replace("_", " ")}
              </Chip>
            ))}
          </ChipRow>

          <ChipRow label="action">
            <Chip active={actionFilter === null} onClick={() => { setActionFilter(null); setPage(0); }}>all</Chip>
            {(["Scale", "Maintain", "Watch", "Pause/Pivot"] as const).map((a) => (
              <Chip
                key={a}
                active={actionFilter === a}
                onClick={() => { setActionFilter(actionFilter === a ? null : a); setPage(0); }}
                actionTone={a}
              >
                {a}
              </Chip>
            ))}
          </ChipRow>
        </motion.div>

        <div className="mt-6 flex items-center justify-between">
          <div className="text-xs text-slate-400">
            showing <span className="text-white font-semibold">{filtered.length === 0 ? 0 : page * PAGE + 1}–{Math.min((page + 1) * PAGE, filtered.length)}</span> of <span className="text-white font-semibold">{filtered.length}</span>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(Math.max(0, page - 1))}
                disabled={page === 0}
                className="rounded-lg border border-white/10 bg-white/5 p-1.5 disabled:opacity-30 hover:bg-white/10"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <span className="text-xs text-slate-400 tabular-nums">{page + 1} / {totalPages}</span>
              <button
                onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
                disabled={page === totalPages - 1}
                className="rounded-lg border border-white/10 bg-white/5 p-1.5 disabled:opacity-30 hover:bg-white/10"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>

        {/* Grid */}
        {view === "gallery" ? (
          <motion.div
            key={`gallery-${page}-${verticalFilter}-${statusFilter}-${actionFilter}-${splitFilter}-${search}`}
            initial="hidden" animate="show"
            variants={{ show: { transition: { staggerChildren: 0.025 } } }}
            className="mt-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3"
          >
            {pageItems.map((p) => <GalleryCard key={p.creative_id} p={p} onSelect={setSelected} />)}
          </motion.div>
        ) : (
          <div className="mt-4 space-y-2">
            {pageItems.map((p) => <CompactRow key={p.creative_id} p={p} onSelect={setSelected} />)}
          </div>
        )}

        {pageItems.length === 0 && (
          <div className="mt-12 text-center text-slate-400">
            <Filter className="mx-auto h-8 w-8 mb-2 opacity-40" />
            <p className="text-sm">No creatives match these filters.</p>
            <button onClick={clearAll} className="mt-3 btn-ghost text-xs">clear filters</button>
          </div>
        )}
      </div>

      {/* Detail drawer */}
      <AnimatePresence>
        {selected && <DetailDrawer p={selected} onClose={() => setSelected(null)} />}
      </AnimatePresence>
    </main>
  );
}

// ---------- Gallery card ----------
function GalleryCard({ p, onSelect }: { p: Prediction; onSelect: (p: Prediction) => void }) {
  const action = actionFromHealth(p.health_score);
  const ac = actionColor(action);
  const correct = p.pred_status === p.true_status;

  return (
    <motion.button
      variants={{ hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0 } }}
      whileHover={{ y: -4 }}
      transition={{ type: "spring", stiffness: 300, damping: 25 }}
      onClick={() => onSelect(p)}
      className={`card-surface text-left overflow-hidden group ${correct ? "" : "ring-1 ring-rose-500/30"}`}
    >
      <div className="relative aspect-square bg-black/40 overflow-hidden">
        <CreativeImage cid={p.creative_id} className="absolute inset-0 h-full w-full transition-transform duration-700 group-hover:scale-105" />
        <div className="absolute inset-x-0 top-0 p-2 flex items-center justify-between">
          <span className="text-[10px] text-white/85 font-bold bg-black/60 backdrop-blur rounded px-1.5 py-0.5">#{p.creative_id}</span>
          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${ac.fg} ${ac.bg} backdrop-blur`}>
            {action}
          </span>
        </div>
        <div className="absolute inset-x-0 bottom-0 p-2 bg-gradient-to-t from-black/80 to-transparent">
          <div className="flex items-center gap-1.5 text-[10px] text-white">
            <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ backgroundColor: VERTICAL_COLORS[p.vertical] || "#94a3b8" }} />
            <span className="capitalize font-medium">{p.vertical}</span>
            <span className="text-white/60">·</span>
            <span className="text-white/80">{p.format}</span>
          </div>
        </div>
      </div>
      <div className="p-3 flex items-center justify-between gap-2">
        <div>
          <div className="text-[9px] uppercase tracking-wider text-slate-500">health</div>
          <div className="flex items-baseline gap-1">
            <span className="text-xl font-bold tabular-nums leading-none">{p.health_score.toFixed(0)}</span>
            <span className="text-[10px] text-slate-500">/100</span>
          </div>
        </div>
        <div className="flex-1 ml-2">
          <div className="h-1 rounded-full bg-white/5 overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-brand-500 to-pink-500"
              initial={{ width: 0 }} animate={{ width: `${p.health_score}%` }} transition={{ duration: 0.8, ease: "easeOut" }}
            />
          </div>
        </div>
        <StatusBadge s={p.pred_status as any} />
      </div>
    </motion.button>
  );
}

// ---------- Compact row ----------
function CompactRow({ p, onSelect }: { p: Prediction; onSelect: (p: Prediction) => void }) {
  const action = actionFromHealth(p.health_score);
  const ac = actionColor(action);
  return (
    <button
      onClick={() => onSelect(p)}
      className="card-surface w-full p-3 flex items-center gap-4 text-left hover:bg-white/[0.06] transition group"
    >
      <CreativeImage cid={p.creative_id} className="h-14 w-14 rounded-lg shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">#{p.creative_id}</span>
          <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ backgroundColor: VERTICAL_COLORS[p.vertical] || "#94a3b8" }} />
          <span className="text-sm font-medium capitalize">{p.vertical}</span>
          <span className="text-[10px] text-slate-500">· {p.format}</span>
        </div>
        <div className="mt-1 h-1 rounded-full bg-white/5 max-w-[260px]">
          <div className="h-full rounded-full bg-gradient-to-r from-brand-500 to-pink-500" style={{ width: `${p.health_score}%` }} />
        </div>
      </div>
      <div className="text-right shrink-0">
        <div className="text-xl font-bold tabular-nums">{p.health_score.toFixed(0)}</div>
        <div className="text-[9px] uppercase tracking-wider text-slate-500">health</div>
      </div>
      <div className="flex flex-col gap-1 items-end shrink-0">
        <StatusBadge s={p.pred_status as any} />
        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${ac.fg} ${ac.bg}`}>{action}</span>
      </div>
    </button>
  );
}

// ---------- Filter chip primitives ----------
function ChipRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500 pt-1.5 w-16 shrink-0">{label}</div>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

function Chip({
  active = false, onClick, children, dotColor, tone, actionTone, count,
}: {
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
  dotColor?: string;
  tone?: "top_performer" | "stable" | "fatigued" | "underperformer";
  actionTone?: "Scale" | "Maintain" | "Watch" | "Pause/Pivot";
  count?: number;
}) {
  const toneFg =
    tone ? STATUS_COLORS[tone].fg :
    actionTone ? actionColor(actionTone).fg :
    "text-slate-200";
  const toneBg =
    tone ? STATUS_COLORS[tone].bg :
    actionTone ? actionColor(actionTone).bg :
    "bg-white/5";
  return (
    <button
      onClick={onClick}
      className={`group inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs capitalize transition
        ${active
          ? `${toneFg} ${toneBg} border-white/30 ring-1 ring-white/20 shadow-sm`
          : "text-slate-300 bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20"}`}
    >
      {dotColor && <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: dotColor }} />}
      <span>{children}</span>
      {count !== undefined && (
        <span className={`text-[10px] tabular-nums ${active ? "text-current/70" : "text-slate-500"}`}>{count}</span>
      )}
    </button>
  );
}

// ---------- Detail drawer ----------
function DetailDrawer({ p, onClose }: { p: Prediction; onClose: () => void }) {
  const action = actionFromHealth(p.health_score);
  const ac = actionColor(action);
  const correct = p.pred_status === p.true_status;

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      onClick={onClose}
      className="fixed inset-0 z-50 bg-ink-950/85 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto"
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96, y: 20 }}
        transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
        onClick={(e) => e.stopPropagation()}
        className="card-surface max-w-4xl w-full overflow-hidden relative"
      >
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 h-8 w-8 grid place-items-center rounded-full bg-black/50 text-slate-200 hover:bg-black/80 hover:text-white"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="grid grid-cols-1 md:grid-cols-2">
          {/* Big image */}
          <div className="relative bg-black/60 aspect-square md:aspect-auto md:min-h-[460px]">
            <CreativeImage cid={p.creative_id} className="absolute inset-0 h-full w-full" />
            <div className="absolute inset-x-0 top-0 p-3 flex items-center justify-between">
              <span className="text-xs font-bold bg-black/70 backdrop-blur rounded px-2 py-1 text-white">#{p.creative_id}</span>
              <span className={`text-xs font-bold px-2.5 py-1 rounded-full backdrop-blur ${ac.fg} ${ac.bg}`}>
                <Target className="inline h-3 w-3 mr-1" /> {action}
              </span>
            </div>
            <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/95 to-transparent">
              <div className="flex items-center gap-2 text-sm">
                <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: VERTICAL_COLORS[p.vertical] || "#94a3b8" }} />
                <span className="font-semibold text-white capitalize">{p.vertical}</span>
                <span className="text-white/60">·</span>
                <span className="text-white/80">{p.format}</span>
              </div>
              <div className="mt-1 text-[11px] text-white/60 uppercase tracking-wider">{p.split} split</div>
            </div>
          </div>

          {/* Right: details */}
          <div className="p-6 max-h-[80vh] overflow-y-auto">
            {/* Health hero */}
            <div className="rounded-xl bg-gradient-to-br from-brand-500/20 to-pink-500/10 border border-white/10 p-4 relative overflow-hidden">
              <div className="absolute -top-12 -right-8 h-32 w-32 rounded-full bg-brand-500/30 blur-2xl pointer-events-none" />
              <div className="relative flex items-end justify-between gap-3">
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-300">creative health score</div>
                  <div className="mt-1 flex items-baseline gap-1">
                    <span className="text-5xl font-extrabold tabular-nums leading-none">{p.health_score.toFixed(0)}</span>
                    <span className="text-slate-400 text-sm">/100</span>
                  </div>
                </div>
                <div className={`text-[11px] font-bold px-2.5 py-1 rounded-full ${correct ? "text-emerald-300 bg-emerald-500/15 border border-emerald-500/30" : "text-rose-300 bg-rose-500/15 border border-rose-500/30"}`}>
                  {correct ? <><Sparkles className="inline h-3 w-3 mr-0.5" /> correct</> : "miss"}
                </div>
              </div>
              <div className="mt-3 h-1.5 rounded-full bg-white/10 overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-brand-500 to-pink-500"
                  initial={{ width: 0 }} animate={{ width: `${p.health_score}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </div>
            </div>

            {/* Status */}
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded-lg bg-white/5 p-3">
                <div className="text-[10px] uppercase tracking-wider text-slate-400">true status</div>
                <div className="mt-1"><StatusBadge s={p.true_status as any} /></div>
              </div>
              <div className="rounded-lg bg-white/5 p-3">
                <div className="text-[10px] uppercase tracking-wider text-slate-400">predicted</div>
                <div className="mt-1"><StatusBadge s={p.pred_status as any} /></div>
              </div>
            </div>

            {/* Probabilities */}
            <div className="mt-5">
              <div className="text-xs text-slate-400 mb-2 uppercase tracking-wider">class probabilities</div>
              <div className="space-y-2">
                {([
                  ["top_performer", p.p_top],
                  ["stable",        p.p_stable],
                  ["fatigued",      p.p_fatigued],
                  ["underperformer",p.p_under],
                ] as [keyof typeof STATUS_COLORS, number][]).map(([name, v], i) => (
                  <div key={name}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className={`${STATUS_COLORS[name].fg} capitalize`}>{name.replace("_"," ")}</span>
                      <span className="text-slate-400 tabular-nums">{(v * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/5">
                      <motion.div
                        className={`h-full rounded-full ${STATUS_COLORS[name].bg.replace("/15","/70")}`}
                        initial={{ width: 0 }} animate={{ width: `${v * 100}%` }}
                        transition={{ duration: 0.6, delay: i * 0.06 }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Performance facts */}
            <div className="mt-5 grid grid-cols-2 gap-2 text-xs">
              <Stat label="early CTR" value={`${(p.early_ctr * 100).toFixed(2)}%`} />
              <Stat label="early imps" value={p.early_imp.toLocaleString()} />
              <Stat label="early spend" value={`$${p.early_spend.toLocaleString()}`} />
              <Stat label="early revenue" value={`$${p.early_revenue.toLocaleString()}`} />
            </div>

            {/* Attributes + flags */}
            <div className="mt-5 grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-lg bg-white/5 p-3 space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-1">attributes</div>
                <Attr k="theme" v={p.theme} />
                <Attr k="hook"  v={p.hook_type} />
                <Attr k="color" v={p.dominant_color} />
                <Attr k="tone"  v={p.emotional_tone} />
              </div>
              <div className="rounded-lg bg-white/5 p-3 space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-1">flags</div>
                <Flag on={!!p.has_price} label="price" />
                <Flag on={!!p.has_discount_badge} label="discount badge" />
                <Flag on={!!p.has_gameplay} label="gameplay" />
                <Flag on={!!p.has_ugc_style} label="UGC style" />
              </div>
            </div>

            {/* Fatigue */}
            <div className="mt-5 grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-lg bg-white/5 p-3">
                <div className="text-[10px] uppercase tracking-wider text-slate-400">true fatigue</div>
                <div className="capitalize font-medium mt-0.5">{p.true_fatigue}</div>
              </div>
              <div className="rounded-lg bg-white/5 p-3">
                <div className="text-[10px] uppercase tracking-wider text-slate-400">predicted fatigue</div>
                <div className="capitalize font-medium mt-0.5">{p.pred_fatigue}</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-white/5 p-2.5">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="font-semibold text-slate-200 mt-0.5">{value}</div>
    </div>
  );
}

function Attr({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-slate-500">{k}</span>
      <span className="font-medium capitalize text-slate-200">{v.replace("_", " ")}</span>
    </div>
  );
}

function Flag({ on, label }: { on: boolean; label: string }) {
  return (
    <div className={`flex items-center gap-1.5 ${on ? "text-emerald-300" : "text-slate-500"}`}>
      <span className="font-bold">{on ? "✓" : "✗"}</span>
      <span className="capitalize">{label}</span>
    </div>
  );
}
