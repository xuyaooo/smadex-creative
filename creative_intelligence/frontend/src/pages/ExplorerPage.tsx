import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  loadPredictions, loadMetadata, type Prediction, type Metadata,
  STATUS_COLORS, VERTICAL_COLORS, actionFromHealth, actionColor,
} from "../lib/data";
import { Filter, X, Search, ChevronLeft, ChevronRight } from "lucide-react";

const PAGE = 24;

function StatusBadge({ s }: { s: keyof typeof STATUS_COLORS }) {
  const c = STATUS_COLORS[s];
  return (
    <span className={`px-2 py-0.5 rounded-md text-[11px] font-medium capitalize ring-1 ${c.fg} ${c.bg} ${c.ring}`}>
      {s.replace("_", " ")}
    </span>
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

  const pageItems = filtered.slice(page * PAGE, (page + 1) * PAGE);
  const totalPages = Math.ceil(filtered.length / PAGE);

  if (!preds || !meta) {
    return (
      <main className="pt-28 container-narrow"><p className="text-slate-400">loading…</p></main>
    );
  }

  return (
    <main className="pt-28 pb-16">
      <div className="container-narrow">
        <motion.div
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        >
          <span className="stat-pill"><Filter className="h-3 w-3" /> creative explorer</span>
          <h1 className="mt-3 text-3xl sm:text-4xl font-bold tracking-tight">Per-creative predictions</h1>
          <p className="mt-2 text-slate-400 text-sm max-w-xl">
            All 1,076 creatives with their model output: predicted status, class probabilities,
            health score, and recommended action.
          </p>
        </motion.div>

        {/* Filter bar */}
        <div className="mt-8 card-surface p-4 flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[180px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              placeholder="search id, vertical, format…"
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(0); }}
              className="w-full rounded-lg bg-white/5 border border-white/10 pl-9 pr-3 py-2 text-sm placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500/40"
            />
          </div>

          <select
            value={splitFilter}
            onChange={(e) => { setSplitFilter(e.target.value as any); setPage(0); }}
            className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
          >
            <option value="all">all splits</option>
            <option value="train">train</option>
            <option value="val">val</option>
            <option value="test">test</option>
          </select>

          <select
            value={verticalFilter ?? ""}
            onChange={(e) => { setVerticalFilter(e.target.value || null); setPage(0); }}
            className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
          >
            <option value="">all verticals</option>
            {meta.verticals.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>

          <select
            value={statusFilter ?? ""}
            onChange={(e) => { setStatusFilter(e.target.value || null); setPage(0); }}
            className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
          >
            <option value="">all statuses</option>
            {meta.class_names.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>

          <select
            value={actionFilter ?? ""}
            onChange={(e) => { setActionFilter(e.target.value || null); setPage(0); }}
            className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
          >
            <option value="">all actions</option>
            {["Scale", "Maintain", "Watch", "Pause/Pivot"].map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>

          {(verticalFilter || statusFilter || actionFilter || search) && (
            <button
              onClick={() => { setVerticalFilter(null); setStatusFilter(null); setActionFilter(null); setSearch(""); setPage(0); }}
              className="text-xs text-slate-400 hover:text-white inline-flex items-center gap-1"
            >
              <X className="h-3.5 w-3.5" /> clear
            </button>
          )}
        </div>

        <div className="mt-4 text-xs text-slate-400">
          showing {filtered.length === 0 ? 0 : page * PAGE + 1}–
          {Math.min((page + 1) * PAGE, filtered.length)} of {filtered.length}
        </div>

        {/* Grid */}
        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {pageItems.map((p) => {
            const action = actionFromHealth(p.health_score);
            const ac = actionColor(action);
            const correct = p.pred_status === p.true_status;
            return (
              <button
                key={p.creative_id}
                onClick={() => setSelected(p)}
                className={`card-surface p-4 text-left transition hover:bg-white/[0.06] hover:-translate-y-0.5
                            ${correct ? "" : "ring-1 ring-rose-500/30"}`}
              >
                <div className="flex items-start justify-between">
                  <div className="text-xs text-slate-400">#{p.creative_id}</div>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${ac.fg} ${ac.bg}`}>
                    {action}
                  </span>
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <span
                    className="inline-flex h-2 w-2 rounded-full"
                    style={{ backgroundColor: VERTICAL_COLORS[p.vertical] || "#94a3b8" }}
                  />
                  <span className="text-xs font-medium capitalize text-slate-200">{p.vertical}</span>
                  <span className="text-[10px] text-slate-500">·</span>
                  <span className="text-[10px] text-slate-400">{p.format}</span>
                </div>
                <div className="mt-3 flex items-end justify-between">
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">health</div>
                    <div className="text-2xl font-bold tracking-tight">
                      {p.health_score.toFixed(0)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">predicted</div>
                    <StatusBadge s={p.pred_status as any} />
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="mt-6 flex justify-center items-center gap-3">
            <button
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
              className="rounded-lg border border-white/10 bg-white/5 p-2 disabled:opacity-30 hover:bg-white/10"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="text-sm text-slate-300">
              page {page + 1} of {totalPages}
            </span>
            <button
              onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
              disabled={page === totalPages - 1}
              className="rounded-lg border border-white/10 bg-white/5 p-2 disabled:opacity-30 hover:bg-white/10"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* Detail drawer */}
        {selected && (
          <div
            className="fixed inset-0 z-50 bg-ink-950/80 backdrop-blur-sm flex items-center justify-center p-4"
            onClick={() => setSelected(null)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.15 }}
              onClick={(e) => e.stopPropagation()}
              className="card-surface max-w-2xl w-full p-7 relative"
            >
              <button
                onClick={() => setSelected(null)}
                className="absolute top-4 right-4 text-slate-400 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
              <div className="text-xs text-slate-400">creative #{selected.creative_id}</div>
              <div className="mt-1 flex items-center gap-2">
                <h3 className="text-2xl font-bold capitalize">{selected.vertical}</h3>
                <span className="text-sm text-slate-400">· {selected.format}</span>
              </div>

              <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-slate-400">health score</div>
                  <div className="text-2xl font-bold">{selected.health_score.toFixed(1)}</div>
                </div>
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-slate-400">action</div>
                  <div className={`text-base font-bold ${actionColor(actionFromHealth(selected.health_score)).fg}`}>
                    {actionFromHealth(selected.health_score)}
                  </div>
                </div>
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-slate-400">true status</div>
                  <StatusBadge s={selected.true_status as any} />
                </div>
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-slate-400">predicted</div>
                  <StatusBadge s={selected.pred_status as any} />
                </div>
              </div>

              <div className="mt-5">
                <div className="text-xs text-slate-400 mb-2">class probabilities</div>
                <div className="space-y-2">
                  {([
                    ["top_performer", selected.p_top],
                    ["stable",        selected.p_stable],
                    ["fatigued",      selected.p_fatigued],
                    ["underperformer",selected.p_under],
                  ] as [keyof typeof STATUS_COLORS, number][]).map(([name, p]) => (
                    <div key={name}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className={`${STATUS_COLORS[name].fg} capitalize`}>{name.replace("_"," ")}</span>
                        <span className="text-slate-400 tabular-nums">{(p * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-white/5">
                        <div
                          className={`h-full rounded-full ${STATUS_COLORS[name].bg.replace("/15","/70")}`}
                          style={{ width: `${p * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-slate-300">
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-slate-500">early CTR</div>
                  <div className="font-semibold">{(selected.early_ctr * 100).toFixed(2)}%</div>
                </div>
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-slate-500">early imps</div>
                  <div className="font-semibold">{selected.early_imp.toLocaleString()}</div>
                </div>
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-slate-500">early spend</div>
                  <div className="font-semibold">${selected.early_spend.toLocaleString()}</div>
                </div>
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-slate-500">early revenue</div>
                  <div className="font-semibold">${selected.early_revenue.toLocaleString()}</div>
                </div>
              </div>

              <div className="mt-5 grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-slate-500 mb-0.5">creative attributes</div>
                  <div>theme: <span className="font-medium capitalize">{selected.theme}</span></div>
                  <div>hook: <span className="font-medium capitalize">{selected.hook_type}</span></div>
                  <div>color: <span className="font-medium capitalize">{selected.dominant_color}</span></div>
                  <div>tone: <span className="font-medium capitalize">{selected.emotional_tone}</span></div>
                </div>
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-slate-500 mb-0.5">flags</div>
                  <div>{selected.has_price ? "✓" : "✗"} price</div>
                  <div>{selected.has_discount_badge ? "✓" : "✗"} discount badge</div>
                  <div>{selected.has_gameplay ? "✓" : "✗"} gameplay</div>
                  <div>{selected.has_ugc_style ? "✓" : "✗"} UGC style</div>
                </div>
              </div>

              <div className="mt-5 grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-slate-500 mb-0.5">true fatigue</div>
                  <div className="capitalize font-medium">{selected.true_fatigue}</div>
                </div>
                <div className="rounded-lg bg-white/5 p-3">
                  <div className="text-slate-500 mb-0.5">predicted fatigue</div>
                  <div className="capitalize font-medium">{selected.pred_fatigue}</div>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </main>
  );
}
