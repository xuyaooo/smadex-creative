import { Link, NavLink } from "react-router-dom";
import Logo from "./Logo";
import { Github } from "lucide-react";

const items = [
  { to: "/", label: "Home" },
  { to: "/stats", label: "Stats" },
  { to: "/explorer", label: "Explorer" },
  { to: "/predict", label: "Predict" },
];

export default function Navbar() {
  return (
    <header className="fixed inset-x-0 top-0 z-40 backdrop-blur-md bg-ink-950/60 border-b border-white/5">
      <nav className="container-narrow flex h-16 items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5">
          <Logo size={28} />
          <span className="text-base font-bold tracking-tight">
            Creative<span className="text-brand-400">Intelligence</span>
          </span>
        </Link>

        <div className="hidden md:flex items-center gap-8">
          {items.map((it) => (
            <NavLink
              key={it.to}
              to={it.to}
              end={it.to === "/"}
              className={({ isActive }) =>
                `nav-link ${isActive ? "nav-link-active" : ""}`
              }
            >
              {it.label}
            </NavLink>
          ))}
        </div>

        <a
          href="https://github.com/xuyaooo/smadex-creative"
          target="_blank"
          rel="noreferrer"
          className="hidden sm:inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-white/10 transition"
        >
          <Github className="h-3.5 w-3.5" />
          GitHub
        </a>
      </nav>
    </header>
  );
}
