/**
 * Smadex Creative Intelligence mark.
 *
 * Concept — a four-point spark fused with an upward growth diagonal:
 *  • The square base is a layered indigo→fuchsia→amber gradient with a
 *    subtle inner highlight in the top-left corner (gives it depth).
 *  • A bold diagonal "growth" stroke goes from bottom-left to top-right.
 *  • A soft 4-point sparkle sits at the top-right tip — the "AI" part.
 *  • A small accent dot lives at the bottom-left start — the "before" state.
 */
export default function Logo({ size = 32, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" className={className} aria-label="Smadex Creative Intelligence">
      <defs>
        <linearGradient id="logo-base" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
          <stop offset="0%"   stopColor="#6366f1" />
          <stop offset="55%"  stopColor="#ec4899" />
          <stop offset="100%" stopColor="#f59e0b" />
        </linearGradient>
        <radialGradient id="logo-highlight" cx="20%" cy="18%" r="60%" fx="18%" fy="14%">
          <stop offset="0%"  stopColor="white" stopOpacity="0.35" />
          <stop offset="55%" stopColor="white" stopOpacity="0.05" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="logo-stroke" x1="14" y1="50" x2="50" y2="14" gradientUnits="userSpaceOnUse">
          <stop offset="0%"  stopColor="#ffffff" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#ffffff" stopOpacity="1" />
        </linearGradient>
        <filter id="logo-glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="0.9" />
        </filter>
      </defs>

      {/* base tile */}
      <rect x="0" y="0" width="64" height="64" rx="16" fill="url(#logo-base)" />
      {/* inner top-left highlight */}
      <rect x="0" y="0" width="64" height="64" rx="16" fill="url(#logo-highlight)" />
      {/* hairline border for definition */}
      <rect x="0.5" y="0.5" width="63" height="63" rx="15.5" fill="none" stroke="white" strokeOpacity="0.18" />

      {/* growth diagonal — soft glow underneath, sharp white on top */}
      <g>
        <path
          d="M 14 50 L 50 14"
          stroke="white" strokeOpacity="0.45" strokeWidth="9" strokeLinecap="round" filter="url(#logo-glow)"
        />
        <path
          d="M 14 50 L 50 14"
          stroke="url(#logo-stroke)" strokeWidth="5" strokeLinecap="round"
        />
      </g>

      {/* origin dot at bottom-left */}
      <circle cx="14" cy="50" r="3" fill="white" fillOpacity="0.95" />

      {/* sparkle at top-right peak — 4-point star */}
      <g transform="translate(50 14)">
        <path
          d="M 0 -8 L 1.4 -1.4 L 8 0 L 1.4 1.4 L 0 8 L -1.4 1.4 L -8 0 L -1.4 -1.4 Z"
          fill="white"
        />
        <circle r="2.2" fill="white" />
      </g>
    </svg>
  );
}
