/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        display: ["\"Bricolage Grotesque\"", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["\"JetBrains Mono\"", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        brand: {
          50: "#eef2ff", 100: "#e0e7ff", 200: "#c7d2fe", 300: "#a5b4fc",
          400: "#818cf8", 500: "#6366f1", 600: "#4f46e5", 700: "#4338ca",
          800: "#3730a3", 900: "#312e81",
        },
        ink: {
          950: "#0b1020", 900: "#10162a", 800: "#161d35",
        },
      },
      backgroundImage: {
        "grid-faint": "radial-gradient(circle at 1px 1px, rgba(255,255,255,0.06) 1px, transparent 0)",
      },
      backgroundSize: { "grid-24": "24px 24px" },
      keyframes: {
        blob: {
          "0%, 100%": { transform: "translate(0,0) scale(1)" },
          "33%":      { transform: "translate(30px,-50px) scale(1.1)" },
          "66%":      { transform: "translate(-20px,20px) scale(0.95)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        blob: "blob 18s infinite ease-in-out",
        shimmer: "shimmer 2.4s linear infinite",
      },
      boxShadow: {
        soft: "0 30px 60px -20px rgba(8,12,30,.6), 0 8px 20px -10px rgba(8,12,30,.45)",
        glow: "0 8px 30px rgba(99,102,241,.45)",
      },
    },
  },
  plugins: [],
};
