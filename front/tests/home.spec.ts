import { test, expect } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SHOTS = path.join(__dirname, "screenshots");
fs.mkdirSync(SHOTS, { recursive: true });

const errors: { url: string; msg: string }[] = [];

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (e) => errors.push({ url: page.url(), msg: e.message }));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push({ url: page.url(), msg: msg.text() });
  });
});

test("home page renders and scrolls without runtime errors", async ({ page }) => {
  await page.goto("/", { waitUntil: "networkidle" });

  // Hero is visible
  await expect(page.getByRole("heading", { name: /not reach what you expected/i })).toBeVisible();
  await expect(page.getByRole("link", { name: /start now/i }).first()).toBeVisible();

  // Take a hero shot
  await page.screenshot({ path: path.join(SHOTS, "01-hero.png"), fullPage: false });

  // Scroll through the cinematic story stage by stage and screenshot
  const layout = await page.evaluate(() => {
    const story = document.getElementById("story");
    const how = document.getElementById("how-it-works");
    const total = document.body.scrollHeight;
    const vh = window.innerHeight;
    const sBox = story?.getBoundingClientRect();
    const hBox = how?.getBoundingClientRect();
    const sTop = (sBox?.top ?? 0) + window.scrollY;
    const sHeight = sBox?.height ?? 0;
    const hTop = (hBox?.top ?? 0) + window.scrollY;
    return { total, vh, sTop, sHeight, hTop };
  });

  // useScroll with offset ["start start", "end end"] maps progress 0..1 to
  // scrollY ∈ [sTop, sTop + sHeight - vh]
  const progressY = (p: number) => layout.sTop + p * (layout.sHeight - layout.vh);

  const stops = [
    { name: "02-story-decline",   y: progressY(0.10) },
    { name: "03-story-numbers",   y: progressY(0.38) },
    { name: "04-story-rise",      y: progressY(0.65) },
    { name: "05-story-dashboard", y: progressY(0.92) },
    { name: "06-marquee",         y: layout.sTop + layout.sHeight + layout.vh * 0.5 },
    { name: "07-how-it-works",    y: layout.hTop - layout.vh * 0.1 },
    { name: "08-final-cta",       y: layout.total - layout.vh },
  ];
  console.log("layout:", layout, "stops:", stops.map(s => `${s.name}@${Math.round(s.y)}`).join(", "));

  // Debug: at stop 03 read visibility info
  await page.evaluate((y) => window.scrollTo({ top: y, behavior: "instant" as any }), stops[1].y);
  await page.waitForTimeout(500);
  const debug = await page.evaluate(() => {
    const story = document.getElementById("story");
    const sticky = story?.querySelector(":scope > div");
    const sBox = story?.getBoundingClientRect();
    const stBox = sticky?.getBoundingClientRect();
    const scenes = Array.from(story?.querySelectorAll(":scope > div > div") ?? []).slice(0, 6).map((el) => {
      const cs = window.getComputedStyle(el as HTMLElement);
      const r = (el as HTMLElement).getBoundingClientRect();
      return {
        cls: (el as HTMLElement).className.slice(0, 60),
        opacity: cs.opacity,
        transform: cs.transform.slice(0, 50),
        top: Math.round(r.top), height: Math.round(r.height),
      };
    });
    return { scrollY: window.scrollY, story: { top: sBox?.top, h: sBox?.height }, sticky: { top: stBox?.top, h: stBox?.height }, scenes };
  });
  console.log("debug @ stop 03:", JSON.stringify(debug, null, 2));
  for (const s of stops) {
    await page.evaluate((y) => window.scrollTo({ top: y, behavior: "instant" as any }), s.y);
    await page.waitForTimeout(700); // let framer settle
    await page.screenshot({ path: path.join(SHOTS, `${s.name}.png`), fullPage: false });
  }

  // No runtime errors should have been logged
  if (errors.length) {
    console.log("Errors:", errors);
  }
  expect(errors, "page emitted runtime errors:\n" + errors.map(e => `· ${e.msg}`).join("\n"))
    .toEqual([]);
});

test("predict page mounts and shows upload dropzone", async ({ page }) => {
  await page.goto("/predict", { waitUntil: "networkidle" });
  await expect(page.getByText(/drop your ad screenshot/i)).toBeVisible();
  await page.screenshot({ path: path.join(SHOTS, "09-predict-upload.png"), fullPage: false });
});

test("hero sample thumbnail opens animated popup", async ({ page }) => {
  const errs: string[] = [];
  page.on("pageerror", (e) => errs.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });

  await page.goto("/", { waitUntil: "networkidle" });
  await expect(page.getByText(/tap a sample to see/i)).toBeVisible();
  // Click the first sample thumbnail. Force click because the thumbnails
  // constantly bob via framer-motion → Playwright otherwise sees them as unstable.
  await page.locator('main button').filter({ hasText: /^87\s*Scale$/ }).first().click({ force: true });
  await page.waitForTimeout(700);
  await expect(page.getByText(/top recommendation/i)).toBeVisible();
  await page.screenshot({ path: path.join(SHOTS, "13-sample-popup.png"), fullPage: false });
  await page.locator('button[aria-label="Close"]').click({ force: true });
  await page.waitForTimeout(400);

  expect(errs, "popup test emitted errors:\n" + errs.join("\n")).toEqual([]);
});

test("home page renders responsively at mobile viewport", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const page = await ctx.newPage();
  const errs: string[] = [];
  page.on("pageerror", (e) => errs.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });

  await page.goto("/", { waitUntil: "networkidle" });
  await expect(page.getByRole("heading", { name: /not reach what you expected/i })).toBeVisible();
  await page.screenshot({ path: path.join(SHOTS, "M1-hero-mobile.png"), fullPage: false });

  // Scroll into the cinematic story dwell of scene 1
  const storyTop = await page.evaluate(() => document.getElementById("story")!.getBoundingClientRect().top + window.scrollY);
  const sHeight = await page.evaluate(() => document.getElementById("story")!.getBoundingClientRect().height);
  await page.evaluate((y) => window.scrollTo({ top: y, behavior: "instant" as any }), storyTop + sHeight * 0.10);
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(SHOTS, "M2-story-mobile.png"), fullPage: false });

  expect(errs, "mobile page emitted errors:\n" + errs.join("\n")).toEqual([]);
  await ctx.close();
});

test("stats page tells the full story", async ({ page }) => {
  const errs: string[] = [];
  page.on("pageerror", (e) => errs.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });

  await page.goto("/stats", { waitUntil: "networkidle" });
  await expect(page.getByRole("heading", { name: /how we built it/i })).toBeVisible();
  await expect(page.getByText(/1,076 creatives/i).first()).toBeVisible();
  await expect(page.getByText(/avoiding leakage/i)).toBeVisible();
  await expect(page.getByText(/feature engineering/i)).toBeVisible();
  await expect(page.getByText(/honest caveats/i)).toBeVisible();
  await page.evaluate(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "instant" as any }));
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(SHOTS, "14-stats-end.png"), fullPage: false });
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" as any }));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(SHOTS, "15-stats-top.png"), fullPage: false });
  expect(errs, "stats page emitted errors:\n" + errs.join("\n")).toEqual([]);
});

test("explorer renders gallery + drawer with real images", async ({ page }) => {
  const errs: string[] = [];
  page.on("pageerror", (e) => errs.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });

  await page.goto("/explorer", { waitUntil: "networkidle" });
  await expect(page.getByRole("heading", { name: /browse all/i })).toBeVisible();

  // Wait for at least one creative image to load
  await page.locator('img[src*="/assets/creative_"]').first().waitFor({ timeout: 8000 });
  await page.screenshot({ path: path.join(SHOTS, "10-explorer-gallery.png"), fullPage: false });

  // Click a chip filter
  await page.getByRole("button", { name: /^gaming/i }).first().click();
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(SHOTS, "11-explorer-filtered.png"), fullPage: false });

  // Open detail drawer on first card
  const firstCard = page.locator('button.card-surface').first();
  await firstCard.click();
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(SHOTS, "12-explorer-detail.png"), fullPage: false });

  expect(errs, "explorer page emitted runtime errors:\n" + errs.join("\n")).toEqual([]);
});
