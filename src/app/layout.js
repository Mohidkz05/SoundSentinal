import localFont from "next/font/local";
import "./globals.css";
import { Theme } from "../../components/theme";
import { AmbientDepth } from "../../components/three/lazy";
import { Providers } from "../../components/providers";

// Both faces are self-hosted from src/app/fonts (SIL Open Font License 1.1,
// commercial use allowed; see fonts/README.md). next/font/google fetches them
// at build time, and a build that cannot reach Google Fonts fails outright —
// which it did here on 1 October 2026. Local files remove that dependency and
// still get next/font's preloading, size-adjusted fallback and hashed URLs.

// Bricolage Grotesque carries display and body (redesign, 10 October 2026:
// Archivo read as generic). Its optical-size axis is the point: at headline
// sizes the ink traps and tight apertures give the page its voice; at 15-17px
// the same file draws calmer, opener shapes for reading. Variable on weight
// (200-800), width (75-100) and opsz (12-96); the browser picks opsz from the
// font size by itself (font-optical-sizing: auto). Latin subset.
const bricolage = localFont({
  src: "./fonts/bricolage-grotesque-latin-variable.woff2",
  variable: "--font-bricolage",
  weight: "200 800",
  display: "swap",
});

// Every number this app shows is a measurement — probabilities, thresholds,
// sample rates, durations. They all get Martian Mono with tabular figures, so
// a changing readout doesn't reflow: an engineered, instrument-panel mono,
// rather than the coding-editor monos every developer tool uses. Variable on
// weight (100-800) and width (75-112.5).
const martian = localFont({
  src: "./fonts/martian-mono-latin-variable.woff2",
  variable: "--font-martian",
  weight: "100 800",
  display: "swap",
});

/* The browser chrome (mobile address bar, PWA title bar) in the canvas colour.
   Follows the OS scheme; the in-app toggle can't reach a static meta tag. */
export const viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f9fafd" }, // --color-ink-50
    { media: "(prefers-color-scheme: dark)", color: "#080721" }, // --color-ink-950
  ],
};

export const metadata = {
  title: "SoundSentinal — audio authenticity analysis",
  description:
    "Upload a voice clip and get a calibrated reading of how likely it is to be synthetic, shown against the model's own decision threshold.",
};

// Applies the stored theme before first paint. Without this the class lands in
// an effect and the page flashes light before switching.
const themeInit = `
try {
  var s = localStorage.getItem('soundsentinal-theme');
  var d = s ? s === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  if (d) document.documentElement.classList.add('dark');
} catch (e) {}
`;

export default function RootLayout({ children }) {
  return (
    /* The font variables go on <html>, not <body>. `--font-sans` is defined
       in @theme, which Tailwind emits on :root, and a custom property resolves
       its var() where it is declared — so with the variables on <body>,
       `var(--font-bricolage)` (then Archivo) was undefined at :root, the whole font stack was
       dropped, and until 1 October 2026 every page rendered in the system sans. */
    <html lang="en" data-scroll-behavior="smooth" suppressHydrationWarning className={`${bricolage.variable} ${martian.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body className="antialiased">
        <a href="#main" className="skip-link">Skip to content</a>
        <Theme>
          {/* One backdrop for the whole app. It sits on a negative z-index, so
              it paints above the body's canvas colour and below every panel —
              which is why no page wrapper may set its own opaque background.
              See "The 3D layer" in DESIGN.md. */}
          <AmbientDepth />
          <Providers>{children}</Providers>
        </Theme>
      </body>
    </html>
  );
}
