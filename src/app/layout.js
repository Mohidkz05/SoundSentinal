import localFont from "next/font/local";
import "./globals.css";
import { Theme } from "../../components/theme";
import { AmbientDepth } from "../../components/three/lazy";

// Both faces are self-hosted from src/app/fonts (SIL Open Font License 1.1,
// commercial use allowed; see fonts/README.md). next/font/google fetches them
// at build time, and a build that cannot reach Google Fonts fails outright —
// which it did here on 1 October 2026. Local files remove that dependency and
// still get next/font's preloading, size-adjusted fallback and hashed URLs.

// One superfamily carries display and body. Archivo is variable on both weight
// and width, so headlines get set expanded (wdth 112) against normal-width body
// text — the contrast between the two roles comes from width rather than from
// dragging in a second typeface. Latin subset, wdth 62–125, wght 100–900.
const archivo = localFont({
  src: "./fonts/archivo-latin-100-900.woff2",
  variable: "--font-archivo",
  weight: "100 900",
  display: "swap",
});

// Every number this app shows is a measurement — probabilities, thresholds,
// sample rates, durations. They all get the mono face and tabular figures so a
// changing readout doesn't reflow.
const plexMono = localFont({
  src: [
    { path: "./fonts/plex-latin-400.woff2", weight: "400" },
    { path: "./fonts/plex-latin-500.woff2", weight: "500" },
    { path: "./fonts/plex-latin-600.woff2", weight: "600" },
  ],
  variable: "--font-plex-mono",
  display: "swap",
});

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
       `var(--font-archivo)` was undefined at :root, the whole font stack was
       dropped, and until 1 October 2026 every page rendered in the system sans. */
    <html lang="en" suppressHydrationWarning className={`${archivo.variable} ${plexMono.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body className="antialiased">
        <Theme>
          {/* One backdrop for the whole app. It sits on a negative z-index, so
              it paints above the body's canvas colour and below every panel —
              which is why no page wrapper may set its own opaque background.
              See "The 3D layer" in DESIGN.md. */}
          <AmbientDepth />
          {children}
        </Theme>
      </body>
    </html>
  );
}
