import { Archivo, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { Theme } from "../../components/theme";
import { AmbientDepth } from "../../components/three/lazy";

// One superfamily carries display and body. Archivo is variable on both weight
// and width, so headlines get set expanded (wdth 112) against normal-width body
// text — the contrast between the two roles comes from width rather than from
// dragging in a second typeface.
const archivo = Archivo({
  variable: "--font-archivo",
  subsets: ["latin"],
  axes: ["wdth"],
  display: "swap",
});

// Every number this app shows is a measurement — probabilities, thresholds,
// sample rates, durations. They all get the mono face and tabular figures so a
// changing readout doesn't reflow.
const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
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
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body className={`${archivo.variable} ${plexMono.variable} antialiased`}>
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
