import type { Metadata } from "next";
import { cookies } from "next/headers";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { ServiceWorkerRegistrar } from "@/components/ServiceWorkerRegistrar";
import { APPEARANCE_COOKIE } from "@/lib/theme";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Multimodal AI Workspace",
  description: "A multimodal AI workspace for chat, documents, and vision.",
};

/**
 * Applies the saved theme before the browser paints anything — first visit
 * only.
 *
 * This used to run on *every* load, because the server had no way to know the
 * theme: localStorage is invisible to it, so the HTML it sent was always
 * light and a dark-mode user saw a white flash. A React effect can't fix that
 * either, since effects run after first paint.
 *
 * The appearance cookie removes the need in the normal case — it arrives with
 * the navigation request, so the server can put `dark` on <html> itself (see
 * below). What it can't cover is a browser that has never been here: no
 * cookie has been set yet, and "follow the OS" is a media query only the
 * client can evaluate. So this stays, rendered only when the cookie is
 * missing, and its last act is to write the cookie so it isn't needed again.
 *
 * It reads Zustand's persisted envelope — `{ state: { theme }, version }` —
 * directly, because no module has loaded yet at this point. Any failure
 * (absent key, malformed JSON, storage blocked) falls through to "system",
 * which is the same default the store starts with.
 *
 * Keep the keys in sync with THEME_STORAGE_KEY and APPEARANCE_COOKIE in
 * lib/theme.ts.
 */
const themeScript = `
(function () {
  try {
    var stored = localStorage.getItem("multimodal:theme");
    var theme = "system";
    if (stored) {
      var parsed = JSON.parse(stored);
      theme = (parsed && parsed.state && parsed.state.theme) || "system";
    }
    var isDark =
      theme === "dark" ||
      (theme === "system" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", isDark);
    document.documentElement.style.colorScheme = isDark ? "dark" : "light";
    document.cookie =
      "multimodal_appearance=" + (isDark ? "dark" : "light") +
      "; Path=/; Max-Age=31536000; SameSite=Lax" +
      (location.protocol === "https:" ? "; Secure" : "");
  } catch (e) {}
})();
`;

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  /**
   * Reading a cookie makes every route dynamic — the build output changes
   * from ○ (Static) to ƒ (Dynamic), because HTML that depends on a request
   * header can't be generated ahead of time.
   *
   * Worth it here: this app is a client-rendered workspace talking to a local
   * backend, so prerendering was buying almost nothing, and the service
   * worker caches the HTML anyway (see public/sw.js). It would *not* be worth
   * it for a CDN-served marketing site, where the usual answer is to keep the
   * blocking script instead.
   */
  const appearance = (await cookies()).get(APPEARANCE_COOKIE)?.value;
  const isDark = appearance === "dark";
  // Anything other than the two known values — absent, stale, hand-edited —
  // means the server can't decide, and the script below takes over.
  const isKnown = isDark || appearance === "light";

  return (
    // suppressHydrationWarning: on a first visit the script below adds `dark`
    // to <html> before React hydrates, so the live DOM intentionally differs
    // from the server-rendered markup. It's scoped to this element's
    // attributes only — it does not silence warnings anywhere else.
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased${
        isDark ? " dark" : ""
      }`}
      style={isKnown ? { colorScheme: isDark ? "dark" : "light" } : undefined}
    >
      <body className="min-h-full flex flex-col">
        {!isKnown && <script dangerouslySetInnerHTML={{ __html: themeScript }} />}
        <ServiceWorkerRegistrar />
        {children}
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
