/**
 * The page shown when a navigation fails because there's no network.
 *
 * Deliberately styled with inline attributes rather than Tailwind classes.
 * The service worker caches this page's HTML, but at this step it caches no
 * CSS or JS — so offline, the stylesheet request fails and any class-based
 * styling would render as unstyled text. Inlining keeps the fallback looking
 * correct with nothing but the HTML document, which is the one condition it
 * exists to handle.
 *
 * No client component, no hooks, no store access, for the same reason: none
 * of that JavaScript is available when this page is doing its job. It's a
 * static document by necessity, not by preference.
 *
 * Both colour schemes are handled by `color-scheme: light dark` plus
 * `currentColor`-ish neutrals, since the theme script in the root layout
 * needs localStorage and this page can't rely on having run any of it.
 */
export const metadata = {
  title: "Offline — Multimodal AI Workspace",
};

export default function OfflinePage() {
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "0.75rem",
        padding: "2rem",
        textAlign: "center",
        colorScheme: "light dark",
        fontFamily:
          "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      <div
        aria-hidden
        style={{ fontSize: "2.5rem", lineHeight: 1, opacity: 0.55 }}
      >
        ⚡
      </div>
      <h1 style={{ fontSize: "1.125rem", fontWeight: 600, margin: 0 }}>
        You&apos;re offline
      </h1>
      <p
        style={{
          margin: 0,
          maxWidth: "26rem",
          fontSize: "0.875rem",
          lineHeight: 1.6,
          opacity: 0.7,
        }}
      >
        This page couldn&apos;t load because there&apos;s no connection. Your saved
        chats are still on this device — they&apos;ll be there when you reconnect.
      </p>
      <p style={{ margin: 0, fontSize: "0.8125rem", opacity: 0.5 }}>
        Check your connection and reload.
      </p>
    </div>
  );
}
