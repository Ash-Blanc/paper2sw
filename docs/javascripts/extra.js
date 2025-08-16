/* Mermaid initialization */
document.addEventListener("DOMContentLoaded", () => {
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: true, theme: "default" });
  }
});

/* Smooth scroll for in-page anchors */
document.addEventListener("click", function (e) {
  const anchor = e.target.closest('a[href^="#"]');
  if (!anchor) return;
  const id = decodeURIComponent(anchor.getAttribute("href").slice(1));
  const target = document.getElementById(id);
  if (!target) return;
  e.preventDefault();
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  history.replaceState(null, "", `#${id}`);
});