/**
 * Light/dark theme control.
 *
 * The initial palette follows the operating system through
 * `prefers-color-scheme`. A manual choice is stored in localStorage and
 * applied through `data-theme`; localStorage failures (private mode, disabled
 * storage) never break the toggle.
 */

const STORAGE_KEY = "rdv-theme";
const THEMES = new Set(["light", "dark"]);

function storedTheme() {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return THEMES.has(value) ? value : null;
  } catch {
    return null;
  }
}

function storeTheme(theme) {
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* The explicit attribute still applies for this page load. */
  }
}

/** Current effective theme, following the system when nothing was chosen. */
export function currentTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (THEMES.has(explicit)) {
    return explicit;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function paint(button) {
  const theme = currentTheme();
  document.documentElement.dataset.theme = theme;
  if (button) {
    button.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
    button.setAttribute(
      "aria-label",
      theme === "dark" ? "Switch to light theme" : "Switch to dark theme",
    );
  }
  const icon = document.getElementById("theme-icon");
  if (icon) {
    icon.textContent = theme === "dark" ? "☀" : "◐";
  }
}

/** Apply the stored theme and wire the header toggle. */
export function initTheme() {
  const button = document.getElementById("theme-toggle");
  const stored = storedTheme();
  if (stored) {
    document.documentElement.dataset.theme = stored;
  } else {
    delete document.documentElement.dataset.theme;
  }
  paint(button);

  if (button) {
    button.addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      storeTheme(next);
      paint(button);
    });
  }

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!storedTheme()) {
      delete document.documentElement.dataset.theme;
      paint(button);
    }
  });
}
