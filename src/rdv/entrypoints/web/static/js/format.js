/**
 * Small formatting and DOM helpers shared by the viewer modules.
 */

export const CLASS_ORDER = ["D00", "D10", "D20", "D40"];

export const CLASS_NAMES = {
  D00: "Longitudinal crack",
  D10: "Transverse crack",
  D20: "Alligator crack",
  D40: "Pothole",
};

export const CLASS_COLORS = {
  D00: "var(--c-d00)",
  D10: "var(--c-d10)",
  D20: "var(--c-d20)",
  D40: "var(--c-d40)",
};

/** Format a detection score with three decimals. */
export function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "—";
}

/** Format a 0..1 ratio as a one-decimal percentage. */
export function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
}

/** Format an integer with thousands separators. */
export function formatCount(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("en-US") : "—";
}

/** Return a short, sentence-safe description of a class code. */
export function className(code) {
  return CLASS_NAMES[code] ?? String(code);
}

/** Return the CSS color token for a class code. */
export function classColor(code) {
  return CLASS_COLORS[code] ?? "var(--accent)";
}

/** Create an element with attributes, text and children in one call. */
export function el(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) {
      continue;
    }
    if (name === "text") {
      node.textContent = String(value);
    } else if (name === "className") {
      node.className = String(value);
    } else if (name === "style") {
      for (const [property, propertyValue] of Object.entries(value)) {
        node.style.setProperty(property, String(propertyValue));
      }
    } else if (name.startsWith("data-") || name.startsWith("aria-")) {
      node.setAttribute(name, String(value));
    } else if (name in node) {
      node[name] = value;
    } else {
      node.setAttribute(name, String(value));
    }
  }
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child === null || child === undefined || child === false) {
      continue;
    }
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Remove every child of a node. */
export function clear(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

/** Clamp a number to an inclusive range. */
export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}
