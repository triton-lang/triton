// @ts-check

// Extra JS capability for selected tabs to be synced.
// The selection is stored in the browser's local storage, so that it persists
// across page loads and browsing sessions. The storage key prefix is
// configurable (and can be set to an empty string to disable persistence).

/**
 * @type {Record<string, HTMLElement[]>}
 */
let sd_id_to_elements = {};

// The storage key prefix is delivered as a `data-` attribute on this script
// tag, and must be captured here at eval time: `document.currentScript` is only
// defined while the script is first executing, not inside later callbacks such
// as `ready`. An empty string disables persistence entirely.
const storageKeyPrefix =
  document.currentScript?.getAttribute("data-sd-tabs-storage-prefix") ??
  "sphinx-design-tab-id-";

/**
 * Create a key for a tab element.
 * @param {HTMLElement} el - The tab element.
 * @returns {[string, string, string] | null} - The key.
 *
 */
function create_key(el) {
  let syncId = el.getAttribute("data-sync-id");
  let syncGroup = el.getAttribute("data-sync-group");
  if (!syncId || !syncGroup) return null;
  return [syncGroup, syncId, syncGroup + "--" + syncId];
}

/**
 * Get the radio input associated with a tab label.
 *
 * Per ``TabSetHtmlTransform`` the HTML DOM order is input, label, content, so
 * the input directly precedes the label; the label's ``for`` attribute also
 * points at the input's id.
 *
 * @param {HTMLElement} label - The tab label.
 * @returns {HTMLInputElement | null} - The associated radio input, if found.
 */
function get_label_input(label) {
  const forId = label.getAttribute("for");
  const el = forId
    ? document.getElementById(forId)
    : label.previousElementSibling;
  return el instanceof HTMLInputElement ? el : null;
}

/**
 * Read the stored sync id for a group (returns null if persistence is disabled).
 *
 * Storage access may throw (e.g. a SecurityError when the browser blocks site
 * data, or ``window.localStorage`` itself being inaccessible), so failures are
 * swallowed: persistence is then simply unavailable, but tab syncing and
 * hash handling must keep working.
 *
 * @param {string} group - The sync group.
 * @returns {string | null} - The stored sync id, if any.
 */
function get_stored_id(group) {
  if (!storageKeyPrefix) return null;
  try {
    return window.localStorage.getItem(storageKeyPrefix + group);
  } catch (err) {
    return null;
  }
}

/**
 * Persist the selected sync id for a group (no-op if persistence is disabled).
 *
 * See ``get_stored_id`` regarding swallowed storage failures.
 *
 * @param {string} group - The sync group.
 * @param {string} id - The selected sync id.
 */
function set_stored_id(group, id) {
  if (!storageKeyPrefix) return;
  try {
    window.localStorage.setItem(storageKeyPrefix + group, id);
  } catch (err) {
    // persistence unavailable; ignore
  }
}

/**
 * Initialize the tab selection.
 *
 */
function ready() {
  // Find all tabs with sync data

  /** @type {string[]} */
  let groups = [];

  document.querySelectorAll(".sd-tab-label").forEach((label) => {
    if (label instanceof HTMLElement) {
      let data = create_key(label);
      if (data) {
        let [group, id, key] = data;

        // Sync on the radio input's `change` event, rather than the label's
        // click: `change` fires exactly once per selection regardless of the
        // activation method (mouse, click+drag release, keyboard arrows, JS).
        const input = get_label_input(label);
        if (input) {
          input.addEventListener("change", onSDInputChange);
        }

        // store map of key to elements
        if (!sd_id_to_elements[key]) {
          sd_id_to_elements[key] = [];
        }
        sd_id_to_elements[key].push(label);

        if (groups.indexOf(group) === -1) {
          groups.push(group);
          // Check if a specific tab has been selected via URL query parameter
          const tabParam = new URLSearchParams(window.location.search).get(
            group
          );
          if (tabParam) {
            set_stored_id(group, tabParam);
          }
        }

        // Check if a specific tab has been selected previously
        let previousId = get_stored_id(group);
        if (previousId === id && input) {
          // set `.checked` directly (does not re-fire `change`)
          input.checked = true;
        }
      }
    }
  });

  // Open the tab targeted by the URL fragment (on load and on later changes)
  select_tab_from_hash();
  window.addEventListener("hashchange", select_tab_from_hash);
}

/**
 * Activate the other tabs sharing the changed input's sync id.
 *
 * @param {Event} event - The `change` event fired by a tab's radio input.
 */
function onSDInputChange(event) {
  const input = event.currentTarget;
  if (!(input instanceof HTMLInputElement) || !input.checked) return;
  // the label carries the sync data, and directly follows the input
  const label = input.nextElementSibling;
  if (!(label instanceof HTMLElement)) return;
  let data = create_key(label);
  if (!data) return;
  let [group, id, key] = data;
  for (const other of sd_id_to_elements[key]) {
    if (other === label) continue;
    const otherInput = get_label_input(other);
    if (otherInput) {
      // set `.checked` directly: this does NOT re-fire `change`, so the sync
      // stays idempotent (no `.click()`, no event cascade between tab-sets)
      otherInput.checked = true;
    }
  }
  set_stored_id(group, id);
}

/**
 * Select the tab targeted by the current URL fragment, if any.
 *
 * The fragment may target a `.sd-tab-label` directly (via a tab-item `:name:`),
 * or an element inside a `.sd-tab-content` panel; in the latter case the
 * panel's own radio input is selected. Per ``TabSetHtmlTransform`` the DOM
 * order is input, label, content, so a content panel's input is its
 * ``previousElementSibling.previousElementSibling``.
 *
 * Tab-sets can be nested inside other tabs' panels, so every enclosing
 * `.sd-tab-content` ancestor is opened as well — otherwise the target would
 * stay hidden inside a non-selected outer tab.
 */
function select_tab_from_hash() {
  const hash = window.location.hash;
  if (!hash) return;
  const target = document.getElementById(hash.slice(1));
  if (!target) return;

  let opened = false;

  // the hash may target a tab label directly (via a tab-item `:name:`)
  const label = target.closest(".sd-tab-label");
  if (label instanceof HTMLElement) {
    const input = get_label_input(label);
    if (input) {
      input.checked = true;
      opened = true;
    }
  }

  // open every enclosing panel: starting from the target (or, for a label,
  // from the label itself, which sits outside its own panel), walk up the
  // `.sd-tab-content` ancestors and select each panel's radio input
  const start = label instanceof HTMLElement ? label : target;
  for (
    let content = start.closest(".sd-tab-content");
    content;
    content = content.parentElement
      ? content.parentElement.closest(".sd-tab-content")
      : null
  ) {
    const input = content.previousElementSibling?.previousElementSibling;
    if (input instanceof HTMLInputElement) {
      input.checked = true;
      opened = true;
    }
  }

  if (!opened) return;

  // re-scroll the target into view, since panel visibility (and hence the
  // page layout) has just changed
  target.scrollIntoView();
}

document.addEventListener("DOMContentLoaded", ready, false);
