/* ───────────────────────────────────────────────────────────────────────────
   On-screen Kannada keyboard.

   Present because the system accepts Kannada input only, and a substantial
   share of users at a heritage site will be on a device with no Kannada IME
   installed. Without this, the language guard would reject them with no way to
   comply — a guard that cannot be satisfied is a barrier, not a safeguard.

   Layout follows the traditional teaching order (ಸ್ವರ, ವ್ಯಂಜನ, ಕಾಗುಣಿತ),
   which is how Kannada is taught in Karnataka schools and therefore the order
   users scan for a character.
   ─────────────────────────────────────────────────────────────────────────── */

window.KannadaKeyboard = (() => {
  "use strict";

  const LAYOUT = [
    {
      label: "ಸ್ವರಗಳು · Vowels",
      keys: ["ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ", "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ", "ಅಂ", "ಅಃ"],
    },
    {
      label: "ವ್ಯಂಜನಗಳು · Consonants",
      keys: [
        "ಕ", "ಖ", "ಗ", "ಘ", "ಙ",
        "ಚ", "ಛ", "ಜ", "ಝ", "ಞ",
        "ಟ", "ಠ", "ಡ", "ಢ", "ಣ",
        "ತ", "ಥ", "ದ", "ಧ", "ನ",
        "ಪ", "ಫ", "ಬ", "ಭ", "ಮ",
        "ಯ", "ರ", "ಲ", "ವ", "ಶ",
        "ಷ", "ಸ", "ಹ", "ಳ",
      ],
    },
    {
      label: "ಕಾಗುಣಿತ · Vowel signs",
      keys: ["ಾ", "ಿ", "ೀ", "ು", "ೂ", "ೃ", "ೆ", "ೇ", "ೈ", "ೊ", "ೋ", "ೌ", "ಂ", "ಃ", "್"],
    },
    {
      label: "ಅಂಕಿ ಮತ್ತು ಚಿಹ್ನೆ · Numerals and punctuation",
      keys: ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯", "।", "॥", "?", ",", " "],
    },
  ];

  let mounted = null;

  function insert(target, text) {
    const start = target.selectionStart ?? target.value.length;
    const end = target.selectionEnd ?? start;

    target.value = target.value.slice(0, start) + text + target.value.slice(end);
    const caret = start + text.length;
    target.setSelectionRange(caret, caret);
    target.focus();
    target.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function backspace(target) {
    const start = target.selectionStart ?? target.value.length;
    const end = target.selectionEnd ?? start;

    if (start !== end) {
      target.value = target.value.slice(0, start) + target.value.slice(end);
      target.setSelectionRange(start, start);
    } else if (start > 0) {
      // Delete one full grapheme cluster where the browser supports it, so a
      // consonant plus its vowel sign disappears as one written unit rather
      // than leaving a stranded combining mark.
      let cut = 1;
      if (typeof Intl !== "undefined" && Intl.Segmenter) {
        const segmenter = new Intl.Segmenter("kn", { granularity: "grapheme" });
        const before = target.value.slice(0, start);
        const clusters = [...segmenter.segment(before)];
        cut = clusters.length ? clusters[clusters.length - 1].segment.length : 1;
      }
      target.value = target.value.slice(0, start - cut) + target.value.slice(start);
      target.setSelectionRange(start - cut, start - cut);
    }

    target.focus();
    target.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function mount(container, target) {
    if (mounted === container && container.childElementCount) return;

    container.innerHTML = "";

    LAYOUT.forEach((section) => {
      const row = document.createElement("div");
      row.className = "keyboard__row";

      const label = document.createElement("span");
      label.className = "keyboard__label";
      label.textContent = section.label;
      row.appendChild(label);

      section.keys.forEach((character) => {
        const key = document.createElement("button");
        key.type = "button";
        key.className = "key";
        key.textContent = character === " " ? "␣" : character;
        key.setAttribute("aria-label", character === " " ? "space" : character);
        key.addEventListener("click", () => insert(target, character));
        row.appendChild(key);
      });

      container.appendChild(row);
    });

    const controls = document.createElement("div");
    controls.className = "keyboard__row";

    const back = document.createElement("button");
    back.type = "button";
    back.className = "key key--wide";
    back.textContent = "⌫ ಅಳಿಸಿ";
    back.addEventListener("click", () => backspace(target));

    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "key key--wide";
    clear.textContent = "ಎಲ್ಲ ಅಳಿಸಿ";
    clear.addEventListener("click", () => {
      target.value = "";
      target.focus();
      target.dispatchEvent(new Event("input", { bubbles: true }));
    });

    controls.append(back, clear);
    container.appendChild(controls);

    mounted = container;
  }

  return { mount };
})();
