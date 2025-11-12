const keyboardContainer = document.getElementById("kannada-keyboard");
const textarea = document.getElementById("query");
const toggleKeyboardBtn = document.getElementById("toggleKeyboardBtn");

const kannadaKeys = [
  'ಅ','ಆ','ಇ','ಈ','ಉ','ಊ','ಋ','ಎ','ಏ','ಐ','ಒ','ಓ','ಔ','ಂ','ಃ',
  'ಕ','ಖ','ಗ','ಘ','ಙ',
  'ಚ','ಛ','ಜ','ಝ','ಞ',
  'ಟ','ಠ','ಡ','ಢ','ಣ',
  'ತ','ಥ','ದ','ಧ','ನ',
  'ಪ','ಫ','ಬ','ಭ','ಮ',
  'ಯ','ರ','ಲ','ವ','ಶ','ಷ','ಸ','ಹ','ಳ','ಕ್ಷ','ಜ್ಞ',
  '್','ಾ','ಿ','ೀ','ು','ೂ','ೃ','ೆ','ೇ','ೈ','ೊ','ೋ','ೌ',
  '⌫','Space'
];

function createKeyboard() {
  keyboardContainer.innerHTML = "";
  kannadaKeys.forEach(ch => {
    const key = document.createElement("div");
    key.classList.add("key");
    key.textContent = ch;
    key.addEventListener("click", () => {
      if (ch === '⌫') {
        textarea.value = textarea.value.slice(0, -1);
      } else if (ch === 'Space') {
        textarea.value += ' ';
      } else {
        textarea.value += ch;
      }
      textarea.focus();
    });
    keyboardContainer.appendChild(key);
  });
}
createKeyboard();

toggleKeyboardBtn.addEventListener("click", () => {
  keyboardContainer.classList.toggle("hidden");
  toggleKeyboardBtn.textContent = keyboardContainer.classList.contains("hidden")
    ? "⌨️ ಕನ್ನಡ ಕೀಬೋರ್ಡ್"
    : "🔽 ಕೀಬೋರ್ಡ್ ಮುಚ್ಚಿ";
});
