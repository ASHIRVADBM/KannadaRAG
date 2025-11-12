const API_URL = "http://localhost:8000";

const askBtn = document.getElementById("askBtn");
const queryBox = document.getElementById("query");
const chatBox = document.getElementById("chat-box");
const newChatBtn = document.getElementById("newChatBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const chatHistoryDiv = document.getElementById("chat-history");
const speakBtn = document.getElementById("speakBtn");

let lastAnswer = "";
let chats = JSON.parse(localStorage.getItem("kannada_chats")) || [];
let currentChat = { id: Date.now(), messages: [] };

// ✅ Auto-grow textarea
queryBox.addEventListener("input", () => {
  queryBox.style.height = "auto";
  queryBox.style.height = queryBox.scrollHeight + "px";
});

askBtn.addEventListener("click", ask);
newChatBtn.addEventListener("click", startNewChat);
clearHistoryBtn.addEventListener("click", clearHistory);
speakBtn.addEventListener("click", speakAnswer);

// 🆕 Start new chat
function startNewChat() {
  if (currentChat.messages.length > 0) saveCurrentChat();
  currentChat = { id: Date.now(), messages: [] };
  chatBox.innerHTML = "";
  queryBox.value = "";
  speakBtn.classList.add("hidden");
  updateChatHistory();
}

// 🗑️ Clear all history
function clearHistory() {
  if (confirm("ನೀವು ಖಚಿತವಾಗಿಯೂ ಚಾಟ್ ಇತಿಹಾಸವನ್ನು ಅಳಿಸಲು ಬಯಸುವಿರಾ?")) {
    localStorage.removeItem("kannada_chats");
    chats = [];
    currentChat = { id: Date.now(), messages: [] };
    chatHistoryDiv.innerHTML = "";
    chatBox.innerHTML = "";
    queryBox.value = "";
    speakBtn.classList.add("hidden");
  }
}

// 🧠 Ask Function
async function ask() {
  const query = queryBox.value.trim();
  if (!query) return alert("ದಯವಿಟ್ಟು ಪ್ರಶ್ನೆ ಬರೆಯಿರಿ!");

  addMessage(query, "user");
  queryBox.value = "";
  addMessage("🤔 ಯೋಚಿಸುತ್ತಿದೆ...", "bot", true);

  const formData = new FormData();
  formData.append("query", query);

  try {
    const res = await fetch(`${API_URL}/ask`, { method: "POST", body: formData });
    const data = await res.json();
    const answer = data.answer || "ಯಾವುದೇ ಉತ್ತರ ಸಿಕ್ಕಿಲ್ಲ.";

    removeThinking();
    addMessage(answer, "bot");
    lastAnswer = answer;
    speakBtn.classList.remove("hidden");

    // Save new chat after each response
    saveCurrentChat();
  } catch (e) {
    removeThinking();
    addMessage("⚠️ ಸರ್ವರ್ ದೋಷ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.", "bot");
  }
}

// 🗨️ Add message (only display, don’t save if loading old chat)
function addMessage(text, sender, isThinking = false, skipSave = false) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  msg.textContent = text;
  if (isThinking) msg.classList.add("thinking");
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;

  if (!isThinking && !skipSave) {
    currentChat.messages.push({ sender, text });
  }
}

function removeThinking() {
  const thinking = document.querySelector(".thinking");
  if (thinking) thinking.remove();
}

// 💾 Save chat only once per conversation
function saveCurrentChat() {
  const index = chats.findIndex(c => c.id === currentChat.id);
  if (index >= 0) chats[index] = currentChat;
  else chats.push(currentChat);
  localStorage.setItem("kannada_chats", JSON.stringify(chats));
  updateChatHistory();
}

// 📜 Update chat history sidebar
function updateChatHistory() {
  chatHistoryDiv.innerHTML = "";
  chats.forEach((chat, i) => {
    const btn = document.createElement("button");
    const title = chat.messages[0]?.text.slice(0, 15) || `ಚಾಟ್ ${i + 1}`;
    btn.textContent = title;
    btn.classList.add("chat-history-btn");
    btn.onclick = () => loadChat(chat.id);
    chatHistoryDiv.appendChild(btn);
  });
}

// 🔁 Load old chat — only display, no duplication
function loadChat(id) {
  const chat = chats.find(c => c.id === id);
  if (!chat) return;
  currentChat = chat;
  chatBox.innerHTML = "";

  // Display previous messages without saving them again
  chat.messages.forEach(m => addMessage(m.text, m.sender, false, true));

  lastAnswer = chat.messages.at(-1)?.text || "";
  speakBtn.classList.remove("hidden");
}

// 🔊 Kannada Voice Output
function speakAnswer() {
  if (!lastAnswer) return;

  const utter = new SpeechSynthesisUtterance(lastAnswer);
  utter.lang = "kn-IN";
  utter.rate = 1.0;
  utter.pitch = 1.0;
  utter.volume = 1.0;

  const voices = window.speechSynthesis.getVoices();
  const kannadaVoice =
    voices.find(v => v.lang.startsWith("kn")) ||
    voices.find(v => v.lang.startsWith("hi")) ||
    voices.find(v => v.lang.startsWith("en-IN"));
  if (kannadaVoice) utter.voice = kannadaVoice;

  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

updateChatHistory();
