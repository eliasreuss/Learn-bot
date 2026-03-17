import { useState, useRef, useEffect, KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import "./App.css";

const API_URL = "/api/chat";

interface Resource {
  title: string;
  url: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  resources?: Resource[];
}

function Spinner() {
  return (
    <div className="spinner-row">
      <div className="spinner" />
    </div>
  );
}

function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`chat-row ${isUser ? "chat-row--user" : "chat-row--bot"}`}>
      <div className={`bubble ${isUser ? "bubble--user" : "bubble--bot"}`}>
        <div className="bubble-content">
          <ReactMarkdown remarkPlugins={[remarkBreaks]}>{message.content}</ReactMarkdown>
        </div>

        {message.resources && message.resources.length > 0 && (
          <div className="bubble-resources">
            <span className="bubble-resources__label">Resources</span>
            {message.resources.map((r) => (
              <a
                key={r.url}
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
                className="bubble-resources__link"
              >
                {r.title}
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hi! I'm the Inact Now assistant. Ask me anything about using Inact Now.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const lastMessageRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    lastMessageRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [messages, loading]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight}px`;
  }, [input]);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    const updatedMessages: Message[] = [
      ...messages,
      { role: "user", content: question },
    ];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    const lastAssistantMsg = [...updatedMessages]
      .reverse()
      .find((m) => m.role === "assistant");

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history: updatedMessages.slice(-6).map((m) => ({
            role: m.role,
            content: m.content,
          })),
          last_answer: lastAssistantMsg?.content ?? "",
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || `Server error: ${res.status}`);

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, resources: data.resources },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Something went wrong.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${msg}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="page">
      <h1 className="page-title">Inact Assistant</h1>
      <p className="page-subtitle">
        Ask me anything Inact related, and I'll do my best to answer
      </p>

      <div className="chat-card">
        <div className="chat-messages">
          {messages.map((m, i) => (
            <div key={i} ref={i === messages.length - 1 ? lastMessageRef : undefined}>
              <ChatMessage message={m} />
            </div>
          ))}

          {loading && <Spinner />}
        </div>

        <div className="chat-input-wrapper">
          <div className="chat-input-row">
            <textarea
              ref={textareaRef}
              className="chat-input"
              placeholder="Ask a question…"
              value={input}
              rows={1}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
            <button
              className="send-btn"
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              aria-label="Send"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path
                  d="M12 19V5M5 12l7-7 7 7"
                  stroke="white"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
