import { useState, useRef, useEffect } from "react";

const API = "http://localhost:8000";

// ─── Cortex Chat UI ───────────────────────────────────────────────

export default function CortexUI() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [tags, setTags] = useState([]);
  const [mode, setMode] = useState("ask"); // "ask" | "search"
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [connected, setConnected] = useState(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetchStats();
    fetchTags();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function fetchStats() {
    try {
      const r = await fetch(`${API}/api/stats`);
      if (r.ok) {
        setStats(await r.json());
        setConnected(true);
      } else setConnected(false);
    } catch { setConnected(false); }
  }

  async function fetchTags() {
    try {
      const r = await fetch(`${API}/api/graph/tags`);
      if (r.ok) setTags((await r.json()).tags || []);
    } catch {}
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!input.trim() || loading) return;
    const q = input.trim();
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);

    try {
      const endpoint = mode === "ask" ? "/api/ask" : "/api/search";
      const body = mode === "ask" ? { question: q } : { query: q, top_k: 5 };
      const r = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!r.ok) throw new Error(`${r.status}`);
      const data = await r.json();

      if (mode === "ask") {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: data.answer, sources: data.sources },
        ]);
      } else {
        const resultText = data.results?.length
          ? data.results
              .map((r, i) => `**${i + 1}. ${r.title}**\n\`${r.path}\`\nTags: ${r.tags}\n${r.snippet}`)
              .join("\n\n")
          : "No results found.";
        setMessages((m) => [...m, { role: "assistant", content: resultText, isSearch: true }]);
      }
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Connection error: ${err.message}. Is the API running?`, isError: true },
      ]);
    }
    setLoading(false);
    inputRef.current?.focus();
  }

  function handleTagClick(tagName) {
    setMode("search");
    setInput(tagName);
    inputRef.current?.focus();
  }

  const greeting = [
    "Ask me anything about your vault.",
    stats ? `${stats.notes} notes indexed across ${stats.tags} tags and ${stats.entities} entities.` : "",
  ].filter(Boolean);

  return (
    <div style={styles.root}>
      <style>{globalCSS}</style>

      {/* ── Sidebar ── */}
      {sidebarOpen && (
        <aside style={styles.sidebar}>
          <div style={styles.sidebarHeader}>
            <div style={styles.logoRow}>
              <div style={styles.logoIcon}>◈</div>
              <span style={styles.logoText}>CORTEX</span>
            </div>
            <span style={styles.logoSub}>local AI second brain</span>
          </div>

          {/* Connection status */}
          <div style={{ ...styles.statusPill, background: connected ? "#0d2818" : "#2d1117", color: connected ? "#3fb950" : "#f85149", borderColor: connected ? "#1a4228" : "#4a1d24" }}>
            <span style={{ ...styles.statusDot, background: connected ? "#3fb950" : "#f85149" }} />
            {connected === null ? "Checking..." : connected ? "Connected" : "API offline"}
          </div>

          {/* Stats */}
          {stats && (
            <div style={styles.statsGrid}>
              {[
                ["Notes", stats.notes],
                ["Tags", stats.tags],
                ["Entities", stats.entities],
                ["Tag edges", stats.tag_links],
                ["Entity edges", stats.entity_links],
              ].map(([label, val]) => (
                <div key={label} style={styles.statCard}>
                  <div style={styles.statVal}>{val}</div>
                  <div style={styles.statLabel}>{label}</div>
                </div>
              ))}
            </div>
          )}

          {/* Tags */}
          {tags.length > 0 && (
            <div style={styles.tagSection}>
              <div style={styles.tagSectionTitle}>KNOWLEDGE GRAPH</div>
              <div style={styles.tagCloud}>
                {tags.slice(0, 20).map((t) => (
                  <button
                    key={t.name}
                    onClick={() => handleTagClick(t.name)}
                    style={styles.tagChip}
                    onMouseEnter={(e) => { e.target.style.background = "#c9953c"; e.target.style.color = "#0d0f12"; }}
                    onMouseLeave={(e) => { e.target.style.background = "transparent"; e.target.style.color = "#c9953c"; }}
                  >
                    {t.name}
                    <span style={styles.tagCount}>{t.count}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div style={styles.sidebarFooter}>
            <span style={{ fontSize: 11, color: "#484e58" }}>
              powered by Kuzu + ChromaDB + Ollama
            </span>
          </div>
        </aside>
      )}

      {/* ── Main Chat Area ── */}
      <main style={styles.main}>
        {/* Top bar */}
        <div style={styles.topBar}>
          <button onClick={() => setSidebarOpen(!sidebarOpen)} style={styles.toggleBtn}>
            {sidebarOpen ? "◂" : "▸"} 
          </button>
          <div style={styles.modeSwitcher}>
            {["ask", "search"].map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                style={{ ...styles.modeBtn, ...(mode === m ? styles.modeBtnActive : {}) }}
              >
                {m === "ask" ? "◈ Ask" : "⊞ Search"}
              </button>
            ))}
          </div>
        </div>

        {/* Messages */}
        <div style={styles.chatArea}>
          {messages.length === 0 && (
            <div style={styles.emptyState}>
              <div style={styles.emptyIcon}>◈</div>
              <div style={styles.emptyTitle}>Cortex</div>
              <div style={styles.emptySubtitle}>
                {greeting.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
              <div style={styles.emptySuggestions}>
                {[
                  "What projects am I working on?",
                  "Find everything about ADINO",
                  "What ideas do I have for side projects?",
                  "Summarize my thesis work",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); inputRef.current?.focus(); }}
                    style={styles.suggestionBtn}
                    onMouseEnter={(e) => { e.target.style.borderColor = "#c9953c"; e.target.style.color = "#e8dcc8"; }}
                    onMouseLeave={(e) => { e.target.style.borderColor = "#2a2e36"; e.target.style.color = "#7a8190"; }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} style={msg.role === "user" ? styles.userRow : styles.assistantRow}>
              <div style={msg.role === "user" ? styles.userBubble : msg.isError ? styles.errorBubble : styles.assistantBubble}>
                <div style={styles.msgContent}>
                  {msg.content.split("\n").map((line, j) => {
                    if (line.startsWith("**") && line.endsWith("**")) {
                      return <div key={j} style={styles.boldLine}>{line.replace(/\*\*/g, "")}</div>;
                    }
                    if (line.startsWith("`") && line.endsWith("`")) {
                      return <div key={j} style={styles.codeLine}>{line.replace(/`/g, "")}</div>;
                    }
                    return <div key={j}>{line || "\u00A0"}</div>;
                  })}
                </div>
                {msg.sources?.length > 0 && (
                  <div style={styles.sourcesBar}>
                    <span style={styles.sourcesLabel}>Sources:</span>
                    {msg.sources.map((s, j) => (
                      <span key={j} style={styles.sourceChip} title={s.path}>
                        {s.title}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div style={styles.assistantRow}>
              <div style={styles.assistantBubble}>
                <div style={styles.loadingDots}>
                  <span style={{ ...styles.dot, animationDelay: "0s" }} />
                  <span style={{ ...styles.dot, animationDelay: "0.15s" }} />
                  <span style={{ ...styles.dot, animationDelay: "0.3s" }} />
                </div>
              </div>
            </div>
          )}
          <div ref={scrollRef} />
        </div>

        {/* Input */}
        <div style={styles.inputArea}>
          <form onSubmit={handleSubmit} style={styles.inputForm}>
            <div style={styles.inputWrapper}>
              <span style={styles.inputPrefix}>{mode === "ask" ? "◈" : "⊞"}</span>
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={mode === "ask" ? "Ask about your knowledge..." : "Search your vault..."}
                style={styles.input}
                disabled={loading || !connected}
              />
              <button
                type="submit"
                disabled={!input.trim() || loading || !connected}
                style={{ ...styles.sendBtn, opacity: input.trim() && !loading ? 1 : 0.3 }}
              >
                →
              </button>
            </div>
          </form>
          <div style={styles.inputHint}>
            {mode === "ask" ? "LLM-powered answers with source citations" : "Semantic vector search across all notes"}
          </div>
        </div>
      </main>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────

const globalCSS = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');
  @keyframes pulse { 0%, 100% { opacity: .25; } 50% { opacity: 1; } }
`;

const font = {
  mono: "'JetBrains Mono', monospace",
  sans: "'DM Sans', sans-serif",
};

const c = {
  bg: "#0d0f12",
  surface: "#13161b",
  surface2: "#1a1e25",
  border: "#2a2e36",
  text: "#c8cdd5",
  textDim: "#636a76",
  textBright: "#e8ecf2",
  accent: "#c9953c",
  accentDim: "#8b6a2f",
  user: "#1c2a3a",
  userBorder: "#2a4060",
  error: "#3d1f1f",
  errorBorder: "#6b3030",
};

const styles = {
  root: {
    display: "flex",
    height: "100vh",
    width: "100%",
    background: c.bg,
    color: c.text,
    fontFamily: font.sans,
    fontSize: 14,
    overflow: "hidden",
  },

  // Sidebar
  sidebar: {
    width: 260,
    minWidth: 260,
    background: c.surface,
    borderRight: `1px solid ${c.border}`,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  sidebarHeader: {
    padding: "24px 20px 16px",
    borderBottom: `1px solid ${c.border}`,
  },
  logoRow: {
    display: "flex",
    alignItems: "center",
    gap: 10,
  },
  logoIcon: {
    fontSize: 22,
    color: c.accent,
    fontFamily: font.mono,
  },
  logoText: {
    fontSize: 16,
    fontWeight: 700,
    letterSpacing: 3,
    color: c.textBright,
    fontFamily: font.mono,
  },
  logoSub: {
    display: "block",
    fontSize: 11,
    color: c.textDim,
    marginTop: 6,
    fontFamily: font.mono,
    letterSpacing: 0.5,
  },
  statusPill: {
    margin: "16px 20px 0",
    padding: "6px 12px",
    borderRadius: 6,
    fontSize: 12,
    fontFamily: font.mono,
    fontWeight: 600,
    display: "flex",
    alignItems: "center",
    gap: 8,
    border: "1px solid",
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    display: "inline-block",
  },
  statsGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 6,
    padding: "16px 20px 8px",
  },
  statCard: {
    background: c.surface2,
    borderRadius: 6,
    padding: "10px 12px",
    border: `1px solid ${c.border}`,
  },
  statVal: {
    fontSize: 20,
    fontWeight: 700,
    color: c.accent,
    fontFamily: font.mono,
  },
  statLabel: {
    fontSize: 10,
    color: c.textDim,
    textTransform: "uppercase",
    letterSpacing: 1,
    fontFamily: font.mono,
    marginTop: 2,
  },
  tagSection: {
    padding: "12px 20px",
    flex: 1,
    overflow: "auto",
  },
  tagSectionTitle: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.5,
    color: c.textDim,
    fontFamily: font.mono,
    marginBottom: 10,
  },
  tagCloud: {
    display: "flex",
    flexWrap: "wrap",
    gap: 4,
  },
  tagChip: {
    background: "transparent",
    border: `1px solid ${c.accentDim}`,
    color: c.accent,
    borderRadius: 4,
    padding: "3px 8px",
    fontSize: 11,
    fontFamily: font.mono,
    cursor: "pointer",
    display: "inline-flex",
    alignItems: "center",
    gap: 5,
    transition: "all .15s",
  },
  tagCount: {
    fontSize: 9,
    opacity: 0.6,
  },
  sidebarFooter: {
    padding: "12px 20px",
    borderTop: `1px solid ${c.border}`,
    textAlign: "center",
  },

  // Main
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    minWidth: 0,
  },
  topBar: {
    padding: "12px 20px",
    borderBottom: `1px solid ${c.border}`,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  toggleBtn: {
    background: "none",
    border: "none",
    color: c.textDim,
    cursor: "pointer",
    fontSize: 16,
    padding: "4px 8px",
    fontFamily: font.mono,
  },
  modeSwitcher: {
    display: "flex",
    gap: 2,
    background: c.surface2,
    borderRadius: 6,
    padding: 2,
  },
  modeBtn: {
    background: "transparent",
    border: "none",
    color: c.textDim,
    padding: "6px 14px",
    borderRadius: 5,
    cursor: "pointer",
    fontSize: 12,
    fontFamily: font.mono,
    fontWeight: 600,
    transition: "all .15s",
  },
  modeBtnActive: {
    background: c.accent,
    color: c.bg,
  },

  // Chat
  chatArea: {
    flex: 1,
    overflow: "auto",
    padding: "24px 20px",
  },
  emptyState: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    textAlign: "center",
    gap: 8,
  },
  emptyIcon: {
    fontSize: 48,
    color: c.accent,
    fontFamily: font.mono,
    opacity: 0.6,
  },
  emptyTitle: {
    fontSize: 28,
    fontWeight: 700,
    color: c.textBright,
    fontFamily: font.mono,
    letterSpacing: 4,
  },
  emptySubtitle: {
    fontSize: 13,
    color: c.textDim,
    maxWidth: 380,
    lineHeight: 1.6,
  },
  emptySuggestions: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
    justifyContent: "center",
    marginTop: 20,
    maxWidth: 500,
  },
  suggestionBtn: {
    background: "transparent",
    border: `1px solid ${c.border}`,
    color: c.textDim,
    padding: "8px 14px",
    borderRadius: 8,
    cursor: "pointer",
    fontSize: 12,
    fontFamily: font.sans,
    transition: "all .15s",
  },

  // Messages
  userRow: {
    display: "flex",
    justifyContent: "flex-end",
    marginBottom: 16,
  },
  assistantRow: {
    display: "flex",
    justifyContent: "flex-start",
    marginBottom: 16,
  },
  userBubble: {
    background: c.user,
    border: `1px solid ${c.userBorder}`,
    borderRadius: "16px 16px 4px 16px",
    padding: "12px 16px",
    maxWidth: "70%",
    color: c.textBright,
  },
  assistantBubble: {
    background: c.surface,
    border: `1px solid ${c.border}`,
    borderRadius: "16px 16px 16px 4px",
    padding: "12px 16px",
    maxWidth: "75%",
  },
  errorBubble: {
    background: c.error,
    border: `1px solid ${c.errorBorder}`,
    borderRadius: "16px 16px 16px 4px",
    padding: "12px 16px",
    maxWidth: "75%",
    color: "#f0a0a0",
  },
  msgContent: {
    lineHeight: 1.65,
    fontSize: 14,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  boldLine: {
    fontWeight: 600,
    color: c.textBright,
    marginTop: 4,
  },
  codeLine: {
    fontFamily: font.mono,
    fontSize: 12,
    color: c.accent,
    background: c.surface2,
    padding: "2px 6px",
    borderRadius: 3,
    display: "inline-block",
    marginTop: 2,
  },
  sourcesBar: {
    marginTop: 10,
    paddingTop: 8,
    borderTop: `1px solid ${c.border}`,
    display: "flex",
    flexWrap: "wrap",
    gap: 6,
    alignItems: "center",
  },
  sourcesLabel: {
    fontSize: 10,
    color: c.textDim,
    fontFamily: font.mono,
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  sourceChip: {
    background: c.surface2,
    border: `1px solid ${c.border}`,
    color: c.accent,
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: 11,
    fontFamily: font.mono,
  },

  // Loading
  loadingDots: {
    display: "flex",
    gap: 6,
    padding: "4px 0",
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: c.accent,
    animation: "pulse 0.8s infinite ease-in-out",
    display: "inline-block",
  },

  // Input
  inputArea: {
    padding: "12px 20px 16px",
    borderTop: `1px solid ${c.border}`,
    background: c.surface,
  },
  inputForm: {
    display: "flex",
  },
  inputWrapper: {
    display: "flex",
    alignItems: "center",
    width: "100%",
    background: c.bg,
    border: `1px solid ${c.border}`,
    borderRadius: 10,
    padding: "0 4px 0 14px",
    transition: "border-color .15s",
  },
  inputPrefix: {
    color: c.accent,
    fontFamily: font.mono,
    fontSize: 16,
    marginRight: 10,
    opacity: 0.7,
  },
  input: {
    flex: 1,
    background: "transparent",
    border: "none",
    outline: "none",
    color: c.textBright,
    fontSize: 14,
    fontFamily: font.sans,
    padding: "12px 0",
  },
  sendBtn: {
    background: c.accent,
    border: "none",
    color: c.bg,
    width: 32,
    height: 32,
    borderRadius: 7,
    cursor: "pointer",
    fontSize: 16,
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "opacity .15s",
    fontFamily: font.mono,
  },
  inputHint: {
    fontSize: 10,
    color: c.textDim,
    fontFamily: font.mono,
    marginTop: 6,
    textAlign: "center",
    letterSpacing: 0.3,
  },
};
