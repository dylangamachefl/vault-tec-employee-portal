import { useState, useEffect, useRef, useCallback } from 'react';
import { User, Message, Document as VaultDocument } from '../types';
import { Sidebar } from './Sidebar';
import { ChatHistory } from './ChatHistory';
import { Toast } from './Toast';
import { DocumentViewer } from './DocumentViewer';
import { sendQuery, logout, JwtPayload } from '../api';
import { jwtDecode } from 'jwt-decode';

interface ChatViewProps {
  user: User;
  onLogout: () => void;
}

function makeWelcomeMessage(user: User): Message {
  return {
    id: 'sys-1',
    role: 'system',
    content: `WELCOME, ${user.username}.\nCLEARANCE LEVEL: ${user.accessLevel}.\nPLEASE ENTER YOUR QUERY.`,
    status: 'info',
  };
}

type WorkerMessage =
  | { type: 'ready' }
  | { type: 'result'; vector: number[] }
  | { type: 'error'; message: string };

export function ChatView({ user, onLogout }: ChatViewProps) {
  const [messages, setMessages] = useState<Message[]>([makeWelcomeMessage(user)]);
  const [input, setInput] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [toastError, setToastError] = useState<string | null>(null);
  const [isWorkerReady, setIsWorkerReady] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<VaultDocument | null>(null);

  const workerRef = useRef<Worker | null>(null);
  // Map of pending embed requests: resolve/reject keyed by a nonce
  const pendingRef = useRef<Map<string, { resolve: (v: number[]) => void; reject: (e: Error) => void }>>(new Map());

  const dismissToast = useCallback(() => setToastError(null), []);

  // ── Web Worker lifecycle ───────────────────────────────────────────────────
  useEffect(() => {
    const worker = new Worker(
      new URL('../embeddingWorker.ts', import.meta.url),
      { type: 'module' },
    );
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerMessage & { nonce?: string }>) => {
      const msg = event.data;
      if (msg.type === 'ready') {
        setIsWorkerReady(true);
        return;
      }
      const nonce = (event.data as { nonce?: string }).nonce;
      if (!nonce) return;
      const pending = pendingRef.current.get(nonce);
      if (!pending) return;
      pendingRef.current.delete(nonce);

      if (msg.type === 'result') {
        pending.resolve(msg.vector);
      } else if (msg.type === 'error') {
        pending.reject(new Error(msg.message));
      }
    };

    worker.onerror = (e) => {
      setToastError(`Embedding worker error: ${e.message}`);
    };

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  function embedQuery(query: string): Promise<number[]> {
    return new Promise((resolve, reject) => {
      const worker = workerRef.current;
      if (!worker) return reject(new Error('Worker not initialised'));
      const nonce = crypto.randomUUID();
      pendingRef.current.set(nonce, { resolve, reject });
      worker.postMessage({ type: 'embed', query, nonce });
    });
  }

  // ── Proactive session expiration ───────────────────────────────────────────
  useEffect(() => {
    const token = localStorage.getItem('vault_token');
    if (!token) return;

    try {
      const { exp } = jwtDecode<JwtPayload>(token);
      const msUntilExpiry = exp * 1000 - Date.now();

      if (msUntilExpiry <= 0) {
        logout();
        onLogout();
        return;
      }

      const timer = setTimeout(() => {
        logout();
        onLogout();
      }, msUntilExpiry);

      return () => clearTimeout(timer);
    } catch {
      logout();
      onLogout();
    }
  }, [onLogout]);

  // ── Clear chat ─────────────────────────────────────────────────────────────
  const handleClear = () => {
    setMessages([makeWelcomeMessage(user)]);
  };

  // ── Send query ─────────────────────────────────────────────────────────────
  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isQuerying || !isWorkerReady) return;

    const query = input.trim();
    setInput('');

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsQuerying(true);

    try {
      const vector = await embedQuery(query);
      const result = await sendQuery(query, 5, vector);

      const hasAnswer =
        result.answer &&
        !result.answer.toLowerCase().includes('not available in the provided documents');

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: result.answer,
        status: hasAnswer ? 'success' : 'warning',
        citations: result.sources.length > 0 ? result.sources : undefined,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setToastError(err instanceof Error ? err.message : 'SYSTEM ERROR: Query failed.');
    } finally {
      setIsQuerying(false);
    }
  };

  // ── Cold-start overlay ─────────────────────────────────────────────────────
  if (!isWorkerReady) {
    return (
      <div className="flex h-full w-full">
        <Sidebar user={user} onLogout={onLogout} onViewDoc={setSelectedDoc} />
        <div className="flex-1 flex flex-col items-center justify-center gap-6 text-vault-green">
          <div className="text-5xl animate-pulse">⬡</div>
          <p className="text-xl uppercase tracking-widest text-center px-8">
            CALIBRATING NEURAL INTERFACE
          </p>
          <p className="text-sm text-vault-green-dark uppercase tracking-widest text-center px-8">
            Downloading cognitive matrix (~22MB) — First-time initialisation only.
            <br />
            Subsequent sessions will load instantly from local cache.
          </p>
          <div className="flex gap-1 mt-2">
            {[0, 1, 2, 3, 4].map((i) => (
              <span
                key={i}
                className="w-2 h-2 rounded-full bg-vault-green animate-bounce"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full">
      <Sidebar user={user} onLogout={onLogout} onViewDoc={setSelectedDoc} />

      <div className="flex-1 flex flex-col h-full relative">
        {toastError && <Toast message={toastError} onDismiss={dismissToast} />}

        {selectedDoc && (
          <DocumentViewer doc={selectedDoc} onClose={() => setSelectedDoc(null)} />
        )}

        <ChatHistory messages={messages} />

        <div className="p-4 border-t-2 border-vault-green bg-vault-bg">
          <form onSubmit={handleSend} className="flex items-center gap-2">
            <span className="text-2xl">&gt;</span>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isQuerying ? 'PROCESSING...' : 'ENTER QUERY HERE...'}
              disabled={isQuerying}
              className="flex-1 bg-transparent text-2xl border-b border-vault-green-dark focus:border-vault-green p-2 placeholder-vault-green-dark/50"
              autoFocus
            />
            <button
              type="button"
              onClick={handleClear}
              disabled={isQuerying}
              className="border border-vault-green-dark px-4 py-2 text-sm uppercase text-vault-green-dark hover:border-vault-green hover:text-vault-green transition-colors disabled:opacity-30"
              title="Clear terminal"
            >
              CLR
            </button>
            <button
              type="submit"
              disabled={isQuerying || !input.trim()}
              className="border-2 border-vault-green px-6 py-2 text-xl uppercase hover:bg-vault-green hover:text-vault-bg transition-colors disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-vault-green"
            >
              {isQuerying ? 'WAIT' : 'Execute'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
