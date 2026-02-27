import { useState } from 'react';
import { User, Message } from '../types';
import { Sidebar } from './Sidebar';
import { ChatHistory } from './ChatHistory';
import { sendQuery } from '../api';

interface ChatViewProps {
  user: User;
  onLogout: () => void;
}

export function ChatView({ user, onLogout }: ChatViewProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'sys-1',
      role: 'system',
      content: `WELCOME, ${user.username}.\nCLEARANCE LEVEL: ${user.accessLevel}.\nPLEASE ENTER YOUR QUERY.`,
      status: 'info',
    },
  ]);
  const [input, setInput] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isQuerying) return;

    const query = input.trim();
    setInput('');

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsQuerying(true);

    try {
      const result = await sendQuery(query, user.accessLevel);

      const hasAnswer =
        result.answer &&
        !result.answer.toLowerCase().includes('not available in the provided documents');

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: result.answer,
        status: hasAnswer ? 'success' : 'warning',
        citations: result.sources.length > 0 ? result.sources : undefined,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: err instanceof Error ? err.message : 'SYSTEM ERROR: Query failed.',
        status: 'error',
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="flex h-full w-full">
      <Sidebar user={user} onLogout={onLogout} />

      <div className="flex-1 flex flex-col h-full relative">
        <ChatHistory messages={messages} />

        <div className="p-4 border-t-2 border-vault-green bg-vault-bg">
          <form onSubmit={handleSend} className="flex items-center">
            <span className="text-2xl mr-3">&gt;</span>
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
              type="submit"
              disabled={isQuerying || !input.trim()}
              className="ml-4 border-2 border-vault-green px-6 py-2 text-xl uppercase hover:bg-vault-green hover:text-vault-bg transition-colors disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-vault-green"
            >
              {isQuerying ? 'WAIT' : 'Execute'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
