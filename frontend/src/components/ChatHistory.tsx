import { useEffect, useRef } from 'react';
import { Message } from '../types';
import {
  AlertTriangle,
  CheckCircle,
  Info,
  TerminalSquare,
  FileText,
} from 'lucide-react';

interface ChatHistoryProps {
  messages: Message[];
}

export function ChatHistory({ messages }: ChatHistoryProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const renderStatusIcon = (status?: string) => {
    switch (status) {
      case 'error':   return <AlertTriangle className="text-vault-red mr-2" size={20} />;
      case 'warning': return <AlertTriangle className="text-vault-amber mr-2" size={20} />;
      case 'success': return <CheckCircle className="text-vault-green mr-2" size={20} />;
      case 'info':    return <Info className="text-blue-400 mr-2" size={20} />;
      default:        return <TerminalSquare className="text-vault-green mr-2" size={20} />;
    }
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'error':   return 'text-vault-red border-vault-red bg-vault-red/10';
      case 'warning': return 'text-vault-amber border-vault-amber bg-vault-amber/10';
      case 'success': return 'text-vault-green border-vault-green bg-vault-green/10';
      default:        return 'text-vault-green border-vault-green bg-vault-green-dark/20';
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {messages.map((msg) => (
        <div
          key={msg.id}
          className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
        >
          {msg.role === 'user' ? (
            <div className="max-w-[80%] border border-vault-green p-3 bg-vault-green-dark/30">
              <div className="text-sm text-vault-green-dim mb-1 uppercase">&gt; USER_INPUT</div>
              <div className="text-xl">{msg.content}</div>
            </div>
          ) : (
            <div className={`max-w-[85%] border p-4 ${getStatusColor(msg.status)}`}>
              <div className="flex items-center text-sm mb-2 uppercase opacity-80 border-b border-current pb-1">
                {renderStatusIcon(msg.status)}
                <span>
                  &gt; SYSTEM_RESPONSE{msg.status ? ` [${msg.status.toUpperCase()}]` : ''}
                </span>
              </div>
              <div className="text-xl whitespace-pre-wrap leading-relaxed">{msg.content}</div>

              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-4 pt-3 border-t border-current/30">
                  <div className="text-sm uppercase mb-2 opacity-80">
                    Source Documents ({msg.citations.length}):
                  </div>
                  <div className="grid grid-cols-1 gap-2">
                    {msg.citations.map((cit, idx) => (
                      <div
                        key={idx}
                        className="border border-current/50 p-2 bg-black/20"
                      >
                        <div className="flex items-start">
                          <FileText size={16} className="mr-2 mt-1 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-lg leading-tight">{cit.source_document}</p>
                            {cit.section_title && (
                              <p className="text-sm opacity-70">§ {cit.section_title}</p>
                            )}
                            <div className="flex flex-wrap gap-x-4 text-sm opacity-60 mt-1">
                              <span>DEPT: {cit.department}</span>
                              <span>CLEARANCE: {cit.access_level}</span>
                              {cit.doc_date && <span>DATE: {cit.doc_date}</span>}
                              {cit.doc_status && (
                                <span
                                  className={
                                    cit.doc_status === 'ARCHIVED'
                                      ? 'text-vault-amber'
                                      : ''
                                  }
                                >
                                  [{cit.doc_status}]
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
