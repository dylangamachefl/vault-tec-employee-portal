import { useEffect, useState, useCallback } from 'react';
import { Document } from '../types';
import { getDocumentContent } from '../api';
import { X, FileText, AlertTriangle, Lock } from 'lucide-react';

interface DocumentViewerProps {
    doc: Document;
    onClose: () => void;
}

export function DocumentViewer({ doc, onClose }: DocumentViewerProps) {
    const [content, setContent] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        setLoading(true);
        setError(null);
        setContent(null);
        getDocumentContent(doc.id)
            .then((res) => setContent(res.content))
            .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load document.'))
            .finally(() => setLoading(false));
    }, [doc.id]);

    // Close on Escape key
    const handleKeyDown = useCallback(
        (e: KeyboardEvent) => {
            if (e.key === 'Escape') onClose();
        },
        [onClose],
    );

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown]);

    const statusColor =
        doc.status === 'ARCHIVED' ? 'text-vault-amber border-vault-amber' : 'text-vault-green border-vault-green';

    return (
        /* Backdrop */
        <div
            className="fixed inset-0 z-50 flex items-stretch bg-black/80 backdrop-blur-sm"
            onClick={onClose}
        >
            {/* Panel — stop propagation so clicks inside don't close */}
            <div
                className="relative flex flex-col w-full max-w-5xl mx-auto my-4 border-2 border-vault-green bg-vault-bg shadow-2xl shadow-vault-green/20 overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex-none flex items-center justify-between p-4 border-b-2 border-vault-green bg-vault-green-dark/20">
                    <div className="flex items-center gap-3 min-w-0">
                        <FileText size={20} className="flex-shrink-0" />
                        <div className="min-w-0">
                            <p className="text-xl uppercase leading-tight truncate">{doc.title}</p>
                            <div className={`flex gap-4 text-sm mt-0.5 ${statusColor}`}>
                                <span className="flex items-center gap-1">
                                    <Lock size={12} />
                                    {doc.accessLevel}
                                </span>
                                <span>DEPT: {doc.department}</span>
                                <span>DATE: {doc.effectiveDate}</span>
                                {doc.status === 'ARCHIVED' && (
                                    <span className="text-vault-amber font-bold">[ARCHIVED]</span>
                                )}
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={onClose}
                        className="flex-shrink-0 flex items-center gap-2 ml-4 border border-vault-red px-3 py-1 text-vault-red hover:bg-vault-red hover:text-vault-bg transition-colors uppercase text-sm"
                        title="Close (Esc)"
                    >
                        <X size={14} />
                        Close File
                    </button>
                </div>

                {/* Sub-header breadcrumb */}
                <div className="flex-none px-4 py-2 text-sm text-vault-green-dim bg-black/30 border-b border-vault-green/30 font-mono">
                    &gt; VAULT-TEC INTERNAL KNOWLEDGE SYSTEM / {doc.department.toUpperCase()} / {doc.id.toUpperCase()}.MD
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-6">
                    {loading && (
                        <div className="flex flex-col items-center justify-center h-48 gap-4 text-vault-green">
                            <div className="text-4xl animate-pulse">⬡</div>
                            <p className="uppercase tracking-widest animate-pulse">ACCESSING CLASSIFIED FILE...</p>
                            <div className="flex gap-1">
                                {[0, 1, 2, 3, 4].map((i) => (
                                    <span
                                        key={i}
                                        className="w-2 h-2 rounded-full bg-vault-green animate-bounce"
                                        style={{ animationDelay: `${i * 0.15}s` }}
                                    />
                                ))}
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="flex items-start gap-3 text-vault-red border border-vault-red p-4 bg-vault-red/10">
                            <AlertTriangle size={20} className="flex-shrink-0 mt-0.5" />
                            <div>
                                <p className="uppercase font-bold mb-1">ACCESS ERROR</p>
                                <p className="text-lg">{error}</p>
                            </div>
                        </div>
                    )}

                    {content !== null && !loading && (
                        <pre className="font-mono text-sm leading-relaxed text-vault-green whitespace-pre-wrap break-words">
                            {content}
                        </pre>
                    )}
                </div>

                {/* Footer */}
                <div className="flex-none px-4 py-2 border-t border-vault-green/30 bg-black/30 text-xs text-vault-green-dim font-mono flex justify-between">
                    <span>VAULT-TEC CORP. | AUTHORIZED PERSONNEL ONLY</span>
                    <span className="text-vault-amber">ESC TO CLOSE</span>
                </div>
            </div>
        </div>
    );
}
