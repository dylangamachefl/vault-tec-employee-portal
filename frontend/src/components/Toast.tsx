import { useEffect } from 'react';

interface ToastProps {
    message: string;
    onDismiss: () => void;
    durationMs?: number;
}

export function Toast({ message, onDismiss, durationMs = 5000 }: ToastProps) {
    useEffect(() => {
        const timer = setTimeout(onDismiss, durationMs);
        return () => clearTimeout(timer);
    }, [onDismiss, durationMs]);

    return (
        <div
            className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3
                 border-2 border-red-500 bg-vault-bg px-5 py-3 text-red-400
                 text-base uppercase tracking-widest shadow-lg shadow-red-900/40
                 animate-pulse"
            role="alert"
        >
            <span>⚠</span>
            <span>{message}</span>
            <button
                onClick={onDismiss}
                className="ml-2 text-red-300 hover:text-red-100 transition-colors"
                aria-label="Dismiss"
            >
                ✕
            </button>
        </div>
    );
}
