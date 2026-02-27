import { useState } from 'react';
import { User } from '../types';
import { login } from '../api';
import { Terminal, ShieldAlert } from 'lucide-react';

interface LoginViewProps {
  onLogin: (user: User) => void;
}

/** Demo credential profiles — match DEMO_USERS in src/api/main.py */
const DEMO_USERS = [
  { id: 'u1', label: '[GENERAL] Dweller-101 — General Employee' },
  { id: 'u2', label: '[HR] Barnsworth B. — HR Specialist' },
  { id: 'u3', label: '[MARKETING] Gable M. — Marketing Associate' },
  { id: 'u4', label: '[ADMIN] Carmichael J. — IT Administrator' },
];

export function LoginView({ onLogin }: LoginViewProps) {
  const [selectedId, setSelectedId] = useState(DEMO_USERS[0].id);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLogin = async () => {
    setIsAuthenticating(true);
    setError(null);
    try {
      const user = await login(selectedId);
      onLogin(user);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Authentication failed.');
      setIsAuthenticating(false);
    }
  };

  return (
    <div className="flex items-center justify-center h-full">
      <div className="border-2 border-vault-green p-8 max-w-md w-full bg-vault-bg shadow-[0_0_20px_rgba(74,222,128,0.2)]">
        <div className="flex items-center justify-center mb-6">
          <Terminal size={48} className="text-vault-green mr-4" />
          <h2 className="text-4xl uppercase tracking-wider">System Login</h2>
        </div>

        <div className="mb-6 border border-vault-amber p-4 bg-vault-amber/10 text-vault-amber flex items-start">
          <ShieldAlert className="mr-3 flex-shrink-0 mt-1" size={24} />
          <p className="text-lg leading-tight">
            UNAUTHORIZED ACCESS IS STRICTLY PROHIBITED. VIOLATORS WILL BE ASSIGNED TO REACTOR MAINTENANCE.
          </p>
        </div>

        {error && (
          <div className="mb-4 border border-vault-red p-3 bg-vault-red/10 text-vault-red text-lg">
            ERROR: {error}
          </div>
        )}

        <div className="space-y-6">
          <div>
            <label className="block text-xl mb-2 uppercase">Select Credential Profile:</label>
            <select
              className="w-full bg-vault-bg border-2 border-vault-green text-vault-green p-3 text-xl appearance-none cursor-pointer hover:bg-vault-green-dark/30 transition-colors"
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              disabled={isAuthenticating}
            >
              {DEMO_USERS.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.label}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleLogin}
            disabled={isAuthenticating}
            className="w-full border-2 border-vault-green bg-vault-green text-vault-bg text-2xl py-3 uppercase font-bold hover:bg-vault-green-dim hover:border-vault-green-dim transition-colors disabled:opacity-50"
          >
            {isAuthenticating ? '> AUTHENTICATING...' : '> INITIALIZE CONNECTION'}
          </button>
        </div>
      </div>
    </div>
  );
}
