import { useEffect, useState } from 'react';
import { User, Document } from '../types';
import { getDocuments } from '../api';
import { FileText, Folder, Lock, LogOut, AlertTriangle } from 'lucide-react';

interface SidebarProps {
  user: User;
  onLogout: () => void;
}

export function Sidebar({ user, onLogout }: SidebarProps) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getDocuments(user.accessLevel)
      .then(setDocs)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load documents.'))
      .finally(() => setLoading(false));
  }, [user.accessLevel]);

  // Group by department
  const grouped = docs.reduce<Record<string, Document[]>>((acc, doc) => {
    if (!acc[doc.department]) acc[doc.department] = [];
    acc[doc.department].push(doc);
    return acc;
  }, {});

  const departments = ['General', 'HR', 'Marketing', 'Admin'];

  return (
    <div className="w-80 border-r-2 border-vault-green flex flex-col h-full bg-vault-green-dark/10">
      {/* User Info */}
      <div className="p-4 border-b-2 border-vault-green">
        <h2 className="text-2xl uppercase mb-2 flex items-center">
          <Lock className="mr-2" size={20} /> Current User
        </h2>
        <div className="text-lg">
          <p>ID: {user.username}</p>
          <p>ROLE: {user.role}</p>
          <p className="text-vault-amber">CLEARANCE: {user.accessLevel}</p>
        </div>
        <button
          onClick={onLogout}
          className="mt-4 flex items-center text-vault-red hover:text-red-400 uppercase text-lg border border-vault-red px-2 py-1 transition-colors"
        >
          <LogOut size={16} className="mr-2" /> Terminate Session
        </button>
      </div>

      {/* Document Browser */}
      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="text-xl uppercase mb-4 border-b border-vault-green pb-1">Knowledge Base</h3>

        {loading && (
          <p className="text-vault-green-dim text-lg animate-pulse">LOADING RECORDS...</p>
        )}

        {error && (
          <div className="flex items-start text-vault-amber text-lg">
            <AlertTriangle size={16} className="mr-2 mt-1 flex-shrink-0" />
            <span>ERROR: {error}</span>
          </div>
        )}

        {!loading && !error && (
          <div className="space-y-6">
            {departments.map((dept) => {
              const deptDocs = grouped[dept];
              if (!deptDocs) return null;
              return (
                <div key={dept}>
                  <div className="flex items-center text-xl mb-2 text-vault-green-dim">
                    <Folder size={18} className="mr-2" />
                    <span className="uppercase">DIR: {dept}</span>
                  </div>
                  <ul className="space-y-2 pl-6">
                    {deptDocs.map((doc) => (
                      <li key={doc.id} className="group">
                        <div className="flex items-start">
                          <FileText
                            size={16}
                            className="mr-2 mt-1 flex-shrink-0 group-hover:text-vault-amber"
                          />
                          <div>
                            <p className="text-lg group-hover:text-vault-amber leading-tight">
                              {doc.title}
                            </p>
                            <div className="flex gap-x-3 text-sm text-vault-green-dim opacity-70">
                              <span>{doc.effectiveDate}</span>
                              {doc.status === 'ARCHIVED' && (
                                <span className="text-vault-amber">[ARCHIVED]</span>
                              )}
                            </div>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
