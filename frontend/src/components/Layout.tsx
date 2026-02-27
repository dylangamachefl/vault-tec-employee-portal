export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col h-screen w-screen bg-vault-bg text-vault-green crt relative overflow-hidden">
      {/* Global Title Bar */}
      <header className="flex-none border-b-2 border-vault-green p-4 flex justify-between items-center bg-vault-green-dark/20">
        <h1 className="text-2xl md:text-3xl uppercase tracking-widest font-bold">
          VAULT-TEC CORPORATION — INTERNAL KNOWLEDGE SYSTEM
        </h1>
        <div className="animate-pulse text-vault-amber text-xl font-bold border border-vault-amber px-2 py-1">
          PHASE 1 DEMO
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 overflow-hidden relative">
        {children}
      </main>

      {/* Global Footer */}
      <footer className="flex-none border-t-2 border-vault-green p-2 text-center text-lg bg-vault-green-dark/20">
        <p>VAULT-TEC CORP. | EST. 2031 | BUILDING A BETTER TOMORROW, TODAY™ | AUTHORIZED PERSONNEL ONLY</p>
        <p className="text-vault-amber text-sm mt-1">
          WARNING: This is a Phase 1 Demo. Backend access-level restrictions are not yet enforced.
        </p>
      </footer>
    </div>
  );
}
