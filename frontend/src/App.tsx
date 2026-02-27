import { useState } from 'react';
import { User } from './types';
import { Layout } from './components/Layout';
import { LoginView } from './components/LoginView';
import { ChatView } from './components/ChatView';

export default function App() {
  const [currentUser, setCurrentUser] = useState<User | null>(null);

  return (
    <Layout>
      {currentUser ? (
        <ChatView user={currentUser} onLogout={() => setCurrentUser(null)} />
      ) : (
        <LoginView onLogin={setCurrentUser} />
      )}
    </Layout>
  );
}
