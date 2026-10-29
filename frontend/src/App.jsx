import React, { useState, useEffect } from 'react';
import { ToastProvider } from './components/ui';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import CaseDetails from './pages/CaseDetails';

export default function App() {
  const [user, setUser] = useState(null);
  const [selectedCase, setSelectedCase] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    if (savedToken && savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch {
        localStorage.clear();
      }
    }
    setLoading(false);
  }, []);

  const handleLoginSuccess = (data) => {
    setUser({ id: data.user_id, email: data.email, role: data.role });
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setSelectedCase(null);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span className="w-3 h-3 border-2 border-current border-indigo-600 rounded-full animate-spin-slow" />
          Initializing DermAI...
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <ToastProvider>
        <Login onLoginSuccess={handleLoginSuccess} />
      </ToastProvider>
    );
  }

  if (selectedCase) {
    return (
      <ToastProvider>
        <CaseDetails
          caseItem={selectedCase}
          onBack={() => setSelectedCase(null)}
        />
      </ToastProvider>
    );
  }

  return (
    <ToastProvider>
      <Dashboard
        user={user}
        onSelectCase={(c) => setSelectedCase(c)}
        onLogout={handleLogout}
      />
    </ToastProvider>
  );
}
