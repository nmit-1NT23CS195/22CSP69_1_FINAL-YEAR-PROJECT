import React, { useState } from 'react';
import PlaceBuddyDashboard from './PlaceBuddyDashboard';
import Dashboard from './pages/Dashboard';
import Onboarding from './pages/Onboarding';
import Quiz from './components/mcq/Quiz';
import { QuizProvider } from './context/QuizContext';
import { AppProvider, useApp } from './context/AppContext';

// ── Inner app — only rendered once onboarding is complete ────────────────────
function AppInner() {
  const { isOnboarded } = useApp();
  const [view, setView] = useState('dashboard');

  // Gate: show onboarding until resume + role are captured
  if (!isOnboarded) {
    return <Onboarding />;
  }

  const renderView = () => {
    switch (view) {
      case 'dashboard':
        return <Dashboard onViewChange={setView} />;

      case 'ats':
        return (
          <div className="relative min-h-screen bg-[#050508]">
            {/* Floating back button — always accessible */}
            <button
              onClick={() => setView('dashboard')}
              className="fixed top-5 left-5 z-[9999] flex items-center gap-2 bg-zinc-900/80 backdrop-blur-md text-zinc-300 hover:text-white px-4 py-2 rounded-xl border border-zinc-800 hover:border-zinc-700 text-sm font-semibold transition-all shadow-xl"
            >
              ← Dashboard
            </button>
            <PlaceBuddyDashboard />
          </div>
        );

      case 'mcq':
        // Quiz manages its own full-screen layout — just pass onBackToDashboard
        return <Quiz onBackToDashboard={() => setView('dashboard')} />;

      default:
        return <Dashboard onViewChange={setView} />;
    }
  };

  return renderView();
}

// ── Root — providers wrapping order: AppProvider (outer) > QuizProvider ───────
export default function App() {
  return (
    <AppProvider>
      <QuizProvider>
        <AppInner />
      </QuizProvider>
    </AppProvider>
  );
}
