import React, { createContext, useState, useContext, useCallback } from 'react';

const QuizContext = createContext();

export const QuizProvider = ({ children }) => {
  // Q10 metric: raw score from the 10-MCQ assessment (0–100)
  const [q10Score, setQ10ScoreRaw] = useState(() => {
    try {
      const stored = localStorage.getItem('placebuddy_q10_score');
      return stored ? parseFloat(stored) : 0;
    } catch {
      return 0;
    }
  });

  // Persist Q10 score to localStorage on every update
  const setQ10Score = useCallback((score) => {
    const clamped = Math.max(0, Math.min(100, score));
    setQ10ScoreRaw(clamped);
    try {
      localStorage.setItem('placebuddy_q10_score', String(clamped));
    } catch (_) {}
  }, []);

  // Computed PRS — extendable when ATS score is wired in
  const prsScore = q10Score; // Future: weighted average with ATS score

  const prsGrade =
    prsScore >= 90 ? 'Elite'
    : prsScore >= 70 ? 'Proficient'
    : prsScore >= 50 ? 'Developing'
    : prsScore > 0 ? 'Needs Work'
    : 'Not Assessed';

  const resetQ10 = useCallback(() => {
    setQ10ScoreRaw(0);
    try { localStorage.removeItem('placebuddy_q10_score'); } catch (_) {}
  }, []);

  return (
    <QuizContext.Provider value={{ q10Score, setQ10Score, prsScore, prsGrade, resetQ10 }}>
      {children}
    </QuizContext.Provider>
  );
};

export const useQuiz = () => {
  const ctx = useContext(QuizContext);
  if (!ctx) throw new Error('useQuiz must be used inside <QuizProvider>');
  return ctx;
};
