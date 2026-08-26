import React, { createContext, useState, useContext, useCallback } from 'react';

/**
 * AppContext — Global State Vault
 * ─────────────────────────────────────────────────────────────────────────────
 * Stores the three core primitives captured at onboarding:
 *   resumeFile          — raw File object (PDF) from the dropzone
 *   extractedResumeText — plain text string returned by /resume/upload (PyMuPDF)
 *   initialTargetRole   — the role string entered by the user
 *   isOnboarded         — boolean gate; when true the main app is shown
 *
 * DESIGN PRINCIPLE: No LLM calls happen here. This context is purely a
 * lightweight data store. Heavy AI endpoints are triggered on-demand by
 * individual module components.
 */

const AppContext = createContext(null);

export const AppProvider = ({ children }) => {
  const [resumeFile, setResumeFile] = useState(null);
  const [extractedResumeText, setExtractedResumeText] = useState('');
  const [initialTargetRole, setInitialTargetRole] = useState('');
  const [isOnboarded, setIsOnboarded] = useState(false);

  const completeOnboarding = useCallback((file, text, role) => {
    setResumeFile(file);
    setExtractedResumeText(text);
    setInitialTargetRole(role);
    setIsOnboarded(true);
  }, []);

  const resetOnboarding = useCallback(() => {
    setResumeFile(null);
    setExtractedResumeText('');
    setInitialTargetRole('');
    setIsOnboarded(false);
  }, []);

  return (
    <AppContext.Provider
      value={{
        resumeFile,
        extractedResumeText,
        initialTargetRole,
        isOnboarded,
        completeOnboarding,
        resetOnboarding,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used inside <AppProvider>');
  return ctx;
};
