import React, {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
} from 'react';

/**
 * AssessmentContext
 * ─────────────────────────────────────────────────────────────────────────────
 * Centralised state and streaming consumer for the dual-module assessment
 * pipeline.
 *
 * SSE over POST
 * ─────────────
 * The standard EventSource API only supports GET requests.  We instead use the
 * Fetch API + ReadableStream to consume the POST /api/assessment/stream SSE
 * endpoint.  Each chunk from the server arrives as a UTF-8 text block
 * containing one or more complete SSE frames in the format:
 *
 *     event: <name>\n
 *     data: <json>\n
 *     \n
 *
 * A lightweight line-based parser accumulates chunks until it finds a full
 * event frame and dispatches the appropriate state update immediately.
 *
 * State shape
 * ───────────
 *   activeModule           'technical' | 'aptitude' | null
 *   technicalQuestions     Array of question objects from the backend
 *   aptitudeQuestions      Array of question objects from the backend
 *   isLoadingActive        true while waiting for the active module to resolve
 *   isPrefetchingSecondary true while the secondary module is still arriving
 *   streamError            string | null — last error message
 */

const API_BASE = 'http://127.0.0.1:8080';

const AssessmentContext = createContext(null);

export const AssessmentProvider = ({ children }) => {
  // ── Core state ────────────────────────────────────────────────────────────
  const [activeModule, setActiveModule] = useState(null);
  const [technicalQuestions, setTechnicalQuestions] = useState([]);
  const [aptitudeQuestions, setAptitudeQuestions] = useState([]);
  const [isLoadingActive, setIsLoadingActive] = useState(false);
  const [isPrefetchingSecondary, setIsPrefetchingSecondary] = useState(false);
  const [streamError, setStreamError] = useState(null);

  // Abort controller ref — lets us cancel an in-flight stream when the user
  // launches a new assessment before the previous one has finished.
  const abortRef = useRef(null);

  /**
   * parseSSEChunk
   * ─────────────
   * Accumulates raw bytes from the stream reader into an internal line buffer.
   * When a complete "event + data + blank line" frame is detected it fires a
   * callback.  Leftover partial data is returned so it can be prepended to the
   * next chunk.
   *
   * @param {string}   rawChunk  — UTF-8 decoded text from the reader
   * @param {string}   carry     — incomplete frame carried over from the last chunk
   * @param {Function} onEvent   — called with (eventName: string, payload: any)
   * @returns {string} remaining partial frame (carry for next call)
   */
  function parseSSEChunk(rawChunk, carry, onEvent) {
    const combined = carry + rawChunk;
    // SSE frames are separated by a blank line (\n\n)
    const frames = combined.split('\n\n');
    // The last element may be an incomplete frame — carry it forward
    const incomplete = frames.pop() ?? '';

    for (const frame of frames) {
      if (!frame.trim()) continue;
      const lines = frame.split('\n');
      let eventName = 'message';
      let dataLine = '';
      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataLine = line.slice(5).trim();
        }
      }
      if (!dataLine) continue;
      let payload;
      try {
        payload = JSON.parse(dataLine);
      } catch {
        payload = dataLine;
      }
      onEvent(eventName, payload);
    }

    return incomplete;
  }

  /**
   * startAssessment
   * ───────────────
   * Kicks off the streaming assessment pipeline.
   *
   * @param {string} activeModuleType  'technical' | 'aptitude'
   * @param {string} role              Target role from AppContext
   * @param {object} skills            { green: [], yellow: [], red: [] }
   */
  const startAssessment = useCallback(
    async (activeModuleType, role, skills = {}) => {
      // Cancel any previous in-flight stream
      if (abortRef.current) {
        abortRef.current.abort();
      }
      const controller = new AbortController();
      abortRef.current = controller;

      // Reset + prime state before the first byte arrives
      setActiveModule(activeModuleType);
      setTechnicalQuestions([]);
      setAptitudeQuestions([]);
      setStreamError(null);
      setIsLoadingActive(true);
      setIsPrefetchingSecondary(true);

      const body = JSON.stringify({
        active_module:  activeModuleType,
        role:           role ?? null,
        green_skills:   skills.green  ?? [],
        yellow_skills:  skills.yellow ?? [],
        red_skills:     skills.red    ?? [],
      });

      try {
        const response = await fetch(`${API_BASE}/api/assessment/stream`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
          body,
          signal: controller.signal,
        });

        if (!response.ok) {
          const errText = await response.text().catch(() => response.statusText);
          throw new Error(`Stream request failed (${response.status}): ${errText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let carry = '';

        // ── Event handler ──────────────────────────────────────────────────
        const handleEvent = (eventName, payload) => {
          switch (eventName) {
            case 'active_module_ready': {
              const qs = payload?.questions ?? [];
              if (activeModuleType === 'technical') {
                setTechnicalQuestions(qs);
              } else {
                setAptitudeQuestions(qs);
              }
              // Active module resolved → UI can render quiz immediately
              setIsLoadingActive(false);
              break;
            }

            case 'secondary_module_ready': {
              const qs = payload?.questions ?? [];
              const secondaryModule =
                activeModuleType === 'technical' ? 'aptitude' : 'technical';
              if (secondaryModule === 'technical') {
                setTechnicalQuestions(qs);
              } else {
                setAptitudeQuestions(qs);
              }
              break;
            }

            case 'stream_complete': {
              setIsPrefetchingSecondary(false);
              break;
            }

            default:
              break;
          }
        };

        // ── Read loop ──────────────────────────────────────────────────────
        let done = false;
        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          if (value) {
            const chunk = decoder.decode(value, { stream: !done });
            carry = parseSSEChunk(chunk, carry, handleEvent);
          }
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          // Intentionally cancelled — not an error
          return;
        }
        console.error('[AssessmentContext] Stream error:', err);
        setStreamError(err.message ?? 'Unknown streaming error');
        setIsLoadingActive(false);
        setIsPrefetchingSecondary(false);
      }
    },
    [],
  );

  /**
   * resetAssessment
   * ───────────────
   * Clears all assessment state and cancels any in-flight stream.
   */
  const resetAssessment = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setActiveModule(null);
    setTechnicalQuestions([]);
    setAptitudeQuestions([]);
    setStreamError(null);
    setIsLoadingActive(false);
    setIsPrefetchingSecondary(false);
  }, []);

  const value = {
    // ── State ────────────────────────────────────────────────────────────
    activeModule,
    technicalQuestions,
    aptitudeQuestions,
    isLoadingActive,
    isPrefetchingSecondary,
    streamError,
    // ── Actions ──────────────────────────────────────────────────────────
    startAssessment,
    resetAssessment,
  };

  return (
    <AssessmentContext.Provider value={value}>
      {children}
    </AssessmentContext.Provider>
  );
};

/**
 * useAssessment
 * ─────────────
 * Custom hook for consuming AssessmentContext. Throws if used outside the
 * provider tree so mis-wiring is caught at development time.
 */
export const useAssessment = () => {
  const ctx = useContext(AssessmentContext);
  if (!ctx) {
    throw new Error('useAssessment must be used inside <AssessmentProvider>');
  }
  return ctx;
};
