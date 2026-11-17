/**
 * VideoInterview.jsx — Module D: Multimodal Behavioral Analytics
 * ================================================================
 * A self-contained state-machine component that:
 *   1. Requests camera + mic permissions via getUserMedia.
 *   2. Fetches 5 questions from the backend for the candidate's role.
 *   3. Steps through Prep (30s) → Record (120s max) for each question.
 *   4. Collects 5 video Blobs, assembles the FormData per the API contract,
 *      and POSTs to /api/interview/submit.
 *   5. Renders a full results view with per-question feedback.
 *
 * Designed to slot cleanly into the existing PlaceBuddy dark theme — no
 * external CSS imports needed; all styling is Tailwind utility classes.
 */

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  AlertCircle,
  Camera,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock,
  Loader2,
  Mic,
  MicOff,
  RefreshCcw,
  Square,
  TrendingUp,
  Video,
  VideoOff,
  X,
} from 'lucide-react';

// ── Constants ──────────────────────────────────────────────────────────────────
const API_BASE        = 'http://127.0.0.1:8080';
const PREP_SECONDS    = 30;   // Phase 2: thinking time per question
const RECORD_SECONDS  = 120;  // Phase 3: max recording time per question
// TOTAL_QUESTIONS is now dynamic — set from the API response length.
// This constant is kept only for the UI info callout text.
const DEFAULT_QUESTION_COUNT = 5;

// ── State machine phases ───────────────────────────────────────────────────────
const PHASE = {
  SETUP:      'SETUP',      // user selects difficulty before anything starts
  INIT:       'INIT',       // requesting camera permissions / fetching questions
  PREP:       'PREP',       // countdown before recording starts
  RECORDING:  'RECORDING',  // MediaRecorder is active
  TRANSITION: 'TRANSITION', // brief pause between questions
  SUBMITTING: 'SUBMITTING', // uploading to backend
  RESULTS:    'RESULTS',    // showing evaluation report
  ERROR:      'ERROR',      // unrecoverable error
};

// ── Difficulty config ──────────────────────────────────────────────────────────
const DIFFICULTY_OPTIONS = [
  {
    value: 'easy',
    label: 'Easy',
    description: 'Foundational questions · 0.85× score multiplier',
    accent: 'text-emerald-400',
    border: 'border-emerald-500/40',
    bg: 'bg-emerald-500/10',
    ring: 'ring-emerald-500/40',
    badge: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/25',
  },
  {
    value: 'medium',
    label: 'Medium',
    description: 'Standard placement-level · 1.00× score multiplier',
    accent: 'text-amber-400',
    border: 'border-amber-500/40',
    bg: 'bg-amber-500/10',
    ring: 'ring-amber-500/40',
    badge: 'bg-amber-500/15 text-amber-300 border-amber-500/25',
  },
  {
    value: 'hard',
    label: 'Hard',
    description: 'Senior-level depth · 1.15× score multiplier',
    accent: 'text-rose-400',
    border: 'border-rose-500/40',
    bg: 'bg-rose-500/10',
    ring: 'ring-rose-500/40',
    badge: 'bg-rose-500/15 text-rose-300 border-rose-500/25',
  },
];

// ── Helpers ────────────────────────────────────────────────────────────────────
function formatTime(secs) {
  const m = Math.floor(secs / 60).toString().padStart(2, '0');
  const s = (secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function scoreColor(score) {
  if (score >= 75) return 'text-emerald-400';
  if (score >= 50) return 'text-amber-400';
  return 'text-red-400';
}

function scoreBarColor(score) {
  if (score >= 75) return 'from-emerald-500 to-cyan-500';
  if (score >= 50) return 'from-amber-500 to-orange-400';
  return 'from-red-500 to-pink-500';
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function Countdown({ seconds, total, label, accent }) {
  const pct = total > 0 ? ((total - seconds) / total) * 100 : 0;
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={`text-6xl font-black tabular-nums tracking-tighter ${accent}`}
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {formatTime(seconds)}
      </div>
      <p className={`text-xs font-bold uppercase tracking-widest ${accent}/70`}>{label}</p>
      <div className="w-48 h-1 bg-white/10 rounded-full overflow-hidden mt-1">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${
            accent.includes('amber') ? 'from-amber-400 to-orange-400' :
            accent.includes('red')   ? 'from-red-500 to-pink-500' :
                                       'from-indigo-400 to-purple-400'
          } transition-all duration-1000`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function PermissionDeniedCard({ onRetry }) {
  return (
    <div className="flex flex-col items-center gap-5 py-10 px-6 text-center">
      <div className="w-16 h-16 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center">
        <VideoOff className="w-8 h-8 text-red-400" />
      </div>
      <div>
        <h3 className="text-lg font-black text-white mb-1">Camera Access Required</h3>
        <p className="text-sm text-zinc-400 max-w-sm">
          Please allow camera and microphone access in your browser settings, then try again.
        </p>
      </div>
      <button
        onClick={onRetry}
        className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 font-bold text-sm hover:bg-indigo-500/30 transition-colors"
      >
        <RefreshCcw className="w-4 h-4" /> Retry Permissions
      </button>
    </div>
  );
}

function QuestionCard({ question, index, total }) {
  return (
    <div className="relative p-6 rounded-2xl border border-white/8 bg-white/[0.03] backdrop-blur-sm">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-[10px] font-bold text-rose-400/80 uppercase tracking-[0.15em] bg-rose-500/10 px-2 py-0.5 rounded-full border border-rose-500/20">
          Q{index + 1} / {total}
        </span>
        <span className="text-[10px] font-semibold text-zinc-600 uppercase tracking-wider">
          {question.question_type} · {question.difficulty}
        </span>
      </div>
      <p className="text-white font-semibold text-base leading-relaxed">
        {question.question_text}
      </p>
    </div>
  );
}

function ResultCard({ evaluation, questionText, index, acoustics }) {
  const score = evaluation?.weighted_score ?? 0;
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.03] overflow-hidden">
      {/* Header row */}
      <button
        className="w-full flex items-center gap-4 p-5 text-left hover:bg-white/[0.02] transition-colors"
        onClick={() => setExpanded(v => !v)}
      >
        <div className="shrink-0 w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
          <span className="text-xs font-black text-zinc-400">Q{index + 1}</span>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-zinc-300 font-medium truncate pr-4">{questionText}</p>
          <div className="mt-1.5 h-1 bg-white/5 rounded-full overflow-hidden w-full">
            <div
              className={`h-full rounded-full bg-gradient-to-r ${scoreBarColor(score)} transition-all duration-700`}
              style={{ width: `${(score / 10) * 100}%` }}
            />
          </div>
        </div>
        <div className={`shrink-0 text-2xl font-black tabular-nums ${scoreColor(score)}`}>
          {score.toFixed(1)}
          <span className="text-xs text-zinc-600 font-normal">/100</span>
        </div>
        <ChevronRight
          className={`shrink-0 w-4 h-4 text-zinc-600 transition-transform duration-300 ${expanded ? 'rotate-90' : ''}`}
        />
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-5 pb-5 space-y-4 border-t border-white/5 pt-4">
          {/* Dimension scores */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Technical',    val: evaluation?.technical_correctness },
              { label: 'Relevance',    val: evaluation?.relevance },
              { label: 'Completeness', val: evaluation?.completeness },
              { label: 'STAR Structure', val: evaluation?.star_structure_score },
            ].map(({ label, val }) => (
              <div key={label} className="bg-white/[0.03] rounded-xl p-3 border border-white/5">
                <p className="text-[10px] text-zinc-500 uppercase tracking-widest font-bold mb-1">{label}</p>
                <p className={`text-lg font-black tabular-nums ${scoreColor(val ?? 0)}`}>
                  {(val ?? 0).toFixed(1)}<span className="text-xs text-zinc-600 font-normal">/100</span>
                </p>
              </div>
            ))}
          </div>

          {/* AI Feedback */}
          {evaluation?.feedback && (
            <div className="bg-indigo-500/5 border border-indigo-500/15 rounded-xl p-4">
              <p className="text-[10px] font-bold text-indigo-400/70 uppercase tracking-widest mb-2">AI Feedback</p>
              <p className="text-sm text-zinc-300 leading-relaxed">{evaluation.feedback}</p>
            </div>
          )}

          {/* Strengths / Improvements */}
          <div className="grid grid-cols-2 gap-3">
            {evaluation?.strengths?.length > 0 && (
              <div>
                <p className="text-[10px] font-bold text-emerald-400/70 uppercase tracking-widest mb-2">Strengths</p>
                <ul className="space-y-1">
                  {evaluation.strengths.slice(0, 3).map((s, i) => (
                    <li key={i} className="text-xs text-zinc-400 flex items-start gap-1.5">
                      <CheckCircle2 className="w-3 h-3 text-emerald-500 mt-0.5 shrink-0" />{s}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {evaluation?.improvements?.length > 0 && (
              <div>
                <p className="text-[10px] font-bold text-amber-400/70 uppercase tracking-widest mb-2">To Improve</p>
                <ul className="space-y-1">
                  {evaluation.improvements.slice(0, 3).map((s, i) => (
                    <li key={i} className="text-xs text-zinc-400 flex items-start gap-1.5">
                      <TrendingUp className="w-3 h-3 text-amber-500 mt-0.5 shrink-0" />{s}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Acoustic snapshot */}
          {acoustics && Object.keys(acoustics).length > 0 && (
            <div className="border-t border-white/5 pt-3">
              <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-2">Acoustic Metrics</p>
              <div className="flex flex-wrap gap-2">
                {acoustics.wpm != null && (
                  <span className="text-[10px] bg-white/5 border border-white/8 rounded-lg px-2 py-1 text-zinc-400 font-mono">
                    {acoustics.wpm} WPM
                  </span>
                )}
                {acoustics.pause_count != null && (
                  <span className="text-[10px] bg-white/5 border border-white/8 rounded-lg px-2 py-1 text-zinc-400 font-mono">
                    {acoustics.pause_count} pauses
                  </span>
                )}
                {acoustics.filler_word_density != null && (
                  <span className="text-[10px] bg-white/5 border border-white/8 rounded-lg px-2 py-1 text-zinc-400 font-mono">
                    {(acoustics.filler_word_density * 100).toFixed(1)}% fillers
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────
export default function VideoInterview({ targetRole = 'Software Engineer', onBack }) {
  // ── State ──────────────────────────────────────────────────────────────────
  const [phase, setPhase]                       = useState(PHASE.SETUP);
  const [selectedDifficulty, setSelectedDifficulty] = useState('medium');
  const [questions, setQuestions]               = useState([]);
  const [currentQ, setCurrentQ]                 = useState(0);
  const [timer, setTimer]                       = useState(0);
  const [blobs, setBlobs]                       = useState([]);
  const [isFetching, setIsFetching]             = useState(false);
  const [setupError, setSetupError]             = useState('');   // shown inline on Setup screen
  const [permError, setPermError]               = useState(false);
  const [fetchError, setFetchError]             = useState('');
  const [submitError, setSubmitError]           = useState('');
  const [results, setResults]                   = useState(null);
  const [isMuted, setIsMuted]                   = useState(false);
  const [isCamOff, setIsCamOff]                 = useState(false);

  // ── Refs ───────────────────────────────────────────────────────────────────
  const videoRef      = useRef(null);    // live preview <video>
  const streamRef     = useRef(null);    // MediaStream
  const recorderRef   = useRef(null);    // MediaRecorder
  const chunksRef     = useRef([]);      // recorded data chunks
  const timerRef      = useRef(null);    // setInterval handle

  // ── Cleanup helpers ────────────────────────────────────────────────────────
  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  // ── On unmount — release camera ────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopTimer();
      stopStream();
      if (recorderRef.current?.state === 'recording') {
        recorderRef.current.stop();
      }
    };
  }, [stopTimer, stopStream]);

  // ── Step 1: request permissions + fetch questions ──────────────────────────
  const initSession = useCallback(async () => {
    setPhase(PHASE.INIT);
    setPermError(false);
    setFetchError('');
    setSetupError('');
    setIsFetching(true);
    setBlobs([]);
    setCurrentQ(0);

    // 1a. Camera + mic
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    } catch {
      setPermError(true);
      setPhase(PHASE.ERROR);
      setIsFetching(false);
      return;
    }
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
    }

    // 1b. Fetch questions — include difficulty param
    try {
      const url = new URL(`${API_BASE}/api/interview/questions`);
      url.searchParams.set('role', targetRole);
      url.searchParams.set('difficulty', selectedDifficulty);

      const res = await fetch(url.toString());
      console.log('[VideoInterview] GET questions response:', res.status, res.statusText);

      if (!res.ok) throw new Error(`HTTP ${res.status} — ${res.statusText}`);

      const data = await res.json();
      console.log('[VideoInterview] Raw questions payload:', data);

      const qs = data?.questions ?? [];

      if (!qs || qs.length === 0) {
        // Surface a friendly error on the Setup screen — don't go to ERROR phase.
        setSetupError('Error: Database returned 0 questions for this role. Please run the seeder or try a different role/difficulty.');
        setPhase(PHASE.SETUP);
        stopStream();
        setIsFetching(false);
        return;
      }

      // ✅ Accept any number of questions — no strict == 5 requirement.
      console.log(`[VideoInterview] Loaded ${qs.length} questions for role="${targetRole}" difficulty="${selectedDifficulty}"`);
      setQuestions(qs);
      setIsFetching(false);
      startPrep(0);
    } catch (err) {
      console.error('[VideoInterview] Fetch error:', err);
      const msg = err?.message ?? 'Failed to fetch questions.';
      setSetupError(`Error: ${msg}`);
      setFetchError(msg);
      setPhase(PHASE.SETUP);
      stopStream();
      setIsFetching(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetRole, selectedDifficulty]);

  // Only auto-init after the SETUP phase is committed — NOT on first mount.
  // initSession is called explicitly by the "Start Interview" button.

  // ── Attach stream to video element whenever it's available ─────────────────
  useEffect(() => {
    if (videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  });

  // ── Phase transitions ──────────────────────────────────────────────────────
  const startPrep = useCallback((qIndex) => {
    stopTimer();
    setCurrentQ(qIndex);
    setPhase(PHASE.PREP);
    setTimer(PREP_SECONDS);

    timerRef.current = setInterval(() => {
      setTimer(prev => {
        if (prev <= 1) {
          clearInterval(timerRef.current);
          timerRef.current = null;
          startRecording(qIndex);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startRecording = useCallback((qIndex) => {
    stopTimer();
    if (!streamRef.current) return;

    chunksRef.current = [];
    const recorder = new MediaRecorder(streamRef.current, {
      mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
        ? 'video/webm;codecs=vp9,opus'
        : 'video/webm',
    });

    recorder.ondataavailable = e => {
      if (e.data?.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      setBlobs(prev => {
        const updated = [...prev];
        updated[qIndex] = blob;
        return updated;
      });
      // move to next question or submit — use questions.length, not hardcoded 5
      setQuestions(currentQs => {
        const totalQs = currentQs.length;
        const nextQ = qIndex + 1;
        if (nextQ < totalQs) {
          setPhase(PHASE.TRANSITION);
          setTimeout(() => startPrep(nextQ), 1200);
        } else {
          setPhase(PHASE.SUBMITTING);
        }
        return currentQs; // no change to questions array
      });
    };

    recorder.start(250); // collect chunks every 250 ms
    recorderRef.current = recorder;

    setPhase(PHASE.RECORDING);
    setTimer(RECORD_SECONDS);

    timerRef.current = setInterval(() => {
      setTimer(prev => {
        if (prev <= 1) {
          clearInterval(timerRef.current);
          timerRef.current = null;
          stopRecording();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopRecording = useCallback(() => {
    stopTimer();
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop();
    }
  }, [stopTimer]);

  // ── Watch blobs to trigger submission once all questions are recorded ─────
  useEffect(() => {
    const totalQs = questions.length;
    if (
      phase === PHASE.SUBMITTING &&
      totalQs > 0 &&
      blobs.filter(Boolean).length === totalQs
    ) {
      submitInterview(blobs, questions);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, blobs, questions]);

  const submitInterview = async (recordedBlobs, qs) => {
    setSubmitError('');
    const form = new FormData();

    // question_ids as JSON string
    form.append('question_ids', JSON.stringify(qs.map(q => q.id)));
    form.append('target_role', targetRole);
    form.append('difficulty', selectedDifficulty);

    // video_1 … video_5  (matches backend param names exactly)
    recordedBlobs.forEach((blob, i) => {
      form.append(`video_${i + 1}`, blob, `answer_${i + 1}.webm`);
    });

    try {
      const res = await fetch(`${API_BASE}/api/interview/submit`, {
        method: 'POST',
        body: form,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Server error ${res.status}: ${txt}`);
      }
      const data = await res.json();
      setResults(data);
      stopStream();
      setPhase(PHASE.RESULTS);
    } catch (err) {
      setSubmitError(err.message ?? 'Submission failed. Please try again.');
      setPhase(PHASE.ERROR);
    }
  };

  // ── Camera / mic toggles (non-critical UX) ─────────────────────────────────
  const toggleMute = () => {
    const audioTrack = streamRef.current?.getAudioTracks()[0];
    if (!audioTrack) return;
    audioTrack.enabled = !audioTrack.enabled;
    setIsMuted(!audioTrack.enabled);
  };

  const toggleCam = () => {
    const videoTrack = streamRef.current?.getVideoTracks()[0];
    if (!videoTrack) return;
    videoTrack.enabled = !videoTrack.enabled;
    setIsCamOff(!videoTrack.enabled);
  };

  // ── Derived values ─────────────────────────────────────────────────────────
  const totalQuestions = questions.length || DEFAULT_QUESTION_COUNT;
  const currentQuestion = questions[currentQ];
  const progressPct = useMemo(
    () => (blobs.filter(Boolean).length / totalQuestions) * 100,
    [blobs, totalQuestions]
  );

  // ══════════════════════════════════════════════════════════════════════════
  //  RENDER
  // ══════════════════════════════════════════════════════════════════════════
  return (
    <div className="relative min-h-screen bg-[#050508] overflow-hidden flex flex-col">
      {/* Background glow */}
      <div
        className="absolute rounded-full pointer-events-none"
        style={{
          background: 'radial-gradient(circle, #be185d, transparent)',
          width: '600px', height: '600px',
          top: '-150px', left: '-100px',
          filter: 'blur(100px)', opacity: 0.12,
        }}
      />
      <div
        className="absolute rounded-full pointer-events-none"
        style={{
          background: 'radial-gradient(circle, #6366f1, transparent)',
          width: '400px', height: '400px',
          bottom: '-50px', right: '-50px',
          filter: 'blur(80px)', opacity: 0.1,
        }}
      />

      {/* ── NAV ──────────────────────────────────────────────────────────── */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-4 border-b border-white/5 backdrop-blur-xl bg-white/[0.02]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-rose-500 to-pink-600 flex items-center justify-center shadow-[0_0_16px_rgba(244,63,94,0.4)]">
            <Video className="w-4 h-4 text-white" />
          </div>
          <div>
            <span className="text-sm font-black text-white tracking-tight">Module D</span>
            <span className="text-xs text-zinc-500 font-mono ml-2">· Behavioral Analytics</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Overall progress bar */}
          <div className="hidden sm:flex items-center gap-2">
            <span className="text-[10px] font-mono text-zinc-600">
              {blobs.filter(Boolean).length}/{totalQuestions} recorded
            </span>
            <div className="w-28 h-1.5 bg-white/8 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-rose-500 to-pink-400 rounded-full transition-all duration-500"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 text-xs font-semibold text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <X className="w-4 h-4" /> Exit
          </button>
        </div>
      </nav>

      {/* ── BODY ─────────────────────────────────────────────────────────── */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-start px-4 py-8 max-w-5xl mx-auto w-full gap-6">

        {/* ── PHASE: SETUP — difficulty selector ────────────────────────── */}
        {phase === PHASE.SETUP && (
          <div className="w-full max-w-lg mx-auto mt-16 flex flex-col gap-6">
            {/* Header */}
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-rose-500/10 border border-rose-500/20 mb-4">
                <Video className="w-7 h-7 text-rose-400" />
              </div>
              <h2 className="text-2xl font-black text-white tracking-tight mb-1">
                Video Mock Interview
              </h2>
              <p className="text-sm text-zinc-500">
                Target: <span className="text-zinc-300 font-semibold">{targetRole}</span>
              </p>
            </div>

            {/* Difficulty selector */}
            <div>
              <p className="text-xs font-bold text-zinc-500 uppercase tracking-widest mb-3 px-1">
                Select Difficulty
              </p>
              <div className="flex flex-col gap-2.5">
                {DIFFICULTY_OPTIONS.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => setSelectedDifficulty(opt.value)}
                    className={`
                      relative w-full flex items-center gap-4 p-4 rounded-2xl border text-left
                      transition-all duration-200 focus:outline-none focus:ring-2 ${opt.ring}
                      ${selectedDifficulty === opt.value
                        ? `${opt.border} ${opt.bg}`
                        : 'border-white/8 bg-white/[0.02] hover:bg-white/[0.04]'
                      }
                    `}
                  >
                    {/* Selection indicator */}
                    <div className={`
                      shrink-0 w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all
                      ${selectedDifficulty === opt.value
                        ? `${opt.border} ${opt.bg}`
                        : 'border-zinc-700'
                      }
                    `}>
                      {selectedDifficulty === opt.value && (
                        <div className={`w-2 h-2 rounded-full ${opt.accent.replace('text-', 'bg-')}`} />
                      )}
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className={`text-sm font-black ${selectedDifficulty === opt.value ? opt.accent : 'text-zinc-300'}`}>
                          {opt.label}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-500">{opt.description}</p>
                    </div>

                    {/* Multiplier badge */}
                    <span className={`shrink-0 text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                      selectedDifficulty === opt.value ? opt.badge : 'bg-white/5 text-zinc-600 border-white/8'
                    }`}>
                      {opt.value === 'easy' ? '0.85×' : opt.value === 'hard' ? '1.15×' : '1.00×'}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Info callout */}
            <div className="flex items-start gap-3 bg-indigo-500/5 border border-indigo-500/15 rounded-2xl px-4 py-3">
              <span className="text-indigo-400 text-sm mt-0.5">ℹ</span>
              <p className="text-xs text-zinc-400 leading-relaxed">
                You will answer <strong className="text-zinc-200">5 questions</strong> with
                <strong className="text-zinc-200"> 30 seconds</strong> to prepare and
                <strong className="text-zinc-200"> 2 minutes</strong> to record each answer.
                Your final score is adjusted by the difficulty multiplier.
              </p>
            </div>

            {/* Inline error — shown if fetch failed without leaving Setup screen */}
            {setupError && (
              <div className="flex items-start gap-3 bg-red-500/8 border border-red-500/25 rounded-2xl px-4 py-3">
                <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
                <p className="text-xs text-red-300 leading-relaxed break-words">{setupError}</p>
              </div>
            )}

            {/* CTA */}
            <button
              onClick={initSession}
              disabled={isFetching}
              className="w-full flex items-center justify-center gap-2 py-3.5 rounded-2xl bg-gradient-to-r from-rose-600 to-pink-600 text-white font-black text-sm shadow-[0_0_30px_rgba(244,63,94,0.3)] hover:shadow-[0_0_40px_rgba(244,63,94,0.5)] transition-all duration-300 hover:scale-[1.02] active:scale-100 disabled:opacity-60 disabled:cursor-not-allowed disabled:scale-100 disabled:shadow-none"
            >
              {isFetching ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading…
                </>
              ) : (
                <>
                  <CircleDot className="w-4 h-4 animate-pulse" />
                  Start Interview — {DIFFICULTY_OPTIONS.find(o => o.value === selectedDifficulty)?.label}
                </>
              )}
            </button>
          </div>
        )}

        {/* ── PHASE: INIT (loading) ──────────────────────────────────────── */}
        {phase === PHASE.INIT && (
          <div className="flex flex-col items-center gap-4 mt-24">
            <Loader2 className="w-10 h-10 text-rose-400 animate-spin" />
            <p className="text-sm font-bold text-zinc-300">Setting up interview session…</p>
            <p className="text-xs font-mono text-zinc-600">Requesting camera · Fetching questions</p>
          </div>
        )}

        {/* ── PHASE: ERROR ──────────────────────────────────────────────── */}
        {phase === PHASE.ERROR && (
          <div className="w-full max-w-lg mx-auto mt-16 p-7 rounded-3xl border border-red-500/20 bg-red-950/20 backdrop-blur-sm">
            {permError ? (
              <PermissionDeniedCard onRetry={initSession} />
            ) : (
              <div className="flex flex-col items-center gap-4 text-center">
                <AlertCircle className="w-10 h-10 text-red-400" />
                <div>
                  <h3 className="text-base font-black text-white mb-1">Session Error</h3>
                  <p className="text-sm text-zinc-400 font-mono break-all">
                    {fetchError || submitError || 'Unknown error. Please try again.'}
                  </p>
                </div>
                <button
                  onClick={() => { stopStream(); setPhase(PHASE.SETUP); setResults(null); setBlobs([]); }}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-500/20 border border-rose-500/30 text-rose-300 font-bold text-sm hover:bg-rose-500/30 transition-colors"
                >
                  <RefreshCcw className="w-4 h-4" /> Try Again
                </button>
              </div>
            )}
          </div>
        )}

        {/* ── PHASE: PREP / RECORDING / TRANSITION ──────────────────────── */}
        {[PHASE.PREP, PHASE.RECORDING, PHASE.TRANSITION].includes(phase) && currentQuestion && (
          <div className="w-full flex flex-col lg:flex-row gap-6">

            {/* Left — video preview */}
            <div className="flex flex-col gap-4 lg:w-1/2">
              {/* Camera preview */}
              <div className="relative rounded-2xl overflow-hidden border border-white/8 bg-black aspect-video">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className={`w-full h-full object-cover transition-opacity duration-300 ${isCamOff ? 'opacity-0' : 'opacity-100'}`}
                />
                {isCamOff && (
                  <div className="absolute inset-0 flex items-center justify-center bg-zinc-900">
                    <VideoOff className="w-10 h-10 text-zinc-600" />
                  </div>
                )}

                {/* Recording dot */}
                {phase === PHASE.RECORDING && (
                  <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm rounded-full px-2.5 py-1">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-[10px] font-bold text-red-400 uppercase tracking-widest">REC</span>
                  </div>
                )}

                {/* Phase badge */}
                {phase === PHASE.PREP && (
                  <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-amber-500/20 border border-amber-500/30 backdrop-blur-sm rounded-full px-2.5 py-1">
                    <Clock className="w-3 h-3 text-amber-400" />
                    <span className="text-[10px] font-bold text-amber-400 uppercase tracking-widest">Prep</span>
                  </div>
                )}

                {/* Transition overlay */}
                {phase === PHASE.TRANSITION && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm">
                    <div className="text-center">
                      <CheckCircle2 className="w-10 h-10 text-emerald-400 mx-auto mb-2" />
                      <p className="text-sm font-bold text-white">Answer saved!</p>
                      <p className="text-xs text-zinc-400 mt-1">Loading next question…</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Media controls */}
              <div className="flex items-center justify-center gap-3">
                <button
                  onClick={toggleMute}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-colors ${
                    isMuted
                      ? 'bg-red-500/20 border-red-500/30 text-red-400'
                      : 'bg-white/5 border-white/10 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {isMuted ? <MicOff className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5" />}
                  {isMuted ? 'Unmute' : 'Mute'}
                </button>
                <button
                  onClick={toggleCam}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-colors ${
                    isCamOff
                      ? 'bg-red-500/20 border-red-500/30 text-red-400'
                      : 'bg-white/5 border-white/10 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {isCamOff ? <VideoOff className="w-3.5 h-3.5" /> : <Video className="w-3.5 h-3.5" />}
                  {isCamOff ? 'Show Cam' : 'Hide Cam'}
                </button>
              </div>
            </div>

            {/* Right — question + controls */}
            <div className="flex flex-col gap-5 lg:w-1/2 justify-center">

              {/* Question */}
              <QuestionCard
                question={currentQuestion}
                index={currentQ}
                total={totalQuestions}
              />

              {/* Timer */}
              <div className="flex flex-col items-center py-5">
                {phase === PHASE.PREP && (
                  <Countdown
                    seconds={timer}
                    total={PREP_SECONDS}
                    label="Think before you answer"
                    accent="text-amber-400"
                  />
                )}
                {phase === PHASE.RECORDING && (
                  <Countdown
                    seconds={timer}
                    total={RECORD_SECONDS}
                    label="Recording — speak clearly"
                    accent="text-red-400"
                  />
                )}
                {phase === PHASE.TRANSITION && (
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
                    <p className="text-sm font-bold text-zinc-300">
                      {currentQ + 1 < totalQuestions
                        ? `Preparing question ${currentQ + 2}…`
                        : 'All answers recorded!'}
                    </p>
                  </div>
                )}
              </div>

              {/* CTA: skip prep / stop recording */}
              <div className="flex flex-col gap-3">
                {phase === PHASE.PREP && (
                  <button
                    onClick={() => {
                      stopTimer();
                      startRecording(currentQ);
                    }}
                    className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-rose-500/20 border border-rose-500/30 text-rose-300 font-bold text-sm hover:bg-rose-500/30 transition-colors"
                  >
                    <CircleDot className="w-4 h-4 animate-pulse" />
                    Start Recording Now
                  </button>
                )}
                {phase === PHASE.RECORDING && (
                  <button
                    onClick={stopRecording}
                    className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-zinc-800 border border-zinc-700 text-zinc-300 font-bold text-sm hover:bg-zinc-700 transition-colors"
                  >
                    <Square className="w-4 h-4 text-red-400 fill-red-400" />
                    Done Answering
                  </button>
                )}
              </div>

              {/* Question pip progress */}
              <div className="flex items-center justify-center gap-2">
                {Array.from({ length: totalQuestions }).map((_, i) => (
                  <div
                    key={i}
                    className={`h-1.5 rounded-full transition-all duration-500 ${
                      i < blobs.filter(Boolean).length
                        ? 'bg-emerald-400 w-6'
                        : i === currentQ
                        ? 'bg-rose-400 w-8'
                        : 'bg-white/10 w-4'
                    }`}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── PHASE: SUBMITTING ──────────────────────────────────────────── */}
        {phase === PHASE.SUBMITTING && (
          <div className="flex flex-col items-center gap-5 mt-24 text-center">
            <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
              <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
            </div>
            <div>
              <h3 className="text-lg font-black text-white mb-1">Analysing your interview…</h3>
              <p className="text-sm text-zinc-400">
                Transcribing audio · Evaluating with AI · Computing your score
              </p>
              <p className="text-xs font-mono text-zinc-600 mt-2">This may take 20–60 seconds</p>
            </div>
            <div className="w-64 h-1 bg-white/8 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full animate-pulse" style={{ width: '60%' }} />
            </div>
          </div>
        )}

        {/* ── PHASE: RESULTS ────────────────────────────────────────────── */}
        {phase === PHASE.RESULTS && results && (
          <div className="w-full flex flex-col gap-6 pb-10">

            {/* Score hero */}
            <div className="relative p-8 rounded-3xl border border-white/8 bg-white/[0.02] backdrop-blur-sm text-center overflow-hidden">
              <div
                className="absolute inset-0 opacity-10"
                style={{
                  background: `radial-gradient(ellipse at center, ${
                    results.composite_score >= 75 ? '#10b981' :
                    results.composite_score >= 50  ? '#f59e0b' : '#ef4444'
                  }, transparent 70%)`,
                }}
              />
              <div className="relative z-10">
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em] block mb-3">
                  Composite Interview Score
                </span>
                <div className={`text-8xl font-black tabular-nums leading-none mb-2 ${scoreColor(results.composite_score)}`}>
                  {results.composite_score?.toFixed(1)}
                </div>
                <p className="text-zinc-500 text-sm font-mono">out of 100.0 · adjusted score</p>

                {/* Difficulty & multiplier callout */}
                {results.difficulty_multiplier != null && results.difficulty_multiplier !== 1.0 && (
                  <div className="mt-3 inline-flex items-center gap-2 text-[11px] font-mono text-zinc-500">
                    <span className="bg-white/5 border border-white/8 rounded-lg px-2.5 py-1">
                      Raw: {results.raw_composite_score?.toFixed(1)}
                    </span>
                    <span className="text-zinc-700">×</span>
                    <span className={`px-2.5 py-1 rounded-lg border font-bold ${
                      results.difficulty === 'hard'
                        ? 'bg-rose-500/10 border-rose-500/25 text-rose-300'
                        : 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300'
                    }`}>
                      {results.difficulty_multiplier?.toFixed(2)}× {results.difficulty}
                    </span>
                    <span className="text-zinc-700">=</span>
                    <span className={`px-2.5 py-1 rounded-lg border font-bold ${scoreColor(results.composite_score)} bg-white/5 border-white/10`}>
                      {results.composite_score?.toFixed(1)}
                    </span>
                  </div>
                )}
                <div className="mt-5 inline-flex items-center gap-2 px-4 py-1.5 rounded-full border text-xs font-bold"
                  style={{
                    borderColor: results.composite_score >= 75 ? 'rgba(16,185,129,0.3)' :
                                 results.composite_score >= 50  ? 'rgba(245,158,11,0.3)' : 'rgba(239,68,68,0.3)',
                    color: results.composite_score >= 75 ? '#34d399' :
                           results.composite_score >= 50  ? '#fbbf24' : '#f87171',
                    background: results.composite_score >= 75 ? 'rgba(16,185,129,0.08)' :
                                results.composite_score >= 50  ? 'rgba(245,158,11,0.08)' : 'rgba(239,68,68,0.08)',
                  }}
                >
                  {results.composite_score >= 75 ? '🎯 Strong Performance' :
                   results.composite_score >= 50  ? '📈 Good — Keep Practicing' :
                                                    '🔁 Needs Work'}
                </div>
              </div>
            </div>

            {/* Per-question breakdown */}
            <div>
              <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest mb-3 px-1">
                Question-by-Question Breakdown
              </h3>
              <div className="space-y-3">
                {results.evaluations?.map((ev, i) => (
                  <ResultCard
                    key={i}
                    index={i}
                    evaluation={ev}
                    questionText={results.questions?.[i]?.question_text ?? `Question ${i + 1}`}
                    acoustics={results.acoustic_metrics?.[i]}
                  />
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-3 pt-2">
              <button
                onClick={() => { stopStream(); setPhase(PHASE.SETUP); setResults(null); setBlobs([]); }}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-rose-500/15 border border-rose-500/25 text-rose-300 font-bold text-sm hover:bg-rose-500/25 transition-colors"
              >
                <RefreshCcw className="w-4 h-4" /> Retake Interview
              </button>
              <button
                onClick={onBack}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-white/5 border border-white/10 text-zinc-300 font-bold text-sm hover:bg-white/8 transition-colors"
              >
                ← Back to Dashboard
              </button>
            </div>

            {/* result_id reference */}
            {results.result_id && (
              <p className="text-center text-[10px] font-mono text-zinc-700">
                Session saved · result_id: {results.result_id}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
