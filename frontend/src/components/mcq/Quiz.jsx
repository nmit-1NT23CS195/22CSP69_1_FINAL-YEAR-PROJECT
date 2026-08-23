import React, { useState, useEffect, useRef, useCallback } from "react";
import { generateQuiz, saveQuizResult, QUESTION_BANK } from "../../api/mcqService";
import QuizResult from "./QuizResult";
import { useQuiz } from "../../context/QuizContext";
import {
  BrainCircuit, ChevronRight, Zap, Terminal, Clock,
  CheckCircle2, XCircle, ArrowLeft, RotateCcw, BookOpen
} from "lucide-react";

const TOPICS = ["Mixed (All Topics)", ...Object.keys(QUESTION_BANK)];
const TOTAL_TIME = 12 * 60; // 12 minutes for 10 questions

function ProgressRing({ percent, size = 52, stroke = 4, color = "#6366f1" }) {
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (percent / 100) * circ;
  return (
    <svg width={size} height={size} className="rotate-[-90deg]">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={stroke} />
      <circle
        cx={size / 2} cy={size / 2} r={r} fill="none"
        stroke={color} strokeWidth={stroke}
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.4s ease' }}
      />
    </svg>
  );
}

function TimerDisplay({ seconds, isWarning }) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return (
    <div className={`flex items-center gap-1.5 font-mono text-sm font-bold tabular-nums transition-colors ${isWarning ? 'text-red-400' : 'text-zinc-400'}`}>
      <Clock className={`w-3.5 h-3.5 ${isWarning ? 'animate-pulse text-red-400' : ''}`} />
      {String(mins).padStart(2, '0')}:{String(secs).padStart(2, '0')}
    </div>
  );
}

// ── LANDING SCREEN ──────────────────────────────────────────────────────────
function QuizLanding({ onStart, loading }) {
  const [selectedTopic, setSelectedTopic] = useState(TOPICS[0]);
  const [hovered, setHovered] = useState(null);

  return (
    <div className="min-h-screen bg-[#050508] flex items-center justify-center p-6">
      {/* Glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-purple-600/10 rounded-full blur-[120px] pointer-events-none" />

      <div className="relative w-full max-w-2xl animate-fade-up">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-gradient-to-br from-purple-500/20 to-indigo-900/40 border border-purple-500/20 mb-6 shadow-[0_0_40px_rgba(168,85,247,0.2)]">
            <BrainCircuit className="w-10 h-10 text-purple-400" />
          </div>
          <div className="inline-flex items-center gap-2 bg-purple-500/10 border border-purple-500/20 rounded-full px-4 py-1 mb-4 text-xs font-semibold text-purple-300">
            <Terminal className="w-3 h-3" />
            PlaceBuddy · Module B · Technical Diagnostic
          </div>
          <h1 className="text-4xl font-black text-white tracking-tighter mb-3">
            10-MCQ Technical{" "}
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-400">
              Assessment
            </span>
          </h1>
          <p className="text-zinc-400 text-sm max-w-lg mx-auto leading-relaxed">
            Curated questions spanning DSA, Operating Systems, Computer Networks, and Web Development.
            Your score feeds directly into your <span className="text-purple-300 font-semibold">Placement Readiness Score (PRS)</span>.
          </p>
        </div>

        {/* Topic selector */}
        <div className="mb-6">
          <p className="text-xs font-bold text-zinc-500 uppercase tracking-widest mb-3 flex items-center gap-2">
            <BookOpen className="w-3 h-3" /> Select Topic Domain
          </p>
          <div className="grid grid-cols-2 gap-2.5">
            {TOPICS.map(topic => (
              <button
                key={topic}
                onClick={() => setSelectedTopic(topic)}
                onMouseEnter={() => setHovered(topic)}
                onMouseLeave={() => setHovered(null)}
                className={`relative p-4 rounded-2xl border text-left text-sm font-semibold transition-all duration-300 focus:outline-none ${selectedTopic === topic
                  ? 'border-purple-500/60 bg-purple-500/10 text-purple-200 shadow-[0_0_20px_rgba(168,85,247,0.1)]'
                  : 'border-white/6 bg-white/[0.03] text-zinc-400 hover:border-white/10 hover:text-zinc-300'
                  }`}
              >
                {selectedTopic === topic && (
                  <div className="absolute top-2 right-2">
                    <CheckCircle2 className="w-4 h-4 text-purple-400" />
                  </div>
                )}
                <span className="block">{topic}</span>
                <span className="text-[10px] font-normal text-zinc-600 mt-0.5 block">
                  {topic === TOPICS[0]
                    ? `${Object.values(QUESTION_BANK).flat().length} questions · Random 10`
                    : `${QUESTION_BANK[topic]?.length || 10} questions`}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Rules */}
        <div className="bg-white/[0.03] border border-white/6 rounded-2xl p-4 mb-6">
          <p className="text-xs font-bold text-zinc-500 uppercase tracking-widest mb-3">Assessment Rules</p>
          <div className="grid grid-cols-3 gap-3">
            {[
              { icon: '🎯', label: '10 Questions', sub: 'MCQ format' },
              { icon: '⏱️', label: '12 Minutes', sub: 'Time limit' },
              { icon: '📊', label: 'PRS Tracked', sub: 'Auto-saved' },
            ].map(r => (
              <div key={r.label} className="text-center">
                <div className="text-xl mb-1">{r.icon}</div>
                <p className="text-xs font-bold text-zinc-300">{r.label}</p>
                <p className="text-[10px] text-zinc-600">{r.sub}</p>
              </div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <button
          onClick={() => onStart(selectedTopic === TOPICS[0] ? null : selectedTopic)}
          disabled={loading}
          className="w-full relative overflow-hidden bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-black py-4 px-6 rounded-2xl transition-all duration-300 shadow-[0_0_30px_rgba(168,85,247,0.3)] hover:shadow-[0_0_40px_rgba(168,85,247,0.5)] disabled:opacity-50 disabled:cursor-not-allowed group"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-3">
              <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Initializing Assessment...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <Zap className="w-5 h-5" />
              Launch Diagnostic Session
              <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </span>
          )}
          <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>
      </div>
    </div>
  );
}

// ── MAIN QUIZ COMPONENT ──────────────────────────────────────────────────────
export default function Quiz({ onBackToDashboard }) {
  const [phase, setPhase] = useState('landing'); // landing | quiz | result
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [quizData, setQuizData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [timeLeft, setTimeLeft] = useState(TOTAL_TIME);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [answered, setAnswered] = useState(false);
  const timerRef = useRef(null);

  const { setQ10Score } = useQuiz();

  // Timer tick
  useEffect(() => {
    if (phase !== 'quiz') return;
    timerRef.current = setInterval(() => {
      setTimeLeft(t => {
        if (t <= 1) {
          clearInterval(timerRef.current);
          finishQuiz();
          return 0;
        }
        return t - 1;
      });
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, [phase]);

  const handleStartQuiz = async (topic) => {
    setLoading(true);
    try {
      const data = await generateQuiz(topic);
      setQuizData(data);
      setAnswers(new Array(data.length).fill(null));
      setCurrentQuestion(0);
      setTimeLeft(TOTAL_TIME);
      setSelectedAnswer(null);
      setAnswered(false);
      setResultData(null);
      setPhase('quiz');
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectAnswer = (option) => {
    if (answered) return;
    setSelectedAnswer(option);
    const newAnswers = [...answers];
    newAnswers[currentQuestion] = option;
    setAnswers(newAnswers);
  };

  const handleNext = useCallback(() => {
    if (!selectedAnswer && !answers[currentQuestion]) return;
    if (currentQuestion < quizData.length - 1) {
      setCurrentQuestion(q => q + 1);
      setSelectedAnswer(answers[currentQuestion + 1]);
      setAnswered(false);
    } else {
      finishQuiz();
    }
  }, [currentQuestion, quizData, answers, selectedAnswer]);

  const finishQuiz = useCallback(async () => {
    clearInterval(timerRef.current);
    if (!quizData) return;

    let correct = 0;
    const questionResults = quizData.map((q, index) => {
      const userAns = answers[index];
      const isCorrect = q.correctAnswer === userAns;
      if (isCorrect) correct++;
      return {
        question: q.question,
        answer: q.correctAnswer,
        userAnswer: userAns || '(Skipped)',
        isCorrect,
        explanation: q.explanation,
      };
    });

    const score = (correct / quizData.length) * 100;
    setQ10Score(score);

    await saveQuizResult(quizData, answers, score);

    const tip = score === 100
      ? "Perfect score! Outstanding technical mastery. Your placement readiness is highly competitive."
      : score >= 70
        ? "Strong performance. Review the explanations below to fortify the remaining gaps."
        : score >= 50
          ? "Good effort. Focus on the incorrect answers and understand the underlying concepts."
          : "Keep going! Technical mastery is built through deliberate practice. Review each explanation carefully.";

    setResultData({
      quizScore: score,
      correct,
      total: quizData.length,
      questions: questionResults,
      improvementTip: tip,
    });
    setPhase('result');
  }, [quizData, answers, setQ10Score]);

  const handleRestart = () => {
    setQuizData(null);
    setResultData(null);
    setAnswers([]);
    setCurrentQuestion(0);
    setSelectedAnswer(null);
    setAnswered(false);
    setPhase('landing');
  };

  // ── LANDING ──
  if (phase === 'landing') {
    return <QuizLanding onStart={handleStartQuiz} loading={loading} />;
  }

  // ── RESULT ──
  if (phase === 'result') {
    return (
      <div className="min-h-screen bg-[#050508] p-6">
        <QuizResult result={resultData} onStartNew={handleRestart} onBackToDashboard={onBackToDashboard} />
      </div>
    );
  }

  // ── QUIZ ──
  const question = quizData[currentQuestion];
  const progress = ((currentQuestion + 1) / quizData.length) * 100;
  const isWarning = timeLeft < 120;
  const currentAnswer = selectedAnswer ?? answers[currentQuestion];

  return (
    <div className="min-h-screen bg-[#050508] flex items-center justify-center p-6">
      {/* Ambient glow */}
      <div className="absolute top-1/4 left-1/3 w-[500px] h-[400px] bg-indigo-600/8 rounded-full blur-[100px] pointer-events-none" />

      <div className="relative w-full max-w-2xl">
        {/* Header bar */}
        <div className="flex items-center justify-between mb-5">
          <button
            onClick={onBackToDashboard}
            className="flex items-center gap-1.5 text-xs font-semibold text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Exit
          </button>

          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-zinc-600">
              Q{currentQuestion + 1} / {quizData.length}
            </span>
            <TimerDisplay seconds={timeLeft} isWarning={isWarning} />
          </div>
        </div>

        {/* Progress bar */}
        <div className="h-1 bg-white/5 rounded-full overflow-hidden mb-6">
          <div
            className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Question card */}
        <div
          className="rounded-3xl p-7 mb-4 border border-white/6 backdrop-blur-2xl"
          style={{ background: 'rgba(14,14,22,0.85)' }}
        >
          {/* Question meta */}
          <div className="flex items-center gap-2 mb-5">
            <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-xs font-black text-indigo-400">
              {currentQuestion + 1}
            </div>
            <div className="flex-1 flex items-center gap-1.5">
              {Array.from({ length: quizData.length }, (_, i) => (
                <div
                  key={i}
                  className={`h-1 flex-1 rounded-full transition-all duration-300 ${i < currentQuestion ? 'bg-indigo-500' : i === currentQuestion ? 'bg-indigo-400' : 'bg-white/5'}`}
                />
              ))}
            </div>
            <div className="relative w-12 h-12">
              <ProgressRing percent={progress} size={48} stroke={3} color={isWarning ? '#f87171' : '#6366f1'} />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className={`text-[10px] font-black ${isWarning ? 'text-red-400' : 'text-indigo-400'}`}>
                  {Math.round(progress)}%
                </span>
              </div>
            </div>
          </div>

          {/* Question text */}
          <p className="text-lg font-bold text-white leading-snug mb-7 tracking-tight">
            {question.question}
          </p>

          {/* Options */}
          <div className="space-y-3">
            {question.options.map((option, index) => {
              const isSelected = currentAnswer === option;
              const letter = ['A', 'B', 'C', 'D'][index];

              return (
                <button
                  key={index}
                  onClick={() => handleSelectAnswer(option)}
                  className={`w-full flex items-center gap-4 p-4 rounded-2xl border text-left transition-all duration-200 group focus:outline-none focus:ring-2 focus:ring-indigo-500/30 ${isSelected
                    ? 'border-indigo-500/60 bg-indigo-500/10 shadow-[0_0_20px_rgba(99,102,241,0.1)]'
                    : 'border-white/5 bg-white/[0.02] hover:border-white/10 hover:bg-white/[0.04]'
                    }`}
                >
                  <div className={`w-8 h-8 rounded-xl flex-shrink-0 flex items-center justify-center text-xs font-black border transition-colors ${isSelected
                    ? 'border-indigo-500 bg-indigo-500/20 text-indigo-300'
                    : 'border-white/10 bg-white/5 text-zinc-500 group-hover:border-white/20 group-hover:text-zinc-400'
                    }`}>
                    {letter}
                  </div>
                  <span className={`text-sm font-medium leading-snug transition-colors ${isSelected ? 'text-white' : 'text-zinc-400 group-hover:text-zinc-300'}`}>
                    {option}
                  </span>
                  {isSelected && (
                    <CheckCircle2 className="ml-auto w-4 h-4 text-indigo-400 flex-shrink-0" />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between gap-3">
          <button
            onClick={() => {
              if (currentQuestion > 0) {
                setCurrentQuestion(q => q - 1);
                setSelectedAnswer(answers[currentQuestion - 1]);
              }
            }}
            disabled={currentQuestion === 0}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-white/6 bg-white/[0.02] text-zinc-500 hover:text-zinc-300 hover:border-white/10 text-sm font-semibold transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>

          <div className="flex items-center gap-2">
            {currentQuestion < quizData.length - 1 ? (
              <button
                onClick={handleNext}
                disabled={!currentAnswer}
                className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold py-2.5 px-6 rounded-xl transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(99,102,241,0.2)] hover:shadow-[0_0_30px_rgba(99,102,241,0.3)] group"
              >
                Next
                <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </button>
            ) : (
              <button
                onClick={finishQuiz}
                className="flex items-center gap-2 bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white font-bold py-2.5 px-6 rounded-xl transition-all duration-200 shadow-[0_0_20px_rgba(16,185,129,0.2)]"
              >
                <Zap className="w-4 h-4" />
                Finish Assessment
              </button>
            )}
          </div>
        </div>

        {/* Skip hint */}
        {!currentAnswer && (
          <p className="text-center text-[10px] text-zinc-700 font-mono mt-3">
            Select an answer to continue · {answers.filter(Boolean).length}/{quizData.length} answered
          </p>
        )}
      </div>
    </div>
  );
}
