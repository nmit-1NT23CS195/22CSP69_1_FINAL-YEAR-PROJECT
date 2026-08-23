import React, { useEffect, useRef } from "react";
import {
  Trophy, CheckCircle2, XCircle, ArrowRight, RotateCcw,
  ArrowLeft, Target, TrendingUp, Zap, Star, BookOpen
} from "lucide-react";

function ScoreRing({ score, size = 140 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const cx = size / 2, cy = size / 2, r = size * 0.38;
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + (score / 100) * Math.PI * 2;

    // Track
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 8;
    ctx.stroke();

    // Arc color
    const color = score >= 70 ? "#34d399" : score >= 50 ? "#a78bfa" : score >= 30 ? "#fbbf24" : "#f87171";

    // Glow
    ctx.shadowColor = color;
    ctx.shadowBlur = 15;

    // Fill arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 8;
    ctx.lineCap = "round";
    ctx.stroke();

    ctx.shadowBlur = 0;
  }, [score, size]);

  const color = score >= 70 ? "from-emerald-400 to-cyan-400"
    : score >= 50 ? "from-purple-400 to-indigo-400"
    : score >= 30 ? "from-amber-400 to-orange-400"
    : "from-red-400 to-rose-400";

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <canvas ref={canvasRef} style={{ width: size, height: size }} />
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-3xl font-black bg-clip-text text-transparent bg-gradient-to-b ${color} tabular-nums leading-none`}>
          {Math.round(score)}
        </span>
        <span className="text-[10px] font-bold text-zinc-600 mt-0.5 tracking-wider">/ 100</span>
      </div>
    </div>
  );
}

function StatPill({ icon: Icon, label, value, accent }) {
  return (
    <div className={`flex flex-col items-center p-4 rounded-2xl border ${accent} bg-white/[0.02]`}>
      <Icon className="w-5 h-5 mb-2 text-zinc-400" />
      <span className="text-2xl font-black text-white tabular-nums">{value}</span>
      <span className="text-[10px] font-semibold text-zinc-600 uppercase tracking-wider mt-0.5">{label}</span>
    </div>
  );
}

export default function QuizResult({ result, onStartNew, onBackToDashboard }) {
  if (!result) return null;

  const { quizScore, correct, total, questions, improvementTip } = result;
  const prsScore = Math.round(quizScore);
  const grade = prsScore >= 90 ? { label: "Elite", emoji: "🏆", color: "text-amber-400", bg: "bg-amber-500/10 border-amber-500/20" }
    : prsScore >= 70 ? { label: "Proficient", emoji: "⚡", color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/20" }
    : prsScore >= 50 ? { label: "Developing", emoji: "📈", color: "text-purple-400", bg: "bg-purple-500/10 border-purple-500/20" }
    : { label: "Needs Work", emoji: "🔧", color: "text-amber-400", bg: "bg-amber-500/10 border-amber-500/20" };

  return (
    <div className="max-w-2xl mx-auto">
      {/* Header */}
      <div className="relative rounded-3xl p-7 mb-4 border border-white/6 overflow-hidden"
        style={{ background: 'rgba(14,14,22,0.9)', backdropFilter: 'blur(20px)' }}>
        <div className="absolute -top-20 -right-20 w-60 h-60 bg-purple-600/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row items-center gap-6">
          {/* Score ring */}
          <div className="flex-shrink-0 flex flex-col items-center">
            <ScoreRing score={prsScore} size={130} />
            <div className={`mt-3 px-4 py-1.5 rounded-full border text-xs font-bold ${grade.bg} ${grade.color}`}>
              {grade.emoji} {grade.label}
            </div>
          </div>

          {/* Info */}
          <div className="flex-1 text-center md:text-left">
            <div className="flex items-center gap-2 mb-2 justify-center md:justify-start">
              <Trophy className="w-4 h-4 text-amber-400" />
              <span className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Assessment Complete</span>
            </div>
            <h1 className="text-3xl font-black text-white tracking-tight mb-1">
              Diagnostic Analysis
            </h1>
            <p className="text-zinc-400 text-sm mb-4">Technical Diagnostic · Module B · PlaceBuddy PRS</p>

            {/* Stat pills */}
            <div className="grid grid-cols-3 gap-2">
              <StatPill icon={CheckCircle2} label="Correct" value={correct} accent="border-emerald-500/20" />
              <StatPill icon={XCircle} label="Missed" value={total - correct} accent="border-red-500/20" />
              <StatPill icon={Target} label="Score" value={`${prsScore}%`} accent="border-indigo-500/20" />
            </div>
          </div>
        </div>
      </div>

      {/* PRS Badge */}
      <div className="rounded-2xl p-4 mb-4 border border-indigo-500/15 bg-indigo-500/5 backdrop-blur-sm flex items-center gap-4">
        <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center flex-shrink-0">
          <Zap className="w-5 h-5 text-indigo-400" />
        </div>
        <div className="flex-1">
          <p className="text-xs font-bold text-indigo-400 uppercase tracking-widest mb-0.5">PRS Updated</p>
          <p className="text-sm text-zinc-300">Your Q10 metric has been saved: <span className="font-bold text-indigo-300">{prsScore}/100</span> · Return to Dashboard to view your full PRS profile.</p>
        </div>
      </div>

      {/* Improvement tip */}
      {improvementTip && (
        <div className="rounded-2xl p-4 mb-4 border border-purple-500/15 bg-purple-500/5 backdrop-blur-sm flex items-start gap-3">
          <BookOpen className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-[10px] font-bold text-purple-400 uppercase tracking-widest mb-1">Engine Assessment</p>
            <p className="text-sm text-zinc-300 leading-relaxed">{improvementTip}</p>
          </div>
        </div>
      )}

      {/* Question Review */}
      <div className="rounded-3xl border border-white/5 overflow-hidden"
        style={{ background: 'rgba(14,14,22,0.85)', backdropFilter: 'blur(20px)' }}>
        <div className="px-6 py-4 border-b border-white/5 flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-zinc-500" />
          <span className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Forensic Question Review</span>
        </div>

        <div className="max-h-[420px] overflow-y-auto divide-y divide-white/5">
          {questions.map((q, index) => (
            <div key={index} className={`p-5 transition-colors ${q.isCorrect ? 'hover:bg-emerald-500/3' : 'hover:bg-red-500/3'}`}>
              <div className="flex items-start justify-between gap-4 mb-3">
                <div className="flex items-start gap-3">
                  <div className={`w-6 h-6 rounded-lg flex-shrink-0 flex items-center justify-center text-[10px] font-black mt-0.5 ${q.isCorrect ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400' : 'bg-red-500/10 border border-red-500/20 text-red-400'}`}>
                    {index + 1}
                  </div>
                  <p className="font-semibold text-white text-sm leading-snug">{q.question}</p>
                </div>
                <span className={`flex-shrink-0 p-1 rounded-lg ${q.isCorrect ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                  {q.isCorrect ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                </span>
              </div>

              <div className="ml-9 space-y-1">
                <div className="flex items-center gap-2 text-xs font-mono text-zinc-500">
                  <span className="text-zinc-700">Your answer:</span>
                  <span className={q.isCorrect ? 'text-emerald-400 font-semibold' : 'text-red-400 font-semibold'}>{q.userAnswer}</span>
                </div>
                {!q.isCorrect && (
                  <div className="flex items-center gap-2 text-xs font-mono">
                    <span className="text-zinc-700">Correct:</span>
                    <span className="text-emerald-400 font-semibold">{q.answer}</span>
                  </div>
                )}
                <div className="mt-2 p-3 rounded-xl bg-white/[0.02] border border-white/5 text-xs text-zinc-500 leading-relaxed">
                  <span className="font-bold text-zinc-400 mr-1">Explanation:</span>
                  {q.explanation}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-3 mt-4">
        <button
          onClick={onStartNew}
          className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold py-3.5 px-6 rounded-2xl transition-all duration-200 shadow-[0_0_20px_rgba(168,85,247,0.2)]"
        >
          <RotateCcw className="w-4 h-4" />
          Retake Assessment
        </button>
        {onBackToDashboard && (
          <button
            onClick={onBackToDashboard}
            className="flex-1 flex items-center justify-center gap-2 border border-white/6 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/10 text-zinc-400 hover:text-white font-semibold py-3.5 px-6 rounded-2xl transition-all duration-200"
          >
            <ArrowLeft className="w-4 h-4" />
            Return to Dashboard
          </button>
        )}
      </div>
    </div>
  );
}
