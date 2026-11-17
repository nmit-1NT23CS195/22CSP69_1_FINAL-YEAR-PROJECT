import React, { useEffect, useRef, useState } from 'react';
import {
  Activity, BrainCircuit, ChevronRight, Rocket, Target, Zap,
  BarChart2, Shield, Clock, TrendingUp, Star, Terminal, Cpu,
  Code2, RotateCcw, BookOpen, Brain, Calculator, Lightbulb,
  MessageSquare, Loader2, AlertCircle,
} from 'lucide-react';
import { useQuiz } from '../context/QuizContext';
import { useApp } from '../context/AppContext';
import { useAssessment } from '../context/AssessmentContext';

// ── Animated gradient orb ─────────────────────────────────────────────────────
function GlowOrb({ color, size, top, left, blur }) {
  return (
    <div
      className="absolute rounded-full pointer-events-none"
      style={{
        background: color,
        width: size,
        height: size,
        top,
        left,
        filter: `blur(${blur || '80px'})`,
        opacity: 0.18,
        zIndex: 0,
      }}
    />
  );
}

// ── Floating particle ─────────────────────────────────────────────────────────
function Particle({ delay, duration, x, y }) {
  return (
    <div
      className="absolute w-1 h-1 rounded-full bg-indigo-400/40 animate-float-particle"
      style={{
        left: `${x}%`,
        top: `${y}%`,
        animationDelay: `${delay}s`,
        animationDuration: `${duration}s`,
      }}
    />
  );
}

// ── Mini stat chip ────────────────────────────────────────────────────────────
function StatChip({ icon: Icon, label, value, color }) {
  return (
    <div className="flex items-center gap-2 bg-white/5 border border-white/10 rounded-xl px-3 py-2 backdrop-blur-sm">
      <Icon className={`w-4 h-4 ${color}`} />
      <div>
        <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider leading-none">{label}</p>
        <p className="text-sm font-bold text-white leading-tight">{value}</p>
      </div>
    </div>
  );
}

// ── Feature pill ──────────────────────────────────────────────────────────────
function FeaturePill({ label }) {
  return (
    <span className="text-[10px] font-semibold text-zinc-500 bg-white/5 border border-white/5 rounded-lg px-2.5 py-1 group-hover:text-zinc-400 group-hover:border-white/10 transition-colors">
      {label}
    </span>
  );
}

// ── Streaming loading overlay shown on module cards while the stream resolves ──
function LoadingOverlay({ moduleLabel, accentColor }) {
  return (
    <div className="absolute inset-0 rounded-3xl flex flex-col items-center justify-center gap-3 z-20"
      style={{ background: 'rgba(5,5,8,0.88)', backdropFilter: 'blur(8px)' }}
    >
      <Loader2 className={`w-8 h-8 animate-spin ${accentColor}`} />
      <p className={`text-sm font-bold ${accentColor}`}>Resolving {moduleLabel}…</p>
      <p className="text-xs text-zinc-600 font-mono">Streaming from assessment engine</p>
    </div>
  );
}

// ── Error banner ──────────────────────────────────────────────────────────────
function StreamErrorBanner({ message, onDismiss }) {
  return (
    <div className="relative z-10 mx-6 mb-4 flex items-start gap-3 bg-red-950/60 border border-red-500/30 rounded-2xl px-4 py-3 backdrop-blur-sm">
      <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
      <div className="flex-1">
        <p className="text-xs font-bold text-red-300">Assessment stream error</p>
        <p className="text-xs text-red-400/80 font-mono mt-0.5 break-all">{message}</p>
      </div>
      <button
        onClick={onDismiss}
        className="text-red-500 hover:text-red-300 text-xs font-bold transition-colors shrink-0"
      >
        ✕
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  DASHBOARD
// ─────────────────────────────────────────────────────────────────────────────
export default function Dashboard({ onViewChange }) {
  const { q10Score } = useQuiz();
  const { initialTargetRole, resetOnboarding, resumeFile } = useApp();
  const {
    startAssessment,
    resetAssessment,
    isLoadingActive,
    isPrefetchingSecondary,
    activeModule,
    streamError,
    technicalQuestions,
    aptitudeQuestions,
  } = useAssessment();

  const [mounted, setMounted] = useState(false);
  const [hoveredCard, setHoveredCard] = useState(null);
  const [prsValue, setPrsValue] = useState(0);
  const [launchingModule, setLaunchingModule] = useState(null); // 'technical' | 'aptitude' | null
  const canvasRef = useRef(null);

  // ── Entry animation ───────────────────────────────────────────────────────
  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 50);
    return () => clearTimeout(t);
  }, []);

  // ── Animate PRS counter ───────────────────────────────────────────────────
  useEffect(() => {
    const target = Math.min(Math.round(q10Score), 100);
    if (target === 0) return;
    let current = 0;
    const step = Math.ceil(target / 40);
    const interval = setInterval(() => {
      current = Math.min(current + step, target);
      setPrsValue(current);
      if (current >= target) clearInterval(interval);
    }, 30);
    return () => clearInterval(interval);
  }, [q10Score]);

  // ── Canvas star field ─────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    const stars = Array.from({ length: 80 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 1.2 + 0.2,
      a: Math.random(),
      speed: Math.random() * 0.003 + 0.001,
    }));
    let animId;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      stars.forEach(s => {
        s.a = Math.abs(Math.sin(Date.now() * s.speed));
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(148,163,255,${s.a * 0.6})`;
        ctx.fill();
      });
      animId = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(animId);
  }, []);

  // ── Transition to quiz when active module questions arrive ────────────────
  useEffect(() => {
    if (!launchingModule) return;
    if (!isLoadingActive && activeModule === launchingModule) {
      const hasQs =
        launchingModule === 'technical'
          ? technicalQuestions.length > 0
          : aptitudeQuestions.length > 0;
      if (hasQs) {
        setLaunchingModule(null);
        onViewChange(launchingModule === 'technical' ? 'mcq' : 'aptitude');
      }
    }
  }, [
    isLoadingActive, activeModule, launchingModule,
    technicalQuestions, aptitudeQuestions, onViewChange,
  ]);

  // ── Launch handlers ───────────────────────────────────────────────────────
  const handleLaunchTechnical = () => {
    if (isLoadingActive || launchingModule) return;
    resetAssessment();
    setLaunchingModule('technical');
    startAssessment('technical', initialTargetRole, {
      green:  [],
      yellow: [],
      red:    [],
    });
  };

  const handleLaunchAptitude = () => {
    if (isLoadingActive || launchingModule) return;
    resetAssessment();
    setLaunchingModule('aptitude');
    startAssessment('aptitude', initialTargetRole, {
      green:  [],
      yellow: [],
      red:    [],
    });
  };

  const particles = Array.from({ length: 12 }, (_, i) => ({
    x: (i * 17 + 5) % 100,
    y: (i * 23 + 10) % 100,
    delay: i * 0.4,
    duration: 4 + (i % 3),
  }));

  const prsColor = prsValue >= 70 ? 'from-emerald-400 to-cyan-400'
    : prsValue >= 40 ? 'from-amber-400 to-orange-400'
    : 'from-indigo-400 to-purple-400';

  const prsLabel = prsValue >= 70 ? 'Placement Ready'
    : prsValue >= 40 ? 'Developing'
    : q10Score > 0 ? 'Needs Work' : 'Not Assessed';

  const isBusy = isLoadingActive && !!launchingModule;

  return (
    <div className="relative min-h-screen bg-[#050508] overflow-hidden flex flex-col">
      {/* Canvas starfield */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-0 opacity-60" />

      {/* Glow orbs */}
      <GlowOrb color="radial-gradient(circle, #6366f1, transparent)" size="600px" top="-100px" left="-150px" />
      <GlowOrb color="radial-gradient(circle, #a855f7, transparent)" size="500px" top="30%" left="60%" />
      <GlowOrb color="radial-gradient(circle, #06b6d4, transparent)" size="400px" top="70%" left="10%" blur="100px" />

      {/* Particles */}
      {particles.map((p, i) => <Particle key={i} {...p} />)}

      {/* Grid lines overlay */}
      <div className="absolute inset-0 pointer-events-none z-0"
        style={{
          backgroundImage: 'linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      {/* ── NAV ────────────────────────────────────────────────────────────── */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-5 border-b border-white/5 backdrop-blur-xl bg-white/[0.02]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-[0_0_20px_rgba(99,102,241,0.5)]">
            <span className="text-white font-black text-xl leading-none">P</span>
          </div>
          <span className="text-xl font-black bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 tracking-tight">
            PlaceBuddy
          </span>
          <span className="ml-2 text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2 py-0.5 rounded-full uppercase tracking-wider">
            Phase 2
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs font-semibold text-zinc-500">
          {isPrefetchingSecondary && (
            <span className="flex items-center gap-1.5 text-purple-400">
              <Loader2 className="w-3 h-3 animate-spin" />
              Prefetching secondary…
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Systems Online
          </span>
          <div className="w-px h-4 bg-white/10" />
          <button
            onClick={resetOnboarding}
            title="Change resume or target role"
            className="flex items-center gap-1.5 text-zinc-600 hover:text-zinc-300 transition-colors text-xs font-semibold"
          >
            <RotateCcw className="w-3 h-3" />
            Re-onboard
          </button>
        </div>
      </nav>

      {/* ── Error banner ─────────────────────────────────────────────────── */}
      {streamError && (
        <StreamErrorBanner
          message={streamError}
          onDismiss={resetAssessment}
        />
      )}

      {/* ── HERO ────────────────────────────────────────────────────────── */}
      <div className={`relative z-10 text-center pt-16 pb-10 px-8 transition-all duration-700 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
        <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/20 rounded-full px-4 py-1.5 mb-6 text-xs font-semibold text-indigo-300 backdrop-blur-sm">
          <Zap className="w-3 h-3" />
          Placement Readiness Score Engine
        </div>
        <h1 className="text-5xl md:text-6xl font-black text-white tracking-tighter leading-none mb-4">
          Master Your{' '}
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
            Career Trajectory
          </span>
        </h1>
        <p className="text-zinc-400 text-lg max-w-xl mx-auto leading-relaxed">
          Three precision instruments. One unified readiness score.
          <br />
          <span className="text-zinc-500 text-sm">ML signal analysis · Technical diagnostics · Aptitude index.</span>
        </p>
        {initialTargetRole && (
          <div className="inline-flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/25 rounded-full px-4 py-1.5 mt-5 text-xs font-semibold text-emerald-300 backdrop-blur-sm">
            <Target className="w-3 h-3" />
            Targeting: {initialTargetRole}
          </div>
        )}
      </div>

      {/* ── BENTO GRID ──────────────────────────────────────────────────── */}
      <div className={`relative z-10 flex-1 max-w-6xl mx-auto w-full px-6 pb-12 transition-all duration-1000 delay-200 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>

        {/* Top row — 3 equal-width module cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-5">

          {/* ── CARD A: Cognitive ATS Insights ────────────────────────────── */}
          <button
            id="module-ats-btn"
            onClick={() => onViewChange('ats')}
            onMouseEnter={() => setHoveredCard('ats')}
            onMouseLeave={() => setHoveredCard(null)}
            className="group relative p-7 rounded-3xl text-left overflow-hidden transition-all duration-500 border focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
            style={{
              background: hoveredCard === 'ats'
                ? 'linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(14,14,20,0.9) 60%)'
                : 'rgba(14,14,20,0.8)',
              borderColor: hoveredCard === 'ats' ? 'rgba(99,102,241,0.4)' : 'rgba(255,255,255,0.06)',
              boxShadow: hoveredCard === 'ats' ? '0 0 40px rgba(99,102,241,0.15), 0 0 0 1px rgba(99,102,241,0.2)' : 'none',
              backdropFilter: 'blur(20px)',
            }}
          >
            {/* Corner glow */}
            <div className="absolute -top-20 -left-20 w-52 h-52 bg-indigo-600/20 rounded-full blur-3xl group-hover:opacity-100 opacity-0 transition-opacity duration-500" />
            {/* Top accent line */}
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-indigo-500/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

            <div className="relative z-10">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-5 bg-gradient-to-br from-indigo-500/20 to-indigo-900/40 border border-indigo-500/20 group-hover:border-indigo-400/40 transition-colors">
                <Activity className="w-7 h-7 text-indigo-400 group-hover:text-indigo-300 transition-colors" />
              </div>

              <div className="flex items-center gap-2 mb-3">
                <span className="text-[10px] font-bold text-indigo-400/80 uppercase tracking-[0.15em] bg-indigo-500/10 px-2 py-0.5 rounded-full border border-indigo-500/20">
                  Module A
                </span>
                <span className="text-[10px] font-semibold text-zinc-600">· ML-Powered</span>
                <span className="ml-auto text-[10px] font-bold text-amber-400/80 uppercase tracking-wide bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20">
                  ⏸ Standby
                </span>
              </div>

              <h2 className="text-xl font-black text-white mb-2 tracking-tight group-hover:text-indigo-100 transition-colors">
                Cognitive ATS Insights
              </h2>
              <p className="text-zinc-400 text-sm leading-relaxed mb-6 group-hover:text-zinc-300 transition-colors">
                Upload your resume and JD. Our forensic ML engine cross-analyses skill matrices, semantic similarity, and keyword signal strength.
              </p>

              <div className="flex flex-wrap gap-2 mb-6">
                {['Semantic Analysis', 'Skill Matrix', 'PDF Report', 'Role Matching'].map(f => (
                  <FeaturePill key={f} label={f} />
                ))}
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-indigo-400 font-bold text-sm group-hover:text-indigo-300 transition-colors">
                  Analyze Resume
                  <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-300" />
                </div>
                <div className="flex items-center gap-1 text-zinc-600 text-xs font-mono">
                  <Shield className="w-3 h-3" />
                  Local-only
                </div>
              </div>
            </div>
          </button>

          {/* ── CARD B: Technical Diagnostic ──────────────────────────────── */}
          <button
            id="module-technical-btn"
            onClick={handleLaunchTechnical}
            onMouseEnter={() => setHoveredCard('mcq')}
            onMouseLeave={() => setHoveredCard(null)}
            disabled={isBusy}
            className="group relative p-7 rounded-3xl text-left overflow-hidden transition-all duration-500 border focus:outline-none focus:ring-2 focus:ring-purple-500/50 disabled:cursor-not-allowed"
            style={{
              background: hoveredCard === 'mcq'
                ? 'linear-gradient(135deg, rgba(168,85,247,0.15) 0%, rgba(14,14,20,0.9) 60%)'
                : 'rgba(14,14,20,0.8)',
              borderColor: hoveredCard === 'mcq' ? 'rgba(168,85,247,0.4)' : 'rgba(255,255,255,0.06)',
              boxShadow: hoveredCard === 'mcq' ? '0 0 40px rgba(168,85,247,0.15), 0 0 0 1px rgba(168,85,247,0.2)' : 'none',
              backdropFilter: 'blur(20px)',
              opacity: isBusy && launchingModule !== 'technical' ? 0.5 : 1,
            }}
          >
            {/* Loading overlay when resolving technical module */}
            {launchingModule === 'technical' && isLoadingActive && (
              <LoadingOverlay moduleLabel="Technical MCQ" accentColor="text-purple-400" />
            )}

            <div className="absolute -top-20 -right-20 w-52 h-52 bg-purple-600/20 rounded-full blur-3xl group-hover:opacity-100 opacity-0 transition-opacity duration-500" />
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-purple-500/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

            <div className="relative z-10">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-5 bg-gradient-to-br from-purple-500/20 to-purple-900/40 border border-purple-500/20 group-hover:border-purple-400/40 transition-colors">
                <BrainCircuit className="w-7 h-7 text-purple-400 group-hover:text-purple-300 transition-colors" />
              </div>

              <div className="flex items-center gap-2 mb-3">
                <span className="text-[10px] font-bold text-purple-400/80 uppercase tracking-[0.15em] bg-purple-500/10 px-2 py-0.5 rounded-full border border-purple-500/20">
                  Module B
                </span>
                <span className="text-[10px] font-semibold text-zinc-600">· Adaptive Diagnostic</span>
                <span className="ml-auto text-[10px] font-bold text-amber-400/80 uppercase tracking-wide bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20">
                  ⏸ Standby
                </span>
              </div>

              <h2 className="text-xl font-black text-white mb-2 tracking-tight group-hover:text-purple-100 transition-colors">
                Dynamic 10-MCQ Diagnostic
              </h2>
              <p className="text-zinc-400 text-sm leading-relaxed mb-6 group-hover:text-zinc-300 transition-colors">
                A curated 10-question technical assessment across DSA, systems, and core CS concepts. Timed. Scored. Saved to your PRS profile.
              </p>

              <div className="flex flex-wrap gap-2 mb-6">
                {['10 Questions', 'DSA & Systems', 'Timed Sessions', 'PRS Tracked'].map(f => (
                  <FeaturePill key={f} label={f} />
                ))}
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-purple-400 font-bold text-sm group-hover:text-purple-300 transition-colors">
                  {launchingModule === 'technical' && isLoadingActive
                    ? <><Loader2 className="w-4 h-4 animate-spin" /> Resolving…</>
                    : <>Launch Technical <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-300" /></>
                  }
                </div>
                <div className="flex items-center gap-1 text-zinc-600 text-xs font-mono">
                  <Terminal className="w-3 h-3" />
                  CLI Mode
                </div>
              </div>
            </div>
          </button>

          {/* ── CARD C: Foundational Aptitude & Reasoning ─────────────────── */}
          <button
            id="module-aptitude-btn"
            onClick={handleLaunchAptitude}
            onMouseEnter={() => setHoveredCard('aptitude')}
            onMouseLeave={() => setHoveredCard(null)}
            disabled={isBusy}
            className="group relative p-7 rounded-3xl text-left overflow-hidden transition-all duration-500 border focus:outline-none focus:ring-2 focus:ring-cyan-500/50 disabled:cursor-not-allowed"
            style={{
              background: hoveredCard === 'aptitude'
                ? 'linear-gradient(135deg, rgba(6,182,212,0.15) 0%, rgba(14,14,20,0.9) 60%)'
                : 'rgba(14,14,20,0.8)',
              borderColor: hoveredCard === 'aptitude' ? 'rgba(6,182,212,0.4)' : 'rgba(255,255,255,0.06)',
              boxShadow: hoveredCard === 'aptitude' ? '0 0 40px rgba(6,182,212,0.15), 0 0 0 1px rgba(6,182,212,0.2)' : 'none',
              backdropFilter: 'blur(20px)',
              opacity: isBusy && launchingModule !== 'aptitude' ? 0.5 : 1,
            }}
          >
            {/* Loading overlay when resolving aptitude module */}
            {launchingModule === 'aptitude' && isLoadingActive && (
              <LoadingOverlay moduleLabel="Aptitude Assessment" accentColor="text-cyan-400" />
            )}

            <div className="absolute -bottom-20 -right-20 w-52 h-52 bg-cyan-600/20 rounded-full blur-3xl group-hover:opacity-100 opacity-0 transition-opacity duration-500" />
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-500/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

            <div className="relative z-10">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-5 bg-gradient-to-br from-cyan-500/20 to-cyan-900/40 border border-cyan-500/20 group-hover:border-cyan-400/40 transition-colors">
                <Brain className="w-7 h-7 text-cyan-400 group-hover:text-cyan-300 transition-colors" />
              </div>

              <div className="flex items-center gap-2 mb-3">
                <span className="text-[10px] font-bold text-cyan-400/80 uppercase tracking-[0.15em] bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20">
                  Module C
                </span>
                <span className="text-[10px] font-semibold text-zinc-600">· Cognitive Index</span>
                <span className="ml-auto text-[10px] font-bold text-amber-400/80 uppercase tracking-wide bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20">
                  ⏸ Standby
                </span>
              </div>

              <h2 className="text-xl font-black text-white mb-2 tracking-tight group-hover:text-cyan-100 transition-colors">
                Foundational Aptitude & Reasoning
              </h2>
              <p className="text-zinc-400 text-sm leading-relaxed mb-6 group-hover:text-zinc-300 transition-colors">
                10 questions spanning quantitative aptitude, logical reasoning, and verbal ability. Benchmarks your cognitive foundation for placement drives.
              </p>

              <div className="flex flex-wrap gap-2 mb-6">
                {['10 Questions', 'Quant', 'Logic', 'Verbal', 'Cognitive Index'].map(f => (
                  <FeaturePill key={f} label={f} />
                ))}
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-cyan-400 font-bold text-sm group-hover:text-cyan-300 transition-colors">
                  {launchingModule === 'aptitude' && isLoadingActive
                    ? <><Loader2 className="w-4 h-4 animate-spin" /> Resolving…</>
                    : <>Launch Aptitude <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-300" /></>
                  }
                </div>
                <div className="flex items-center gap-1 text-zinc-600 text-xs font-mono">
                  <Calculator className="w-3 h-3" />
                  3 Domains
                </div>
              </div>
            </div>
          </button>
        </div>

        {/* Bottom row — PRS Score card + System Status card */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

          {/* PRS Score Card */}
          <div
            className="relative p-6 rounded-3xl overflow-hidden border border-white/5 backdrop-blur-xl"
            style={{ background: 'rgba(14,14,20,0.8)' }}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 via-purple-500/5 to-transparent" />
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-indigo-400" />
                  <span className="text-xs font-bold text-zinc-400 uppercase tracking-widest">PRS Dashboard</span>
                </div>
                <span className="text-[10px] font-mono text-zinc-600 bg-white/5 px-2 py-0.5 rounded-full">
                  {q10Score > 0 ? 'Live' : 'Awaiting Data'}
                </span>
              </div>

              <div className="flex items-end gap-3 mb-4">
                <div className={`text-6xl font-black bg-clip-text text-transparent bg-gradient-to-r ${prsColor} tabular-nums leading-none`}>
                  {q10Score > 0 ? prsValue : '--'}
                </div>
                {q10Score > 0 && (
                  <div className="mb-1">
                    <p className={`text-sm font-bold bg-clip-text text-transparent bg-gradient-to-r ${prsColor}`}>{prsLabel}</p>
                    <p className="text-xs text-zinc-600 font-mono">Q10 metric</p>
                  </div>
                )}
              </div>

              <div className="h-1.5 bg-white/5 rounded-full overflow-hidden mb-4">
                <div
                  className={`h-full rounded-full bg-gradient-to-r ${prsColor} transition-all duration-1000`}
                  style={{ width: q10Score > 0 ? `${Math.min(prsValue, 100)}%` : '0%' }}
                />
              </div>

              {q10Score === 0 ? (
                <p className="text-xs text-zinc-600 font-mono">Complete Module B or C to populate your PRS score.</p>
              ) : (
                <div className="grid grid-cols-3 gap-2 mt-2">
                  <StatChip icon={BarChart2} label="Q10 Raw" value={`${Math.round(q10Score)}%`} color="text-indigo-400" />
                  <StatChip icon={TrendingUp} label="Status" value={prsLabel} color="text-purple-400" />
                  <StatChip icon={Star} label="Rank" value={q10Score >= 80 ? 'Elite' : q10Score >= 60 ? 'Strong' : 'Growing'} color="text-amber-400" />
                </div>
              )}
            </div>
          </div>

          {/* System Status Card */}
          <div
            className="relative p-6 rounded-3xl overflow-hidden border border-white/5 backdrop-blur-xl"
            style={{ background: 'rgba(14,14,20,0.8)' }}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent" />
            <div className="relative z-10">
              <div className="flex items-center gap-2 mb-5">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-bold text-zinc-400 uppercase tracking-widest">System Status</span>
              </div>

              <div className="space-y-3">
                {[
                  { label: 'ATS Core Engine',       status: 'online',  icon: Activity,    color: 'text-indigo-400' },
                  { label: 'MCQ Diagnostic',         status: 'online',  icon: BrainCircuit,color: 'text-purple-400' },
                  { label: 'Aptitude Assessment',    status: 'online',  icon: Brain,       color: 'text-cyan-400'   },
                  { label: 'PDF Generator',          status: 'online',  icon: Code2,       color: 'text-emerald-400'},
                  { label: 'FastAPI Backend',        status: 'standby', icon: Rocket,      color: 'text-amber-400'  },
                ].map(({ label, status, icon: Icon, color }) => (
                  <div key={label} className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
                    <div className="flex items-center gap-2.5">
                      <Icon className={`w-3.5 h-3.5 ${color}`} />
                      <span className="text-sm text-zinc-400 font-medium">{label}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full ${status === 'online' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                      <span className={`text-xs font-mono font-semibold ${status === 'online' ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 pt-3 border-t border-white/5 flex items-center gap-2">
                <Clock className="w-3 h-3 text-zinc-600" />
                <span className="text-[10px] font-mono text-zinc-600">
                  {new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })} IST
                </span>
                <span className="text-zinc-700 mx-1">·</span>
                <span className="text-[10px] font-mono text-zinc-600">PlaceBuddy FYP v2.0</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer tagline */}
        <p className="text-center text-xs text-zinc-700 font-mono mt-8 tracking-wider">
          PLACEBUDDY · PHASE 2 · COGNITIVE ASSESSMENT ENGINE · 22CSP69
        </p>
      </div>
    </div>
  );
}
