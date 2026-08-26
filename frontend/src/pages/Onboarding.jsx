import React, { useState, useRef, useCallback } from 'react';
import { useApp } from '../context/AppContext';
import { UploadCloud, FileText, Target, Zap, Shield, Cpu, X, ChevronRight } from 'lucide-react';

const API_BASE_URL = "http://127.0.0.1:8080";

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
        filter: `blur(${blur || '100px'})`,
        opacity: 0.15,
        zIndex: 0,
      }}
    />
  );
}

export default function Onboarding() {
  const { completeOnboarding } = useApp();

  const [resumeFile, setResumeFile] = useState(null);
  const [targetRole, setTargetRole] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [loadingStage, setLoadingStage] = useState('');
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type === 'application/pdf') {
      setResumeFile(file);
      setError('');
    } else {
      setError('Please drop a PDF file.');
    }
  }, []);

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) {
      setResumeFile(file);
      setError('');
    }
  }, []);

  const handleSubmit = async () => {
    if (!resumeFile) { setError('Please upload your resume PDF.'); return; }
    if (!targetRole.trim()) { setError('Please enter your target role.'); return; }

    setIsLoading(true);
    setError('');
    setLoadingStage('Extracting resume text...');

    try {
      const formData = new FormData();
      formData.append('file', resumeFile);

      const res = await fetch(`${API_BASE_URL}/resume/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || 'Resume extraction failed.');
      }

      const data = await res.json();
      const text = data.resume_text || '';

      setLoadingStage('Securing to vault...');
      await new Promise(r => setTimeout(r, 400)); // brief UX moment

      completeOnboarding(resumeFile, text, targetRole.trim());
    } catch (err) {
      setError(err.message || 'Something went wrong. Is the backend running?');
      setIsLoading(false);
    }
  };

  const canSubmit = resumeFile && targetRole.trim() && !isLoading;

  return (
    <div className="relative min-h-screen bg-[#050508] overflow-hidden flex items-center justify-center p-6">
      {/* Glow orbs */}
      <GlowOrb color="radial-gradient(circle, #6366f1, transparent)" size="700px" top="-200px" left="-200px" />
      <GlowOrb color="radial-gradient(circle, #a855f7, transparent)" size="600px" top="50%" left="60%" />
      <GlowOrb color="radial-gradient(circle, #06b6d4, transparent)" size="400px" top="80%" left="5%" blur="120px" />

      {/* Grid overlay */}
      <div className="absolute inset-0 pointer-events-none z-0"
        style={{
          backgroundImage: 'linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      <div className="relative z-10 w-full max-w-lg animate-fade-up">

        {/* Logo / Brand */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-[0_0_30px_rgba(99,102,241,0.5)]">
              <span className="text-white font-black text-2xl leading-none">P</span>
            </div>
            <span className="text-2xl font-black bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 tracking-tight">
              PlaceBuddy
            </span>
          </div>

          <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/20 rounded-full px-4 py-1.5 mb-5 text-xs font-semibold text-indigo-300 backdrop-blur-sm">
            <Zap className="w-3 h-3" />
            Cognitive Assessment Engine — v2.0
          </div>

          <h1 className="text-4xl font-black text-white tracking-tighter leading-tight mb-3">
            Let's get you{' '}
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
              placement-ready.
            </span>
          </h1>
          <p className="text-zinc-400 text-sm leading-relaxed max-w-sm mx-auto">
            Upload your resume once. We extract the text locally — no AI analysis yet.
            Run Cognitive ATS and PRS Quiz when <em>you</em> choose.
          </p>
        </div>

        {/* Card */}
        <div
          className="relative rounded-3xl p-7 border border-white/8 backdrop-blur-xl"
          style={{ background: 'rgba(14,14,22,0.85)' }}
        >
          {/* Top accent */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-indigo-500/40 to-transparent rounded-t-3xl" />

          {/* Step 1: Resume */}
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-3">
              <span className="w-5 h-5 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-[10px] font-black text-indigo-400">1</span>
              <p className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Resume PDF</p>
            </div>

            {resumeFile ? (
              <div className="flex items-center justify-between bg-indigo-500/10 border border-indigo-500/25 rounded-2xl px-4 py-3.5">
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-indigo-400 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-semibold text-white truncate max-w-[240px]">{resumeFile.name}</p>
                    <p className="text-[10px] text-zinc-500 font-mono">{(resumeFile.size / 1024).toFixed(1)} KB · PDF</p>
                  </div>
                </div>
                <button
                  onClick={() => { setResumeFile(null); if (fileInputRef.current) fileInputRef.current.value = ''; }}
                  className="text-zinc-600 hover:text-zinc-300 transition-colors p-1"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-8 cursor-pointer transition-all duration-300 ${
                  isDragging
                    ? 'border-indigo-500/60 bg-indigo-500/10 scale-[1.01]'
                    : 'border-white/10 bg-white/[0.02] hover:border-indigo-500/40 hover:bg-indigo-500/5'
                }`}
              >
                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-300 ${isDragging ? 'bg-indigo-500/30 scale-110' : 'bg-white/5'}`}>
                  <UploadCloud className={`w-6 h-6 transition-colors ${isDragging ? 'text-indigo-300' : 'text-zinc-500'}`} />
                </div>
                <div className="text-center">
                  <p className="text-sm font-semibold text-zinc-300">Drop your resume here</p>
                  <p className="text-xs text-zinc-600 mt-0.5">or click to browse — PDF only</p>
                </div>
                <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleFileSelect} className="hidden" />
              </div>
            )}
          </div>

          {/* Step 2: Target Role */}
          <div className="mb-7">
            <div className="flex items-center gap-2 mb-3">
              <span className="w-5 h-5 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-[10px] font-black text-purple-400">2</span>
              <p className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Target Role</p>
            </div>
            <div className="relative">
              <Target className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-600 pointer-events-none" />
              <input
                type="text"
                value={targetRole}
                onChange={(e) => { setTargetRole(e.target.value); setError(''); }}
                onKeyDown={(e) => e.key === 'Enter' && canSubmit && handleSubmit()}
                placeholder="e.g. Backend Developer, Data Scientist..."
                className="w-full pl-11 pr-4 py-3.5 bg-white/[0.04] border border-white/10 rounded-2xl text-sm text-white placeholder:text-zinc-600 focus:outline-none focus:border-purple-500/50 focus:bg-white/[0.06] transition-all duration-200 font-medium"
              />
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="mb-5 flex items-center gap-2.5 bg-red-500/10 border border-red-500/25 rounded-xl px-4 py-3">
              <span className="w-1.5 h-1.5 rounded-full bg-red-400 flex-shrink-0" />
              <p className="text-xs font-medium text-red-300">{error}</p>
            </div>
          )}

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="w-full relative overflow-hidden bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-black py-4 px-6 rounded-2xl transition-all duration-300 shadow-[0_0_30px_rgba(99,102,241,0.3)] hover:shadow-[0_0_45px_rgba(99,102,241,0.5)] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none group"
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-3">
                <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span className="font-semibold">{loadingStage}</span>
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <Zap className="w-5 h-5" />
                Initialize PlaceBuddy
                <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </span>
            )}
            <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity rounded-2xl" />
          </button>

          {/* Privacy note */}
          <div className="mt-5 flex items-center justify-center gap-2 text-zinc-600 text-[10px] font-mono">
            <Shield className="w-3 h-3" />
            Text extraction only — no LLM called until you choose to run a module
            <Cpu className="w-3 h-3 ml-1" />
          </div>
        </div>

        <p className="text-center text-[10px] text-zinc-700 font-mono mt-6 tracking-wider">
          PLACEBUDDY · PHASE 2 · COGNITIVE ASSESSMENT ENGINE · 22CSP69
        </p>
      </div>
    </div>
  );
}
