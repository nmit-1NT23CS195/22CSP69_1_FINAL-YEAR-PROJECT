import React, { useState, useEffect, useRef } from 'react';
import { UserCircle, UploadCloud, FileText, ChevronDown, CheckCircle2, AlertCircle, XCircle, Target, Compass, BrainCircuit, Rocket, Activity, Briefcase, Lightbulb, ArrowLeft, ArrowRight } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
const API_BASE_URL = "http://127.0.0.1:8080";

const formatSkillName = (skill) => {
  if (!skill) return "";
  const acronyms = ["AWS", "API", "SQL", "GCP", "CSS", "HTML", "PHP", "UI", "UX", "CI/CD", "SEO", "NLP", "LLM", "REST", "JSON", "XML"];
  const upper = skill.toUpperCase();
  if (acronyms.includes(upper)) return upper;

  return skill.split(/[- ]/).map(word => {
    if (acronyms.includes(word.toUpperCase())) return word.toUpperCase();
    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
  }).join(' ');
};

export default function PlaceBuddyDashboard() {
  const [appState, setAppState] = useState('idle'); // idle, analyzing, results_heatmap, results_detailed
  const [loadingText, setLoadingText] = useState('Processing Analysis...');
  const [coreResult, setCoreResult] = useState(null);
  const [deepResult, setDeepResult] = useState(null);
  const [isDeepLoading, setIsDeepLoading] = useState(false);

  const resetResults = () => {
    setCoreResult(null);
    setDeepResult(null);
    setIsDeepLoading(false);
  };

  // Input State
  const [jdText, setJdText] = useState('');
  const [resumeFile, setResumeFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Dropdown State
  const [roles, setRoles] = useState([]);
  const [searchRole, setSearchRole] = useState('');
  const [selectedRole, setSelectedRole] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    // Fetch roles on mount 
    fetch(`${API_BASE_URL}/roles/`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setRoles(data);
        } else if (data && Array.isArray(data.roles)) {
          setRoles(data.roles);
        } else {
          setRoles(["Backend Developer", "Data Scientist", "Frontend Engineer", "Product Manager", "UI/UX Designer"]);
        }
      })
      .catch(err => {
        console.error("Failed to fetch roles", err);
        setRoles(["Backend Developer", "Data Scientist", "Frontend Engineer", "Product Manager", "UI/UX Designer"]);
      });

    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filteredRoles = roles.filter(r => r.toLowerCase().includes(searchRole.toLowerCase()));

  // Drag & Drop Handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setResumeFile(e.dataTransfer.files[0]);
      resetResults();
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setResumeFile(e.target.files[0]);
      resetResults();
    }
  };

  const handleAnalyze = async (type) => {
    if (!resumeFile) {
      alert("Please upload a resume first.");
      return;
    }

    if (coreResult) {
      setAppState(type);
      return;
    }

    setAppState('analyzing');
    setLoadingText('Processing Analysis...');
    resetResults();

    try {
      const formData = new FormData();
      formData.append("resume", resumeFile);
      if (jdText) formData.append("jd_text", jdText);
      if (selectedRole) formData.append("role", selectedRole);

      const coreRes = await fetch(`${API_BASE_URL}/ats/score/core`, {
        method: "POST",
        body: formData,
      });

      if (!coreRes.ok) {
        console.error("Backend returned error:", await coreRes.text());
        setAppState('idle');
      } else {
        const data = await coreRes.json();
        setCoreResult(data);
        setAppState(type);

        const extractedResumeText = data.extracted_resume_text || "";
        const extractedJdText = data.extracted_jd_text || jdText || "";

        if (extractedResumeText && extractedJdText) {
          setIsDeepLoading(true);
          const deepFormData = new FormData();
          deepFormData.append("resume_text", extractedResumeText);
          deepFormData.append("jd_text", extractedJdText);

          fetch(`${API_BASE_URL}/ats/score/deep`, {
            method: "POST",
            body: deepFormData,
          })
            .then(res => res.ok ? res.json() : Promise.reject(res))
            .then(deepData => { setDeepResult(deepData); })
            .catch(err => console.error("Deep analysis error:", err))
            .finally(() => setIsDeepLoading(false));
        }
      }
    } catch (error) {
      console.error("Network or fetch error:", error);
      setAppState('idle');
    }
  };

  const renderNavbar = () => (
    <nav className="flex items-center justify-between px-8 py-4 border-b border-white/10 bg-zinc-900/40 backdrop-blur-xl sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-[0_0_20px_rgba(99,102,241,0.5)]">
          <span className="text-white font-black text-2xl leading-none">P</span>
        </div>
        <h1 className="text-2xl font-black bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 drop-shadow-[0_0_10px_rgba(168,85,247,0.4)] tracking-tight">
          PlaceBuddy AI
        </h1>
      </div>
      <div className="flex items-center gap-8 text-sm font-semibold text-zinc-300">
        <a href="#" className="hover:text-white transition-colors">Home</a>
        <a href="#" className="text-white border-b-2 border-indigo-500 pb-1">Dashboard</a>
        <a href="#" className="hover:text-white transition-colors">Upskill</a>
        <button className="flex items-center gap-2 hover:text-white transition-colors ml-4 bg-white/5 p-2 rounded-full border border-white/10 hover:bg-white/10">
          <UserCircle className="w-6 h-6 text-zinc-300" />
        </button>
      </div>
    </nav>
  );

  const renderInputZone = () => (
    <div className="max-w-6xl mx-auto mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8 px-8 animate-in fade-in slide-in-from-bottom-8 duration-700">
      <div className="flex flex-col gap-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <FileText className="w-6 h-6 text-indigo-400 drop-shadow-[0_0_5px_rgba(99,102,241,0.8)]" /> Resume
        </h2>
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative group cursor-pointer h-[340px] rounded-3xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center overflow-hidden ${isDragging ? 'border-indigo-500 bg-zinc-800/80' : 'border-zinc-700 hover:border-indigo-500 bg-zinc-900/40 hover:bg-zinc-800/60'
            } backdrop-blur-md`}
        >
          <input
            type="file"
            accept=".pdf,.docx,.txt"
            ref={fileInputRef}
            onChange={handleFileSelect}
            className="hidden"
          />
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

          {resumeFile ? (
            <div className="flex flex-col items-center z-10 px-6 text-center">
              <FileText className="w-16 h-16 text-emerald-400 mb-6 drop-shadow-lg" />
              <p className="text-emerald-300 font-bold text-xl break-all">{resumeFile.name}</p>
              <p className="text-zinc-400 text-sm mt-3">Click or drag to replace</p>
            </div>
          ) : (
            <div className="flex flex-col items-center z-10">
              <UploadCloud className="w-16 h-16 text-zinc-500 group-hover:text-indigo-400 group-hover:scale-110 transition-all duration-500 mb-6 drop-shadow-lg" />
              <p className="text-zinc-200 font-semibold text-lg">Drag & drop your Resume PDF</p>
              <p className="text-zinc-500 text-sm mt-2">or click to browse local files</p>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-4 h-full">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <CheckCircle2 className="w-6 h-6 text-emerald-400 drop-shadow-[0_0_5px_rgba(52,211,153,0.8)]" /> Target Requirements
        </h2>

        <div className="flex flex-col gap-2 h-full">
          <label className="text-xs font-semibold text-zinc-400 ml-1 uppercase tracking-wider">Job Description (Optional)</label>
          <textarea
            value={jdText}
            onChange={(e) => {
              setJdText(e.target.value);
              resetResults();
            }}
            placeholder="Paste the job description here..."
            className="flex-1 min-h-[180px] resize-none rounded-2xl bg-zinc-900/40 backdrop-blur-md border border-zinc-700/50 text-zinc-200 p-5 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all duration-300 placeholder:text-zinc-600 shadow-inner"
          />
        </div>

        <div className="flex flex-col gap-2 relative mt-2" ref={dropdownRef}>
          <label className="text-xs font-semibold text-zinc-400 ml-1 uppercase tracking-wider">Target Role</label>
          <div
            className="relative"
            onClick={() => setIsDropdownOpen(true)}
          >
            <input
              type="text"
              value={isDropdownOpen ? searchRole : (selectedRole || searchRole)}
              onChange={(e) => {
                setSearchRole(e.target.value);
                setIsDropdownOpen(true);
                resetResults();
              }}
              placeholder="Search or select a role..."
              className="w-full rounded-2xl bg-zinc-900/40 backdrop-blur-md border border-zinc-700/50 text-zinc-200 px-5 py-4 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all duration-300 pr-12 shadow-inner"
            />
            <ChevronDown className={`absolute right-4 top-4 w-6 h-6 text-zinc-500 transition-transform duration-300 ${isDropdownOpen ? 'rotate-180 text-indigo-400' : ''}`} />
          </div>

          {isDropdownOpen && (
            <div className="absolute top-full left-0 right-0 mt-2 max-h-60 overflow-y-auto rounded-2xl bg-zinc-800/90 backdrop-blur-xl border border-zinc-700 shadow-2xl z-20 p-2 scrollbar-thin scrollbar-thumb-zinc-600">
              {filteredRoles.length > 0 ? filteredRoles.map((role, idx) => (
                <div
                  key={idx}
                  onClick={() => {
                    setSelectedRole(role);
                    setSearchRole('');
                    setIsDropdownOpen(false);
                    resetResults();
                  }}
                  className="px-4 py-3 hover:bg-indigo-500/20 hover:text-indigo-300 text-zinc-300 cursor-pointer rounded-xl transition-colors font-medium"
                >
                  {role}
                </div>
              )) : (
                <div className="px-4 py-3 text-zinc-500 text-center italic">No roles found</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderActionButtons = () => (
    <div className="max-w-4xl mx-auto mt-12 flex flex-col sm:flex-row gap-8 justify-center px-8 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-200">
      <button
        onClick={() => handleAnalyze('results_heatmap')}
        className="flex-1 relative group overflow-hidden rounded-2xl p-[2px] transition-transform hover:-translate-y-1"
      >
        <span className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-teal-500 rounded-2xl opacity-70 group-hover:opacity-100 blur-sm transition-opacity duration-500"></span>
        <span className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-teal-500 rounded-2xl"></span>
        <div className="relative bg-zinc-950 px-8 py-5 rounded-2xl flex items-center justify-center gap-2 transition-all group-hover:bg-zinc-900 h-full">
          <Activity className="w-5 h-5 text-emerald-400" />
          <span className="font-bold text-lg text-emerald-400 group-hover:text-emerald-300 drop-shadow-sm tracking-wide">Forensic Skill Matrix</span>
        </div>
      </button>

      <button
        onClick={() => handleAnalyze('results_detailed')}
        className="flex-1 relative group overflow-hidden rounded-2xl p-[2px] transition-transform hover:-translate-y-1"
      >
        <span className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl opacity-70 group-hover:opacity-100 blur-sm transition-opacity duration-500"></span>
        <span className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl"></span>
        <div className="relative bg-zinc-950 px-8 py-5 rounded-2xl flex items-center justify-center gap-2 transition-all group-hover:bg-zinc-900 h-full">
          <BrainCircuit className="w-5 h-5 text-indigo-400" />
          <span className="font-bold text-lg text-indigo-400 group-hover:text-indigo-300 drop-shadow-sm tracking-wide">Cognitive ATS Insights</span>
        </div>
      </button>
    </div>
  );

  const renderAnalyzingView = () => (
    <div className="flex flex-col items-center justify-center mt-32 animate-in fade-in zoom-in duration-500">
      <div className="relative flex items-center justify-center w-40 h-40 mb-10">
        <div className="absolute inset-0 rounded-full border-t-4 border-indigo-500 animate-spin opacity-80" style={{ animationDuration: '3s' }}></div>
        <div className="absolute inset-3 rounded-full border-r-4 border-purple-500 animate-spin opacity-80" style={{ animationDirection: 'reverse', animationDuration: '2s' }}></div>
        <div className="absolute inset-6 rounded-full border-b-4 border-emerald-500 animate-spin opacity-80" style={{ animationDuration: '1.5s' }}></div>
        <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full animate-pulse shadow-[0_0_50px_rgba(99,102,241,0.8)] flex items-center justify-center">
          <span className="text-white font-bold tracking-widest text-xs opacity-50">AI</span>
        </div>
      </div>
      <h2 className="text-3xl font-black text-white tracking-wider mb-3">Analyzing Profile</h2>
      <p className="text-indigo-400 font-mono text-lg animate-pulse">{loadingText}</p>
    </div>
  );

  const renderScoreDisplay = () => (
    <div className="flex flex-col items-center justify-center p-10 bg-zinc-900/40 rounded-[2.5rem] border border-white/10 backdrop-blur-xl mb-12 shadow-[0_0_50px_rgba(0,0,0,0.4)] relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent pointer-events-none"></div>
      <div className="text-sm font-bold text-zinc-400 uppercase tracking-[0.2em] mb-6 relative z-10">ATS Score</div>
      <div className="relative flex items-center justify-center">
        <svg className="w-48 h-48 transform -rotate-90">
          <circle cx="96" cy="96" r="84" stroke="currentColor" strokeWidth="10" fill="transparent" className="text-zinc-800" />
          <circle cx="96" cy="96" r="84" stroke="currentColor" strokeWidth="10" fill="transparent" strokeDasharray="527.7" strokeDashoffset={527.7 - (527.7 * (coreResult ? Math.round(coreResult.ats_score) : 0)) / 100} className="text-emerald-400 drop-shadow-[0_0_12px_rgba(52,211,153,0.8)] transition-all duration-1000 ease-out" />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-zinc-400">{coreResult ? Math.round(coreResult.ats_score) : 0}</span>
          <span className="text-lg font-semibold text-zinc-500">/100</span>
        </div>
      </div>
    </div>
  );

  const renderResultsHeatmap = () => {
    let radarData = [];
    let bsDetector = [];

    if (coreResult?.cognitive_analysis) {
      const { skill_matrix, bullshit_detector } = coreResult.cognitive_analysis;
      bsDetector = bullshit_detector || [];
      if (skill_matrix) {
        if (Array.isArray(skill_matrix)) {
          radarData = skill_matrix.map((data) => ({
            skill: formatSkillName(data.skill_name),
            proficiency: data.proficiency_score || 0,
            yoe: data.estimated_yoe || 0
          }));
        } else {
          radarData = Object.entries(skill_matrix).map(([skill, data]) => ({
            skill: formatSkillName(skill),
            proficiency: data.proficiency_score || 0,
            yoe: data.estimated_yoe || 0
          }));
        }

        // Clamp data to top 15 skills to prevent visual overlap on Radar Chart
        radarData = radarData.sort((a, b) => b.proficiency - a.proficiency).slice(0, 15);
      }
    }

    return (
      <div className="max-w-6xl mx-auto mt-12 px-8 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-20">
        <button onClick={() => setAppState('idle')} className="flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200 transition-colors font-medium mb-8">
          <ArrowLeft className="w-4 h-4" /> Back to Upload Screen
        </button>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="bg-zinc-900/40 border border-indigo-500/30 rounded-[2rem] p-8 backdrop-blur-xl shadow-[0_0_30px_rgba(99,102,241,0.1)] relative overflow-hidden group h-fit lg:col-span-1">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent"></div>
            <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3 relative z-10">
              <Activity className="w-7 h-7 text-indigo-400" />
              Forensic Skill Matrix
            </h3>

            {radarData.length > 0 ? (
              <div className="h-[400px] w-full relative z-10">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                    <PolarAngleAxis dataKey="skill" tick={{ fill: '#a1a1aa', fontSize: 11, fontWeight: 'bold' }} />
                    <PolarRadiusAxis angle={30} domain={[0, 10]} tick={{ fill: '#52525b', fontSize: 10 }} />
                    <Radar name="Proficiency" dataKey="proficiency" stroke="#8b5cf6" strokeWidth={2} fill="#8b5cf6" fillOpacity={0.35} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="flex items-center justify-center h-[400px] text-zinc-500 italic">No skill matrix data available.</div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:col-span-2 h-fit">

            <div className="bg-zinc-900/40 border border-emerald-500/30 rounded-[2rem] p-8 backdrop-blur-xl shadow-[0_0_30px_rgba(52,211,153,0.1)] relative overflow-hidden h-fit lg:col-span-2">
              <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent"></div>
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3 relative z-10">
                <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                Verified Competencies
              </h3>
              <div className="flex flex-wrap gap-3 w-full relative z-10">
                {radarData.filter(r => r.proficiency >= 5).map((r, idx) => (
                  <div key={idx} className="inline-flex flex-col items-start justify-center px-4 py-2 rounded-lg bg-emerald-900/20 border border-emerald-800/50 w-fit text-emerald-400 font-semibold text-sm">
                    <span>{r.skill}</span>
                    <span className="text-xs opacity-70 mt-1">{r.yoe} Yrs Exp</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-zinc-900/40 border border-amber-500/30 rounded-[2rem] p-8 backdrop-blur-xl shadow-[0_0_30px_rgba(251,191,36,0.1)] relative overflow-hidden h-fit lg:col-span-1">
              <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-transparent"></div>
              <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-3 relative z-10">
                <AlertCircle className="w-7 h-7 text-amber-400" />
                Unverified Skills (Stated Only)
              </h3>
              <p className="text-sm text-zinc-400 mb-6 relative z-10">
                The LLM Panel found no concrete project evidence for the following skills. Be prepared to defend these in an interview.
              </p>

              <div className="flex flex-wrap gap-3 w-full relative z-10">
                {bsDetector.length > 0 ? bsDetector.map((skill, idx) => (
                  <div key={idx} className="inline-flex flex-col items-start justify-center px-4 py-2 rounded-lg bg-amber-900/20 border border-amber-800/50 w-fit text-amber-400 font-semibold text-sm">
                    <div className="flex items-center gap-2">
                      <XCircle className="w-4 h-4" /> {formatSkillName(skill)}
                    </div>
                  </div>
                )) : (
                  <div className="text-emerald-400 flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/30 px-4 py-2 rounded-xl font-semibold">
                    <CheckCircle2 className="w-5 h-5" /> All stated skills verified!
                  </div>
                )}
              </div>
            </div>

            <div className="bg-zinc-900/40 border border-red-500/30 rounded-[2rem] p-8 backdrop-blur-xl shadow-[0_0_30px_rgba(239,68,68,0.1)] relative overflow-hidden h-fit lg:col-span-1">
              <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-transparent"></div>
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3 relative z-10">
                <AlertCircle className="w-6 h-6 text-red-400" />
                Missing Skills (Critical Gaps)
              </h3>
              <div className="flex flex-wrap gap-3 w-full relative z-10">
                {coreResult?.missing_skills?.length > 0 ? (
                  coreResult.missing_skills.map((skill, idx) => (
                    <div key={idx} className="inline-flex flex-col items-start justify-center px-4 py-2 rounded-lg bg-red-900/20 border border-red-800/50 w-fit text-red-400 font-semibold text-sm">
                      {formatSkillName(skill)}
                    </div>
                  ))
                ) : (
                  <div className="text-emerald-400 flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/30 px-4 py-2 rounded-xl font-semibold">
                    <CheckCircle2 className="w-5 h-5" /> No critical missing skills!
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>
      </div>
    );
  };

  const renderResultsDetailed = () => {
    const signals = coreResult ? [
      { name: "Semantic Similarity", value: `${(coreResult.keyword_metrics?.semantic_similarity * 100 || 0).toFixed(0)}%`, desc: "Cosine similarity between profile and JD", color: "text-indigo-400", bg: "bg-indigo-500/5", border: "border-indigo-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(99,102,241,0.2)]" },
      { name: "TF-IDF Match", value: `${(coreResult.keyword_metrics?.tfidf_score * 100 || 0).toFixed(0)}%`, desc: "Keyword frequency alignment", color: "text-blue-400", bg: "bg-blue-500/5", border: "border-blue-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(59,130,246,0.2)]" },
      { name: "Skill Alignment", value: `${Object.values(coreResult.contextual_skill_weights || {}).reduce((acc, w) => acc + (w > 1 ? 1 : 0), 0)}`, desc: "Skills verified in context", color: "text-emerald-400", bg: "bg-emerald-500/5", border: "border-emerald-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(52,211,153,0.2)]" },
      { name: "Experience Est.", value: `${coreResult.estimated_experience?.total_yoe || 0} Yrs`, desc: "NER-extracted timeline", color: "text-purple-400", bg: "bg-purple-500/5", border: "border-purple-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(168,85,247,0.2)]" },
      { name: "Resume Impact", value: `${coreResult.action_verbs_found?.length || 0}`, desc: "Impact-driven action verbs used", color: "text-pink-400", bg: "bg-pink-500/5", border: "border-pink-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(236,72,153,0.2)]" },
      { name: "Soft Skills", value: `${coreResult.soft_skills_found?.length || 0}`, desc: "Interpersonal attributes identified", color: "text-amber-400", bg: "bg-amber-500/5", border: "border-amber-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(251,191,36,0.2)]" },
      { name: "Keyword Match", value: `${(coreResult.keyword_metrics?.keyword_match || 0).toFixed(0)}%`, desc: "Stop-word filtered keyword match", color: "text-teal-400", bg: "bg-teal-500/5", border: "border-teal-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(20,184,166,0.2)]" },
    ] : [];

    const cog = coreResult?.cognitive_analysis || {};
    const roles = cog.best_fit_roles || [];
    const pivots = cog.pivot_opportunities || [];
    const questions = deepResult?.targeted_questions || [];
    const dsaBridge = deepResult?.dsa_bridge || [];
    const microProject = deepResult?.micro_project_suggestion || "";

    return (
      <div className="max-w-6xl mx-auto mt-12 px-8 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-20">
        <button onClick={() => setAppState('idle')} className="flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200 transition-colors font-medium mb-8">
          <ArrowLeft className="w-4 h-4" /> Back to Upload Screen
        </button>

        {renderScoreDisplay()}

        <h3 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
          ML Signal Breakdown
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-12">
          {signals.map((sig, i) => (
            <div key={i} className={`p-6 rounded-3xl border ${sig.bg} ${sig.border} backdrop-blur-md relative overflow-hidden group transition-all duration-300 hover:-translate-y-1 ${sig.glow}`}>
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-2">{sig.name}</div>
              <div className={`text-4xl font-black ${sig.color} mb-3 drop-shadow-md`}>{sig.value}</div>
              <div className="text-sm text-zinc-500 font-medium leading-relaxed">{sig.desc}</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Constellation View */}
          <div className="lg:col-span-1 flex flex-col gap-8">
            <div className="bg-zinc-900/40 border border-teal-500/30 rounded-[2rem] p-8 backdrop-blur-xl shadow-[0_0_30px_rgba(20,184,166,0.1)] relative overflow-hidden group h-fit">
              <div className="absolute inset-0 bg-gradient-to-br from-teal-500/5 to-transparent"></div>
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3 relative z-10">
                <Compass className="w-6 h-6 text-teal-400" />
                Career Constellation
              </h3>

              <div className="flex flex-col gap-4 relative z-10">
                {roles.map((r, i) => (
                  <div key={i} className="bg-teal-500/5 border border-teal-500/20 rounded-xl p-4 transition-all hover:bg-teal-500/10 hover:border-teal-500/40 hover:-translate-y-1 group/role cursor-pointer relative">
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-teal-300 flex items-center gap-2">
                        {r.role}
                        <ChevronDown className="w-4 h-4 opacity-50 group-hover/role:rotate-180 transition-transform" />
                      </span>
                      <span className="text-teal-400 font-black text-sm bg-teal-500/20 px-2 py-1 rounded-md">{r.match_percentage}% Match</span>
                    </div>
                    <div className="max-h-0 opacity-0 overflow-hidden group-hover/role:max-h-[500px] group-hover/role:opacity-100 group-hover/role:mt-4 transition-all duration-500 ease-in-out">
                      <p className="text-xs text-zinc-400 leading-relaxed border-t border-teal-500/10 pt-3">{r.rationale}</p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-8 relative z-10 border-t border-white/5 pt-6">
                <h4 className="text-sm font-bold text-zinc-400 mb-4 flex items-center gap-2 uppercase tracking-wider">
                  <Rocket className="w-4 h-4" /> Pivot Opportunities
                </h4>
                <div className="flex flex-wrap gap-2">
                  {pivots.map((p, i) => (
                    <span key={i} className="text-xs font-semibold bg-zinc-800 text-zinc-300 border border-zinc-700 px-3 py-1.5 rounded-full">
                      {p}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Interview Prep & Roadmap */}
          <div className="lg:col-span-2 flex flex-col gap-8">
            {/* The Brutal Interviewer */}
            <div className="bg-zinc-900/40 border border-rose-500/30 rounded-[2rem] p-8 backdrop-blur-xl shadow-[0_0_30px_rgba(244,63,94,0.1)] relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 to-transparent"></div>
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3 relative z-10">
                <Target className="w-6 h-6 text-rose-400" />
                The "Brutal Interviewer" Simulator
              </h3>

              <div className="grid grid-cols-1 gap-4 relative z-10">
                {isDeepLoading ? (
                  <div className="text-zinc-400 italic animate-pulse">Loading deep analysis questions...</div>
                ) : questions.length > 0 ? questions.map((q, i) => (
                  <div key={i} className="bg-rose-500/5 border-l-4 border-rose-500/50 p-4 rounded-r-xl text-zinc-300 text-sm font-medium group/q cursor-pointer hover:bg-rose-500/10 transition-all relative">
                    <div className="flex justify-between items-center">
                      <span className="text-rose-400 font-bold whitespace-nowrap">Question {i + 1}</span>
                      <ChevronDown className="w-4 h-4 opacity-50 group-hover/q:rotate-180 transition-transform" />
                    </div>
                    <div className="max-h-0 opacity-0 overflow-hidden group-hover/q:max-h-[500px] group-hover/q:opacity-100 group-hover/q:mt-3 transition-all duration-500 ease-in-out">
                      <p className="text-zinc-300 leading-relaxed text-sm">{q}</p>
                    </div>
                  </div>
                )) : (
                  <div className="text-zinc-500 italic text-sm">No targeted questions available.</div>
                )}
              </div>

              {dsaBridge.length > 0 && (
                <div className="mt-6 relative z-10 bg-zinc-950/50 rounded-xl p-5 border border-white/5">
                  <h4 className="text-sm font-bold text-rose-300 mb-3 flex items-center gap-2">
                    <Briefcase className="w-4 h-4" /> Real-World DSA Bridge
                  </h4>
                  <div className="flex flex-col gap-3">
                    {dsaBridge.map((bridge, i) => (
                      <div key={i} className="bg-zinc-950/80 rounded-lg p-4 border border-zinc-800/50 group/dsa cursor-pointer transition-all hover:border-zinc-700">
                        <div className="flex justify-between items-center">
                          <span className="text-rose-400 font-bold text-xs uppercase tracking-wider">{bridge.dsa_concept}</span>
                          <ChevronDown className="w-4 h-4 opacity-50 text-zinc-500 group-hover/dsa:rotate-180 transition-transform" />
                        </div>
                        <div className="max-h-0 opacity-0 overflow-hidden group-hover/dsa:max-h-[800px] group-hover/dsa:opacity-100 group-hover/dsa:mt-4 transition-all duration-500 ease-in-out">
                          <div className="flex flex-col sm:flex-row gap-4 text-xs">
                            <div className="flex-1 bg-zinc-900 p-4 rounded-xl border border-zinc-800 text-zinc-400 relative">
                              <span className="block text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Project Logic</span>
                              {bridge.project_logic}
                              <div className="absolute top-1/2 -right-5 -translate-y-1/2 w-6 h-6 bg-zinc-800 rounded-full flex flex-col items-center justify-center border border-zinc-700 hidden sm:flex z-10 shadow-lg">
                                <ArrowRight className="w-3 h-3 text-rose-400" />
                              </div>
                            </div>
                            <div className="flex-1 bg-rose-500/10 p-4 rounded-xl border border-rose-500/20 text-rose-300 font-semibold flex items-center justify-center text-center">
                              {bridge.dsa_concept}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Actionable Roadmap */}
            <div className="bg-zinc-900/40 border border-sky-500/30 rounded-[2rem] p-8 backdrop-blur-xl shadow-[0_0_30px_rgba(14,165,233,0.1)] relative overflow-hidden group/road h-fit cursor-pointer">
              <div className="absolute inset-0 bg-gradient-to-br from-sky-500/5 to-transparent"></div>
              <div className="flex justify-between items-center relative z-10">
                <h3 className="text-xl font-bold text-white flex items-center gap-3">
                  <Lightbulb className="w-6 h-6 text-sky-400" />
                  Actionable Micro-Project Roadmap
                </h3>
                <ChevronDown className="w-5 h-5 text-sky-400 opacity-50 group-hover/road:rotate-180 transition-transform" />
              </div>
              <div className="max-h-0 opacity-0 overflow-hidden group-hover/road:max-h-[1000px] group-hover/road:opacity-100 group-hover/road:mt-6 transition-all duration-700 ease-in-out relative z-10">
                <p className="text-zinc-300 leading-relaxed text-sm bg-sky-500/10 border border-sky-500/20 p-5 rounded-xl font-medium">
                  {isDeepLoading ? (
                    <span className="text-sky-300 italic animate-pulse">Generating actionable micro-project...</span>
                  ) : (
                    microProject || "No roadmap data generated."
                  )}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 font-sans selection:bg-indigo-500/30 overflow-x-hidden">
      {/* Background glowing effects */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-indigo-900/20 rounded-full blur-[150px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-purple-900/20 rounded-full blur-[150px]" />
      </div>

      <div className="relative z-10">
        {renderNavbar()}

        {appState === 'idle' && (
          <div className="pb-24">
            {renderInputZone()}
            {renderActionButtons()}
          </div>
        )}

        {appState === 'analyzing' && renderAnalyzingView()}
        {appState === 'results_heatmap' && renderResultsHeatmap()}
        {appState === 'results_detailed' && renderResultsDetailed()}
      </div>
    </div>
  );
}
