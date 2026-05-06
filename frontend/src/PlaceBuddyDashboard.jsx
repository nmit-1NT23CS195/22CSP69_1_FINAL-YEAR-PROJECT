import React, { useState, useEffect, useRef } from 'react';
import { UserCircle, UploadCloud, FileText, ChevronDown, CheckCircle2, AlertCircle, XCircle } from 'lucide-react';

export default function PlaceBuddyDashboard() {
  const [appState, setAppState] = useState('idle'); // idle, analyzing, results_heatmap, results_detailed
  const [loadingText, setLoadingText] = useState('Segmenting document...');
  
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
    fetch('http://127.0.0.1:8080/roles/')
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
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setResumeFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async (type) => {
    if (!resumeFile) {
      alert("Please upload a resume first.");
      return;
    }

    setAppState('analyzing');
    
    const texts = ["Segmenting document...", "Running FlashText...", "Querying LLM..."];
    let i = 0;
    const interval = setInterval(() => {
      i++;
      if (i < texts.length) {
        setLoadingText(texts[i]);
      }
    }, 1000);

    try {
      // Construction of FormData for file payload
      const formData = new FormData();
      formData.append("resume", resumeFile);
      if (jdText) formData.append("jd_text", jdText);
      if (selectedRole) formData.append("role", selectedRole);

      // NO "Content-Type" header here, fetch handles it natively with boundary for FormData
      const response = await fetch("http://127.0.0.1:8080/ats/score", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        console.error("Backend returned error:", await response.text());
      } else {
        const data = await response.json();
        console.log("Success! Backend Data:", data);
        // You could set data to state here for real visualization
      }
    } catch (error) {
      console.error("Network or fetch error:", error);
    } finally {
      clearInterval(interval);
      setAppState(type);
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
          className={`relative group cursor-pointer h-[340px] rounded-3xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center overflow-hidden ${
            isDragging ? 'border-indigo-500 bg-zinc-800/80' : 'border-zinc-700 hover:border-indigo-500 bg-zinc-900/40 hover:bg-zinc-800/60'
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
            onChange={(e) => setJdText(e.target.value)}
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
          <span className="font-bold text-lg text-emerald-400 group-hover:text-emerald-300 drop-shadow-sm tracking-wide">Analyze Skill Gap Heatmap</span>
        </div>
      </button>

      <button 
        onClick={() => handleAnalyze('results_detailed')}
        className="flex-1 relative group overflow-hidden rounded-2xl p-[2px] transition-transform hover:-translate-y-1"
      >
        <span className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl opacity-70 group-hover:opacity-100 blur-sm transition-opacity duration-500"></span>
        <span className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl"></span>
        <div className="relative bg-zinc-950 px-8 py-5 rounded-2xl flex items-center justify-center gap-2 transition-all group-hover:bg-zinc-900 h-full">
          <span className="font-bold text-lg text-indigo-400 group-hover:text-indigo-300 drop-shadow-sm tracking-wide">Detailed ATS Scoring</span>
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
      <div className="text-sm font-bold text-zinc-400 uppercase tracking-[0.2em] mb-6 relative z-10">Placement Readiness Score</div>
      <div className="relative flex items-center justify-center">
        <svg className="w-48 h-48 transform -rotate-90">
          <circle cx="96" cy="96" r="84" stroke="currentColor" strokeWidth="10" fill="transparent" className="text-zinc-800" />
          <circle cx="96" cy="96" r="84" stroke="currentColor" strokeWidth="10" fill="transparent" strokeDasharray="527.7" strokeDashoffset="116.1" className="text-emerald-400 drop-shadow-[0_0_12px_rgba(52,211,153,0.8)] transition-all duration-1000 ease-out" />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-zinc-400">78</span>
          <span className="text-lg font-semibold text-zinc-500">/100</span>
        </div>
      </div>
    </div>
  );

  const renderResultsHeatmap = () => {
    const skills = [
      { name: 'React', status: 'applied' },
      { name: 'Node.js', status: 'applied' },
      { name: 'TypeScript', status: 'stated' },
      { name: 'GraphQL', status: 'missing' },
      { name: 'Tailwind CSS', status: 'applied' },
      { name: 'Docker', status: 'stated' },
      { name: 'Kubernetes', status: 'missing' },
      { name: 'AWS', status: 'missing' },
      { name: 'PostgreSQL', status: 'applied' },
      { name: 'Python', status: 'stated' },
    ];

    return (
      <div className="max-w-5xl mx-auto mt-12 px-8 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-20">
        <button onClick={() => setAppState('idle')} className="text-zinc-500 hover:text-white transition-colors text-sm font-semibold mb-8 flex items-center gap-2 bg-zinc-900/50 px-4 py-2 rounded-full border border-zinc-800 w-fit">
          ← Back to Dashboard
        </button>
        
        {renderScoreDisplay()}

        <h3 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
          Contextual Readiness Grid
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-5">
          {skills.map((s, i) => (
            <div key={i} className={`p-5 rounded-2xl border flex flex-col items-center justify-center gap-3 transition-all duration-300 hover:scale-105 hover:shadow-lg ${
              s.status === 'applied' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 shadow-[0_0_15px_rgba(52,211,153,0.15)]' :
              s.status === 'stated' ? 'bg-amber-500/10 border-amber-500/30 text-amber-400 shadow-[0_0_15px_rgba(251,191,36,0.15)]' :
              'bg-red-500/10 border-red-500/30 text-red-400 shadow-[0_0_15px_rgba(248,113,113,0.15)]'
            }`}>
              {s.status === 'applied' && <CheckCircle2 className="w-8 h-8 opacity-80" />}
              {s.status === 'stated' && <AlertCircle className="w-8 h-8 opacity-80" />}
              {s.status === 'missing' && <XCircle className="w-8 h-8 opacity-80" />}
              <span className="font-bold text-center">{s.name}</span>
            </div>
          ))}
        </div>
        
        <div className="flex flex-wrap gap-8 mt-12 justify-center text-sm font-semibold bg-zinc-900/50 p-6 rounded-3xl border border-zinc-800">
          <div className="flex items-center gap-2 text-emerald-400 bg-emerald-500/10 px-4 py-2 rounded-xl"><CheckCircle2 className="w-5 h-5"/> Applied Context (Verified)</div>
          <div className="flex items-center gap-2 text-amber-400 bg-amber-500/10 px-4 py-2 rounded-xl"><AlertCircle className="w-5 h-5"/> Stated Only (Unverified)</div>
          <div className="flex items-center gap-2 text-red-400 bg-red-500/10 px-4 py-2 rounded-xl"><XCircle className="w-5 h-5"/> Missing Skill (Gap)</div>
        </div>
      </div>
    );
  };

  const renderResultsDetailed = () => {
    const signals = [
      { name: "Semantic Similarity", value: "84%", desc: "Cosine similarity between profile and JD", color: "text-indigo-400", bg: "bg-indigo-500/5", border: "border-indigo-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(99,102,241,0.2)]" },
      { name: "TF-IDF Relevance", value: "72%", desc: "Keyword frequency alignment", color: "text-blue-400", bg: "bg-blue-500/5", border: "border-blue-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(59,130,246,0.2)]" },
      { name: "Contextual Weighting", value: "18", desc: "Skills verified in context", color: "text-emerald-400", bg: "bg-emerald-500/5", border: "border-emerald-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(52,211,153,0.2)]" },
      { name: "Experience Est.", value: "4.2 Yrs", desc: "NER-extracted timeline", color: "text-purple-400", bg: "bg-purple-500/5", border: "border-purple-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(168,85,247,0.2)]" },
      { name: "Action Verbs", value: "24", desc: "Impact-driven vocabulary used", color: "text-pink-400", bg: "bg-pink-500/5", border: "border-pink-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(236,72,153,0.2)]" },
      { name: "Soft Skills", value: "6", desc: "Interpersonal attributes identified", color: "text-amber-400", bg: "bg-amber-500/5", border: "border-amber-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(251,191,36,0.2)]" },
      { name: "Format Parse", value: "100%", desc: "Structure & readability score", color: "text-teal-400", bg: "bg-teal-500/5", border: "border-teal-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(20,184,166,0.2)]" },
    ];

    return (
      <div className="max-w-6xl mx-auto mt-12 px-8 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-20">
        <button onClick={() => setAppState('idle')} className="text-zinc-500 hover:text-white transition-colors text-sm font-semibold mb-8 flex items-center gap-2 bg-zinc-900/50 px-4 py-2 rounded-full border border-zinc-800 w-fit">
          ← Back to Dashboard
        </button>
        
        {renderScoreDisplay()}

        <h3 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
          ML Signal Breakdown
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
          {signals.map((sig, i) => (
            <div key={i} className={`p-6 rounded-3xl border ${sig.bg} ${sig.border} backdrop-blur-md relative overflow-hidden group transition-all duration-300 hover:-translate-y-1 ${sig.glow}`}>
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-2">{sig.name}</div>
              <div className={`text-4xl font-black ${sig.color} mb-3 drop-shadow-md`}>{sig.value}</div>
              <div className="text-sm text-zinc-500 font-medium leading-relaxed">{sig.desc}</div>
            </div>
          ))}
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
