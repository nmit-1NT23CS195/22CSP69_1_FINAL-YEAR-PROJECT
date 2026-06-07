import React, { useState, useEffect, useRef } from 'react';
import { UserCircle, UploadCloud, FileText, ChevronDown, CheckCircle2, AlertCircle, XCircle, Target, Compass, BrainCircuit, Rocket, Activity, Briefcase, Lightbulb, ArrowLeft, ArrowRight, Eye, EyeOff, Download, History, Trash2, Clock } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import jsPDF from 'jspdf';
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
  const [revealedAnswers, setRevealedAnswers] = useState({});

  const toggleAnswer = (index) => {
    setRevealedAnswers(prev => ({ ...prev, [index]: !prev[index] }));
  };

  const [recentAnalyses, setRecentAnalyses] = useState(() => {
    try {
      const stored = localStorage.getItem('placebuddy_recent');
      return stored ? JSON.parse(stored) : [];
    } catch (e) {
      console.error("Failed to load recent analyses", e);
      return [];
    }
  });
  const [currentAnalysisId, setCurrentAnalysisId] = useState(null);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

  // Staged loading screen states
  const stages = [
    "Parsing Resume",
    "Extracting Skills",
    "Matching Against Job Requirements",
    "Calculating ATS Score",
    "Generating Cognitive Insights",
    "Finalizing Report"
  ];
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [analysisResultData, setAnalysisResultData] = useState(null);
  const [targetType, setTargetType] = useState(null);

  // Drive progression through stages
  useEffect(() => {
    if (appState !== 'analyzing') return;

    const hasData = analysisResultData !== null;
    const intervalTime = hasData ? 400 : 1500;

    const timer = setTimeout(() => {
      if (currentStageIndex < stages.length - 1) {
        setCurrentStageIndex(prev => prev + 1);
      } else if (hasData && targetType) {
        setCoreResult(analysisResultData);
        setAppState(targetType);
      }
    }, intervalTime);

    return () => clearTimeout(timer);
  }, [appState, currentStageIndex, analysisResultData, targetType]);

  const handleNavClick = (sectionId) => {
    if (appState !== 'idle') {
      setAppState('idle');
      setTimeout(() => {
        const element = document.getElementById(sectionId);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } else {
      const element = document.getElementById(sectionId);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  const resetResults = () => {
    setCoreResult(null);
    setDeepResult(null);
    setIsDeepLoading(false);
    setIsAdvancedOpen(false);
  };

  const deleteRecentEntry = (id) => {
    setRecentAnalyses(prev => {
      const updated = prev.filter(item => item.id !== id);
      localStorage.setItem('placebuddy_recent', JSON.stringify(updated));
      return updated;
    });
  };

  const clearRecentEntries = () => {
    if (window.confirm("Clear all recent analyses?")) {
      setRecentAnalyses([]);
      localStorage.removeItem('placebuddy_recent');
    }
  };

  const downloadPdfDataUri = (dataUri, filename) => {
    try {
      const link = document.createElement('a');
      link.href = dataUri;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error("Failed to download stored PDF", err);
    }
  };

  // Input State
  const [jdText, setJdText] = useState('');
  const [jdFile, setJdFile] = useState(null);
  const [resumeFile, setResumeFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const jdFileInputRef = useRef(null);

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
    setCurrentStageIndex(0);
    setAnalysisResultData(null);
    setTargetType(null);
    resetResults();

    try {
      const formData = new FormData();
      formData.append("resume", resumeFile);
      if (jdText) formData.append("jd_text", jdText);
      if (jdFile) formData.append("jd_file", jdFile);
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

        const newId = Date.now().toString();
        setCurrentAnalysisId(newId);

        const newRecent = {
          id: newId,
          resumeFilename: resumeFile.name,
          atsScore: Math.round(data.ats_score || 0),
          bestFitRole: (data.cognitive_analysis?.best_fit_roles || [])[0]?.role || 'N/A',
          timestamp: new Date().toISOString(),
          pdfDataUri: null
        };
        setRecentAnalyses(prev => {
          const updated = [newRecent, ...prev].slice(0, 5);
          localStorage.setItem('placebuddy_recent', JSON.stringify(updated));
          return updated;
        });

        const extractedResumeText = data.extracted_resume_text || "";
        const extractedJdText = data.extracted_jd_text || jdText || "";

        if (extractedResumeText && extractedJdText) {
          setIsDeepLoading(true);
          const deepPayload = {
            resume_text: extractedResumeText,
            jd_text: extractedJdText,
            // Pass pre-parsed structured data to enable the compact JSON prompt path
            skill_matrix: data.cognitive_analysis?.skill_matrix || null,
            resume_sections: data.resume_sections || null,
          };

          fetch(`${API_BASE_URL}/ats/score/deep`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(deepPayload),
          })
            .then(res => res.ok ? res.json() : Promise.reject(res))
            .then(deepData => { setDeepResult(deepData); })
            .catch(err => console.error("Deep analysis error:", err))
            .finally(() => setIsDeepLoading(false));
        }

        // Save data and target type for the timer-based staged loading screen
        setAnalysisResultData(data);
        setTargetType(type);
      }
    } catch (error) {
      console.error("Network or fetch error:", error);
      setAppState('idle');
    }
  };

  // -- PDF Report Generator ---------------------------------------------------
  const generatePdfReport = () => {
    if (!coreResult) return;
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });
    const pageW = doc.internal.pageSize.getWidth();
    const pageH = doc.internal.pageSize.getHeight();
    const margin = 48;
    const contentW = pageW - margin * 2;
    let y = margin;

    // helpers
    const checkPage = (needed = 24) => {
      if (y + needed > pageH - margin) { doc.addPage(); y = margin; }
    };

    const drawSectionHeader = (title, r, g, b) => {
      checkPage(40);
      doc.setFillColor(r, g, b);
      doc.roundedRect(margin, y, contentW, 28, 4, 4, 'F');
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(11);
      doc.setTextColor(255, 255, 255);
      doc.text(title.toUpperCase(), margin + 12, y + 19);
      y += 38;
      doc.setTextColor(30, 30, 30);
    };

    const drawKeyValue = (label, value) => {
      checkPage(20);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(9);
      doc.setTextColor(100, 100, 120);
      doc.text(label.toUpperCase(), margin + 10, y);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(30, 30, 30);
      doc.text(String(value), margin + 155, y);
      y += 17;
    };

    const drawWrappedText = (text, indent, fontSize) => {
      const ind = indent || 0;
      const fs = fontSize || 9;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(fs);
      doc.setTextColor(50, 50, 60);
      const lines = doc.splitTextToSize(String(text), contentW - ind);
      lines.forEach(line => { checkPage(14); doc.text(line, margin + ind, y); y += 13; });
      y += 3;
    };

    // Flowing badge tags
    const drawTags = (items, r, g, b) => {
      let x = margin;
      const tagH = 18;
      const padX = 9;
      doc.setFontSize(8);
      doc.setFont('helvetica', 'bold');
      items.forEach(item => {
        const label = String(item);
        const tw = doc.getTextWidth(label) + padX * 2;
        if (x + tw > pageW - margin) { x = margin; y += tagH + 5; checkPage(tagH + 5); }
        doc.setFillColor(r, g, b, 0.12);
        doc.setDrawColor(r, g, b);
        doc.roundedRect(x, y - 13, tw, tagH, 3, 3, 'FD');
        doc.setTextColor(r, g, b);
        doc.text(label, x + padX, y - 1);
        x += tw + 7;
      });
      y += tagH + 8;
      doc.setTextColor(30, 30, 30);
    };

    // 3-column badge grid for verified skills
    const drawSkillGrid = (items, r, g, b) => {
      const cols = 3;
      const cellW = Math.floor(contentW / cols);
      const cellH = 22;
      doc.setFontSize(8);
      doc.setFont('helvetica', 'bold');
      let startY = y;
      items.forEach((item, idx) => {
        const col = idx % cols;
        if (col === 0 && idx > 0) { startY += cellH + 5; checkPage(cellH + 5); }
        const x = margin + col * cellW;
        const rowY = idx === 0 ? startY : startY;
        const label = String(item);
        const badgeW = cellW - 10;
        doc.setFillColor(r, g, b, 0.1);
        doc.setDrawColor(r, g, b);
        doc.roundedRect(x + 2, rowY, badgeW, cellH, 3, 3, 'FD');
        doc.setTextColor(r, g, b);
        const trunc = label.length > 22 ? label.slice(0, 21) + '..' : label;
        doc.text(trunc, x + 11, rowY + 14);
      });
      const rows = Math.ceil(items.length / cols);
      y = startY + rows * (cellH + 5) + 8;
      doc.setTextColor(30, 30, 30);
    };

    // -- COVER --
    doc.setFillColor(14, 14, 22);
    doc.rect(0, 0, pageW, 145, 'F');
    doc.setFillColor(99, 102, 241);
    doc.rect(0, 145, pageW, 4, 'F');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(28);
    doc.setTextColor(255, 255, 255);
    doc.text('PlaceBuddy', margin, 68);
    doc.setFontSize(11);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(148, 163, 255);
    doc.text('ATS Analysis Report', margin, 90);
    doc.setFontSize(9);
    doc.setTextColor(100, 105, 140);
    doc.text(
      'Generated: ' + new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }),
      margin, 110
    );
    if (resumeFile) doc.text('Resume: ' + resumeFile.name, margin, 126);

    // ATS ring (cover right)
    const score = Math.round(coreResult.ats_score || 0);
    const cx = pageW - 86, cy = 76, r2 = 44;
    doc.setDrawColor(35, 35, 50); doc.setLineWidth(8);
    doc.circle(cx, cy, r2, 'S');
    const scoreRGB = score >= 70 ? [52, 211, 153] : score >= 45 ? [251, 191, 36] : [239, 68, 68];
    doc.setDrawColor(scoreRGB[0], scoreRGB[1], scoreRGB[2]); doc.setLineWidth(8);
    doc.circle(cx, cy, r2, 'S');
    doc.setFont('helvetica', 'bold'); doc.setFontSize(24); doc.setTextColor(255, 255, 255);
    doc.text(String(score), cx - doc.getTextWidth(String(score)) / 2, cy + 8);
    doc.setFontSize(8); doc.setTextColor(140, 140, 175);
    doc.text('/ 100', cx - doc.getTextWidth('/ 100') / 2, cy + 22);
    doc.text('ATS SCORE', cx - doc.getTextWidth('ATS SCORE') / 2, cy + 36);

    y = 168;

    // -- Pre-compute shared data --
    const cog = coreResult.cognitive_analysis || {};
    const sm = cog.skill_matrix;
    const exp = coreResult.estimated_experience || {};
    const km = coreResult.keyword_metrics || {};

    const allSkills = sm
      ? (Array.isArray(sm)
        ? sm.map(s => ({ skill: s.skill_name, proficiency: s.proficiency_score || 0, yoe: s.estimated_yoe || 0 }))
        : Object.entries(sm).map(([k, v]) => ({ skill: k, proficiency: v.proficiency_score || 0, yoe: v.estimated_yoe || 0 })))
        .sort((a, b) => b.proficiency - a.proficiency)
      : [];

    const verifiedSkills = allSkills.filter(s => s.proficiency >= 50).map(s => formatSkillName(s.skill));
    const missingSkills = (coreResult.missing_skills || []).map(formatSkillName);
    const bestRole = (cog.best_fit_roles || [])[0];

    // -- 1. EXECUTIVE SUMMARY --
    drawSectionHeader('Executive Summary', 99, 102, 241);
    checkPage(108);
    doc.setFillColor(246, 246, 255);
    const summaryH = 81;
    doc.roundedRect(margin, y, contentW, summaryH, 6, 6, 'F');
    doc.setDrawColor(200, 202, 245);
    doc.roundedRect(margin, y, contentW, summaryH, 6, 6, 'S');
    y += 14;
    const scoreLabel = score >= 70 ? 'Strong Match' : score >= 45 ? 'Moderate Match' : 'Needs Improvement';
    drawKeyValue('ATS Score', score + ' / 100 — ' + scoreLabel);
    drawKeyValue('Best Fit Role', bestRole ? bestRole.role + '  (' + bestRole.match_percentage + '% match)' : 'N/A');
    drawKeyValue('Top 3 Verified Skills', verifiedSkills.slice(0, 3).join(', ') || 'N/A');
    drawKeyValue('Top 3 Skill Gaps', missingSkills.slice(0, 3).join(', ') || 'None identified');
    y += 10;

    // -- 2. SKILL PROFICIENCY MATRIX --
    if (allSkills.length) {
      const top15 = allSkills.slice(0, 15);
      drawSectionHeader('Skill Proficiency Matrix', 139, 92, 246);
      top15.forEach(s => {
        checkPage(20);
        const pct = Math.max(0, Math.min(100, Math.round(s.proficiency)));
        const barW = 150;
        const filled = Math.round((pct / 100) * barW);
        const barRGB = pct >= 70 ? [52, 211, 153] : pct >= 40 ? [139, 92, 246] : [251, 191, 36];

        doc.setFont('helvetica', 'normal'); doc.setFontSize(9); doc.setTextColor(30, 30, 50);
        const name = formatSkillName(s.skill);
        doc.text(name.length > 24 ? name.slice(0, 22) + '..' : name, margin, y - 2);

        // bar track
        doc.setFillColor(220, 220, 235);
        doc.roundedRect(margin + 155, y - 11, barW, 10, 2, 2, 'F');
        // bar fill
        doc.setFillColor(barRGB[0], barRGB[1], barRGB[2]);
        if (filled > 0) doc.roundedRect(margin + 155, y - 11, filled, 10, 2, 2, 'F');

        // percentage
        doc.setFont('helvetica', 'bold'); doc.setFontSize(8); doc.setTextColor(70, 70, 100);
        doc.text(pct + '%', margin + 155 + barW + 8, y - 2);

        y += 18;
      });
      y += 6;
    }

    // -- 3. VERIFIED COMPETENCIES --
    if (verifiedSkills.length) {
      drawSectionHeader('Verified Competencies', 16, 160, 100);
      drawSkillGrid(verifiedSkills, 16, 150, 90);
    }

    // -- 4. CRITICAL SKILL GAPS --
    if (missingSkills.length) {
      drawSectionHeader('Critical Skill Gaps', 210, 50, 50);
      checkPage(32);
      doc.setFontSize(9); doc.setFont('helvetica', 'normal'); doc.setTextColor(140, 60, 60);
      doc.text('The following skills are required but insufficient or absent in the resume:', margin, y);
      y += 16;
      drawTags(missingSkills, 200, 55, 55);
    }

    // -- 5. UNVERIFIED SKILLS --
    const bsDetector = cog.bullshit_detector || [];
    if (bsDetector.length) {
      drawSectionHeader('Unverified Skills (Stated Only)', 180, 100, 10);
      doc.setFontSize(9); doc.setFont('helvetica', 'normal'); doc.setTextColor(140, 90, 20);
      doc.text('No concrete project evidence found for these skills:', margin, y);
      y += 16;
      drawTags(bsDetector.map(formatSkillName), 200, 120, 20);
    }

    // -- 6. ROLE MATCH ANALYSIS --
    const roles = cog.best_fit_roles || [];
    if (roles.length) {
      drawSectionHeader('Role Match Analysis', 20, 184, 166);
      roles.forEach((r, i) => {
        checkPage(44);
        const pct = r.match_percentage || 0;
        const rc = pct >= 70 ? [16, 155, 95] : pct >= 45 ? [180, 130, 10] : [190, 50, 50];
        doc.setFont('helvetica', 'bold'); doc.setFontSize(10); doc.setTextColor(rc[0], rc[1], rc[2]);
        doc.text((i + 1) + '.  ' + r.role, margin, y);
        const badge = pct + '% Match';
        const bw = doc.getTextWidth(badge) + 14;
        doc.setFillColor(rc[0], rc[1], rc[2]);
        doc.roundedRect(pageW - margin - bw, y - 12, bw, 16, 3, 3, 'F');
        doc.setFont('helvetica', 'bold'); doc.setFontSize(9); doc.setTextColor(255, 255, 255);
        doc.text(badge, pageW - margin - bw + 7, y - 1);
        y += 16;
        if (r.rationale) drawWrappedText(r.rationale, 14, 8);
        y += 4;
      });
    }

    // -- 7. ML SIGNAL BREAKDOWN --
    drawSectionHeader('ML Signal Breakdown', 79, 70, 229);
    drawKeyValue('Semantic Similarity', ((km.semantic_similarity || 0) * 100).toFixed(0) + '%');
    drawKeyValue('TF-IDF Match', ((km.tfidf_score || 0) * 100).toFixed(0) + '%');
    drawKeyValue('Keyword Match', Number(km.keyword_match || 0).toFixed(0) + '%');
    drawKeyValue('Action Verbs Found', (coreResult.action_verbs_found || []).length + ' verbs');
    drawKeyValue('Soft Skills Found', (coreResult.soft_skills_found || []).length + ' attributes');
    y += 8;

    // -- 8. TOP 3 INTERVIEW QUESTIONS --
    const questions = (deepResult && deepResult.targeted_questions ? deepResult.targeted_questions : []).slice(0, 3);
    if (questions.length) {
      drawSectionHeader('Top Interview Questions', 200, 50, 80);
      questions.forEach((q, i) => {
        checkPage(50);
        const lines = doc.splitTextToSize(String(q), contentW - 32);
        const blockH = lines.length * 13 + 24;
        doc.setFillColor(255, 245, 248);
        doc.roundedRect(margin, y, contentW, blockH, 4, 4, 'F');
        doc.setDrawColor(220, 130, 150);
        doc.roundedRect(margin, y, contentW, blockH, 4, 4, 'S');
        doc.setFont('helvetica', 'bold'); doc.setFontSize(9); doc.setTextColor(180, 50, 80);
        doc.text('Q' + (i + 1), margin + 10, y + 15);
        doc.setFont('helvetica', 'normal'); doc.setFontSize(9); doc.setTextColor(50, 30, 40);
        lines.forEach((line, li) => { doc.text(line, margin + 30, y + 15 + li * 13); });
        y += blockH + 8;
      });
    }

    // -- FOOTER on every page --
    const totalPages = doc.internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
      doc.setPage(i);
      doc.setFillColor(240, 240, 250);
      doc.rect(0, pageH - 28, pageW, 28, 'F');
      doc.setDrawColor(210, 210, 232);
      doc.line(0, pageH - 28, pageW, pageH - 28);
      doc.setFont('helvetica', 'normal'); doc.setFontSize(8); doc.setTextColor(130, 130, 155);
      doc.text('PlaceBuddy  |  Confidential ATS Report', margin, pageH - 10);
      doc.text('Page ' + i + ' of ' + totalPages, pageW - margin - 52, pageH - 10);
    }

    doc.save('PlaceBuddy_Report_' + new Date().toISOString().slice(0, 10) + '.pdf');

    try {
      const pdfDataUri = doc.output('datauristring');
      if (currentAnalysisId) {
        setRecentAnalyses(prev => {
          const updated = prev.map(item => {
            if (item.id === currentAnalysisId) {
              return { ...item, pdfDataUri };
            }
            return item;
          });
          localStorage.setItem('placebuddy_recent', JSON.stringify(updated));
          return updated;
        });
      }
    } catch (err) {
      console.error("Failed to store generated PDF to recent analyses", err);
    }
  };
  // ---------------------------------------------------------------------------
  const renderNavbar = () => (
    <nav className="flex items-center justify-between px-8 py-4 border-b border-white/10 bg-zinc-900/40 backdrop-blur-xl sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-[0_0_20px_rgba(99,102,241,0.5)]">
          <span className="text-white font-black text-2xl leading-none">P</span>
        </div>
        <h1 className="text-2xl font-black bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 drop-shadow-[0_0_10px_rgba(168,85,247,0.4)] tracking-tight">
          PlaceBuddy
        </h1>
      </div>
      <div className="flex items-center gap-8 text-sm font-semibold text-zinc-300">
        <button
          onClick={() => handleNavClick('dashboard-section')}
          className="text-white border-b-2 border-indigo-500 pb-1 transition-all duration-300 hover:text-white"
        >
          Dashboard
        </button>
        <button
          onClick={() => handleNavClick('recent-analyses-section')}
          className="hover:text-white transition-colors pb-1 border-b-2 border-transparent"
        >
          Recent Analyses
        </button>
        <button className="flex items-center gap-2 hover:text-white transition-colors ml-4 bg-white/5 p-2 rounded-full border border-white/10 hover:bg-white/10">
          <UserCircle className="w-6 h-6 text-zinc-300" />
        </button>
      </div>
    </nav>
  );

  const renderInputZone = () => (
    <div id="dashboard-section" className="scroll-mt-24 max-w-6xl mx-auto mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8 px-8 animate-in fade-in slide-in-from-bottom-8 duration-700">
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

        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold text-zinc-400 ml-1 uppercase tracking-wider">Job Description (Optional)</label>
          <textarea
            value={jdText}
            onChange={(e) => {
              setJdText(e.target.value);
              resetResults();
            }}
            placeholder="Paste the job description here..."
            className="min-h-[180px] resize-none rounded-2xl bg-zinc-900/40 backdrop-blur-md border border-zinc-700/50 text-zinc-200 p-5 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all duration-300 placeholder:text-zinc-600 shadow-inner"
          />

          {/* JD PDF Upload */}
          <div className="flex flex-col gap-1 mt-1">
            <label className="text-xs font-semibold text-zinc-400 ml-1 uppercase tracking-wider">Or Upload JD as PDF</label>
            <input
              type="file"
              accept=".pdf"
              ref={jdFileInputRef}
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  setJdFile(e.target.files[0]);
                  resetResults();
                }
              }}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => jdFileInputRef.current?.click()}
              className={`flex items-center gap-3 px-4 py-3 rounded-2xl border transition-all duration-300 text-sm font-medium ${jdFile
                ? 'border-emerald-500/50 bg-emerald-900/20 text-emerald-300 hover:bg-emerald-900/30'
                : 'border-zinc-700/50 bg-zinc-900/40 text-zinc-400 hover:border-indigo-500/50 hover:text-indigo-300 hover:bg-zinc-800/60'
                }`}
            >
              <UploadCloud className={`w-4 h-4 flex-shrink-0 ${jdFile ? 'text-emerald-400' : 'text-zinc-500'}`} />
              <span className="truncate">
                {jdFile ? jdFile.name : 'Click to upload JD PDF'}
              </span>
              {jdFile && (
                <span
                  role="button"
                  aria-label="Remove JD file"
                  onClick={(e) => { e.stopPropagation(); setJdFile(null); resetResults(); }}
                  className="ml-auto text-zinc-500 hover:text-red-400 transition-colors cursor-pointer"
                >
                  ✕
                </span>
              )}
            </button>
          </div>
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

  const renderRecentAnalyses = () => {
    return (
      <div id="recent-analyses-section" className="scroll-mt-24 max-w-4xl mx-auto mt-16 px-8 animate-in fade-in slide-in-from-bottom-6 duration-700">
        <div className="flex items-center justify-between mb-6 border-b border-white/10 pb-4">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <History className="w-5 h-5 text-indigo-400" />
            Recent Analyses
          </h3>
          {recentAnalyses.length > 0 && (
            <button
              onClick={clearRecentEntries}
              className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-red-400 transition-colors py-1 px-2 rounded-lg hover:bg-white/5"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear All
            </button>
          )}
        </div>

        {recentAnalyses.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-8 rounded-2xl bg-zinc-900/20 border border-dashed border-zinc-800/50 text-center text-zinc-500">
            <History className="w-8 h-8 mb-3 opacity-30 text-zinc-400" />
            <p className="text-sm font-medium">No recent analyses yet. Analyzed resumes will appear here.</p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {recentAnalyses.map((entry) => {
              const date = new Date(entry.timestamp);
              const isToday = date.toDateString() === new Date().toDateString();
              const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
              const dateStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
              const displayTime = isToday ? `Today, ${timeStr}` : `${dateStr}, ${timeStr}`;

              return (
                <div
                  key={entry.id}
                  className="flex items-center justify-between p-4 rounded-2xl bg-zinc-900/40 border border-white/5 backdrop-blur-md hover:border-white/10 transition-all duration-300"
                >
                  <div className="flex items-center gap-3.5 min-w-0">
                    <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center flex-shrink-0 text-indigo-400">
                      <FileText className="w-4 h-4" />
                    </div>
                    <div className="min-w-0">
                      <h4 className="text-sm font-semibold text-zinc-200 truncate">{entry.resumeFilename}</h4>
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-0.5 text-xs text-zinc-500">
                        <span>{entry.bestFitRole}</span>
                        <span>•</span>
                        <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {displayTime}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 flex-shrink-0">
                    <div className="bg-zinc-800/60 px-3 py-1.5 rounded-xl border border-white/5 flex items-center gap-1.5">
                      <span className={`text-sm font-black ${entry.atsScore >= 70 ? 'text-emerald-400' : entry.atsScore >= 45 ? 'text-amber-400' : 'text-red-400'
                        }`}>
                        {entry.atsScore}
                      </span>
                      <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">ATS</span>
                    </div>

                    {entry.pdfDataUri ? (
                      <button
                        onClick={() => downloadPdfDataUri(entry.pdfDataUri, `PlaceBuddy_Report_${entry.resumeFilename.replace(/\.[^/.]+$/, "")}.pdf`)}
                        className="p-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white border border-indigo-600/10 transition-all flex items-center justify-center shadow-[0_0_15px_rgba(99,102,241,0.3)] hover:shadow-[0_0_20px_rgba(99,102,241,0.5)]"
                        title="Download PDF report"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                    ) : (
                      <span className="text-[10px] text-zinc-500 font-semibold italic border border-dashed border-zinc-800 px-2 py-1.5 rounded-xl bg-zinc-900/20 select-none">
                        Report not downloaded yet
                      </span>
                    )}

                    <button
                      onClick={() => deleteRecentEntry(entry.id)}
                      className="p-2 rounded-xl bg-white/5 border border-white/5 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 hover:border-red-500/20 transition-all"
                      title="Delete entry"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  const renderAnalyzingView = () => (
    <div className="flex flex-col items-center justify-center mt-20 animate-in fade-in zoom-in duration-500 pb-20">
      <div className="relative flex items-center justify-center w-32 h-32 mb-10">
        <div className="absolute inset-0 rounded-full border-t-2 border-indigo-500 animate-spin opacity-80" style={{ animationDuration: '3s' }}></div>
        <div className="absolute inset-2 rounded-full border-r-2 border-purple-500 animate-spin opacity-80" style={{ animationDirection: 'reverse', animationDuration: '2s' }}></div>
        <div className="absolute inset-4 rounded-full border-b-2 border-emerald-500 animate-spin opacity-80" style={{ animationDuration: '1.5s' }}></div>
        <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full animate-pulse shadow-[0_0_40px_rgba(99,102,241,0.6)] flex items-center justify-center">
          <BrainCircuit className="w-8 h-8 text-white opacity-90 animate-pulse" />
        </div>
      </div>

      <h2 className="text-3xl font-black text-white tracking-wider mb-2">Analyzing Profile</h2>
      <p className="text-zinc-400 text-sm mb-8 font-mono">Analyzing resume against target requirements...</p>

      <div className="w-full max-w-md bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl shadow-2xl flex flex-col gap-4">
        {stages.map((stage, idx) => {
          const isCompleted = idx < currentStageIndex;
          const isActive = idx === currentStageIndex;

          return (
            <div
              key={idx}
              className={`flex items-center gap-4 p-3 rounded-xl transition-all duration-300 ${isActive ? 'bg-indigo-500/5 border border-indigo-500/20 animate-pulse' : 'border border-transparent'
                }`}
            >
              <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center">
                {isCompleted ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 animate-in zoom-in duration-300" />
                ) : isActive ? (
                  <div className="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <div className="w-3.5 h-3.5 rounded-full border border-zinc-700 bg-zinc-800/40"></div>
                )}
              </div>

              <span
                className={`text-sm font-medium transition-colors duration-300 ${isActive
                  ? 'text-indigo-400 font-semibold'
                  : isCompleted
                    ? 'text-zinc-300'
                    : 'text-zinc-600'
                  }`}
              >
                {stage}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );

  const renderExecutiveSummary = () => {
    if (!coreResult) return (
      <div className="flex items-center justify-center p-16 text-zinc-500 italic text-sm">
        No analysis data available.
      </div>
    );
    const score = Math.round(coreResult?.ats_score || 0);
    const cog = coreResult?.cognitive_analysis || {};
    const bestRole = (cog.best_fit_roles || [])[0];

    const matched = coreResult?.technical_skills?.matched || coreResult?.matched_skills || [];
    const missing = coreResult?.technical_skills?.missing || coreResult?.missing_skills || [];
    const km = coreResult?.keyword_metrics || {};
    const verbs = coreResult?.action_verbs_found || [];

    // Compute dynamic strengths list (max 3 items)
    const strengths = [];
    matched.slice(0, 2).forEach(skill => {
      strengths.push(`Strong ${formatSkillName(skill)} alignment`);
    });
    if (km.keyword_match >= 50) {
      strengths.push("Good keyword coverage");
    }
    if (km.semantic_similarity >= 0.5) {
      strengths.push("Strong project relevance");
    } else if (matched.length > 5) {
      strengths.push("Strong core skillset");
    }
    if (verbs.length >= 8 && strengths.length < 3) {
      strengths.push("Strong action verb usage");
    }
    const displayStrengths = strengths.slice(0, 3);

    // Compute dynamic improvements list (max 3 items)
    const improvements = [];
    missing.slice(0, 2).forEach(skill => {
      improvements.push(`Missing ${formatSkillName(skill)}`);
    });
    if (km.keyword_match < 50) {
      improvements.push("Low keyword alignment");
    }
    if (km.semantic_similarity < 0.5) {
      improvements.push("Missing target JD keywords");
    }
    if (verbs.length < 5 && improvements.length < 3) {
      improvements.push("Weak action verb usage");
    }
    const displayImprovements = improvements.slice(0, 3);

    return (
      <div className="bg-zinc-900/40 border border-indigo-500/20 rounded-[2.5rem] p-8 backdrop-blur-xl mb-12 shadow-[0_0_50px_rgba(0,0,0,0.4)] relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent pointer-events-none"></div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 items-center relative z-10">
          <div className="flex flex-col items-center justify-center lg:border-r lg:border-white/10 lg:pr-8">
            <span className="text-xs font-bold text-zinc-400 uppercase tracking-[0.2em] mb-4">ATS Score</span>
            <div className="relative flex items-center justify-center">
              <svg className="w-36 h-36 transform -rotate-90">
                <circle cx="72" cy="72" r="62" stroke="currentColor" strokeWidth="8" fill="transparent" className="text-zinc-800" />
                <circle cx="72" cy="72" r="62" stroke="currentColor" strokeWidth="8" fill="transparent" strokeDasharray="389.5" strokeDashoffset={389.5 - (389.5 * score) / 100} className="text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.6)] transition-all duration-1000 ease-out" />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-zinc-400">{score}</span>
                <span className="text-xs font-semibold text-zinc-500">/100</span>
              </div>
            </div>
          </div>

          <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex flex-col gap-1.5">
              <span className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Best Fit Role</span>
              <span className="text-lg font-bold text-white leading-snug">
                {bestRole ? bestRole.role : 'N/A'}
              </span>
              {bestRole && (
                <span className="text-xs text-indigo-400 font-semibold mt-0.5">
                  {bestRole.match_percentage}% Profile Match
                </span>
              )}
            </div>

            <div className="flex flex-col gap-2">
              <span className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Top Strengths</span>
              <ul className="flex flex-col gap-1.5 mt-1.5">
                {displayStrengths.length > 0 ? displayStrengths.map((str, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-emerald-400 text-xs font-semibold">
                    <span className="text-emerald-400 flex-shrink-0">✓</span>
                    <span className="truncate" title={str}>{str}</span>
                  </li>
                )) : (
                  <li className="text-zinc-500 text-xs italic">No matched strengths</li>
                )}
              </ul>
            </div>

            <div className="flex flex-col gap-2">
              <span className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Areas to Improve</span>
              <ul className="flex flex-col gap-1.5 mt-1.5">
                {displayImprovements.length > 0 ? displayImprovements.map((imp, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-amber-400 text-xs font-semibold">
                    <span className="text-amber-500 flex-shrink-0">⚠</span>
                    <span className="truncate" title={imp}>{imp}</span>
                  </li>
                )) : (
                  <li className="text-emerald-400 text-xs font-semibold flex items-center gap-1">
                    ✓ Perfect job fit!
                  </li>
                )}
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderSkillsAssessment = () => {
    if (!coreResult) return (
      <div className="flex items-center justify-center p-16 text-zinc-500 italic text-sm">
        No skill data available.
      </div>
    );
    let radarData = [];
    let bsDetector = [];

    if (coreResult?.cognitive_analysis) {
      const { skill_matrix, bullshit_detector } = coreResult.cognitive_analysis;
      bsDetector = bullshit_detector || [];
      if (skill_matrix) {
        if (Array.isArray(skill_matrix)) {
          radarData = skill_matrix.map((data) => ({
            skill: formatSkillName(data.skill_name),
            proficiency: data.proficiency_score || 0
          }));
        } else {
          radarData = Object.entries(skill_matrix).map(([skill, data]) => ({
            skill: formatSkillName(skill),
            proficiency: data.proficiency_score || 0
          }));
        }
        radarData = radarData.sort((a, b) => b.proficiency - a.proficiency).slice(0, 15);
      }
    }

    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
        <div className="bg-zinc-900/40 border border-white/10 rounded-[2rem] p-8 backdrop-blur-xl shadow-lg relative overflow-hidden group h-fit lg:col-span-1">
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent"></div>
          <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3 relative z-10">
            <Activity className="w-6 h-6 text-indigo-400" />
            Forensic Skill Matrix
          </h3>

          {radarData.length > 0 ? (
            <div className="h-[350px] w-full relative z-10">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)" />
                  <PolarAngleAxis dataKey="skill" tick={{ fill: '#a1a1aa', fontSize: 10, fontWeight: 'bold' }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#52525b', fontSize: 9 }} />
                  <Radar name="Proficiency" dataKey="proficiency" stroke="#8b5cf6" strokeWidth={2} fill="#8b5cf6" fillOpacity={0.35} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex items-center justify-center h-[350px] text-zinc-500 italic">No skill matrix data available.</div>
          )}
        </div>

        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="bg-zinc-900/40 border border-white/10 rounded-[2rem] p-6 backdrop-blur-xl shadow-md relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent"></div>
            <h4 className="text-sm font-bold text-zinc-300 mb-4 flex items-center gap-2 relative z-10 uppercase tracking-wider">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              Verified Competencies
            </h4>
            <div className="flex flex-wrap gap-2 relative z-10">
              {radarData.filter(r => r.proficiency >= 50).length > 0 ? (
                radarData.filter(r => r.proficiency >= 50).map((r, idx) => (
                  <div key={idx} className="inline-flex items-center justify-center px-3 py-1.5 rounded-lg bg-emerald-950/20 border border-emerald-800/40 text-emerald-400 font-semibold text-xs transition-colors hover:bg-emerald-900/25">
                    {r.skill}
                  </div>
                ))
              ) : (
                <span className="text-xs text-zinc-500 italic">No verified competencies found</span>
              )}
            </div>
          </div>

          <div className="bg-zinc-900/40 border border-white/10 rounded-[2rem] p-6 backdrop-blur-xl shadow-md relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-transparent"></div>
            <h4 className="text-sm font-bold text-zinc-300 mb-4 flex items-center gap-2 relative z-10 uppercase tracking-wider">
              <AlertCircle className="w-5 h-5 text-red-400" />
              Missing Skills (Critical Gaps)
            </h4>
            <div className="flex flex-wrap gap-2 relative z-10">
              {coreResult?.missing_skills?.length > 0 ? (
                coreResult.missing_skills.map((skill, idx) => (
                  <div key={idx} className="inline-flex items-center justify-center px-3 py-1.5 rounded-lg bg-red-950/20 border border-red-900/40 text-red-400 font-semibold text-xs">
                    {formatSkillName(skill)}
                  </div>
                ))
              ) : (
                <div className="text-emerald-400 flex items-center gap-1.5 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-lg font-semibold text-xs">
                  <CheckCircle2 className="w-4 h-4" /> No critical missing skills!
                </div>
              )}
            </div>
          </div>

          <div className="bg-zinc-900/40 border border-white/10 rounded-[2rem] p-6 backdrop-blur-xl shadow-md relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-transparent"></div>
            <h4 className="text-sm font-bold text-zinc-300 mb-2 flex items-center gap-2 relative z-10 uppercase tracking-wider">
              <AlertCircle className="w-5 h-5 text-amber-400" />
              Unverified Stated Skills
            </h4>
            <p className="text-xs text-zinc-400 mb-4 relative z-10">
              No project evidence found. Be ready to defend these in interviews.
            </p>
            <div className="flex flex-wrap gap-2 relative z-10">
              {bsDetector.length > 0 ? bsDetector.map((skill, idx) => (
                <div key={idx} className="inline-flex items-center justify-center px-3 py-1.5 rounded-lg bg-amber-950/20 border border-amber-900/40 text-amber-400 font-semibold text-xs">
                  {formatSkillName(skill)}
                </div>
              )) : (
                <div className="text-emerald-400 flex items-center gap-1.5 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-lg font-semibold text-xs">
                  <CheckCircle2 className="w-4 h-4" /> All stated skills verified!
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderCareerAndInterviewPrep = () => {
    if (!coreResult) return (
      <div className="flex items-center justify-center p-16 text-zinc-500 italic text-sm">
        No analysis data. Upload a resume above to begin.
      </div>
    );
    const cog = coreResult?.cognitive_analysis || {};
    const roles = cog.best_fit_roles || [];
    const dsaBridge = deepResult?.dsa_bridge || [];
    const questions = deepResult?.targeted_questions || [];

    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
        <div className="lg:col-span-1 flex flex-col gap-6">
          <div className="bg-zinc-900/40 border border-white/10 rounded-[2rem] p-8 backdrop-blur-xl shadow-lg relative overflow-hidden group h-fit">
            <div className="absolute inset-0 bg-gradient-to-br from-teal-500/5 to-transparent"></div>
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3 relative z-10">
              <Compass className="w-6 h-6 text-teal-400" />
              Career Constellation
            </h3>

            <div className="flex flex-col gap-4 relative z-10">
              {roles.map((r, i) => (
                <div key={i} className="bg-teal-500/5 border border-teal-500/20 rounded-xl p-4 transition-all hover:bg-teal-500/10 hover:border-teal-500/40 group/role cursor-pointer relative">
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
          </div>
        </div>

        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="bg-zinc-900/40 border border-white/10 rounded-[2rem] p-8 backdrop-blur-xl shadow-lg relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 to-transparent"></div>
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3 relative z-10">
              <Target className="w-6 h-6 text-rose-400" />
              The "Brutal Interviewer" Simulator
            </h3>

            <div className="grid grid-cols-1 gap-4 relative z-10">
              {isDeepLoading ? (
                <div className="text-zinc-400 italic animate-pulse">Loading deep analysis questions...</div>
              ) : questions.length > 0 ? questions.map((item, index) => {
                const questionText = typeof item === 'object' ? item.question : item;
                const expectedAnswer = typeof item === 'object' ? item.expected_answer : null;
                return (
                  <div key={index} className="bg-rose-500/5 border-l-4 border-rose-500/50 rounded-r-xl text-zinc-300 text-sm font-medium transition-all relative">
                    <div className="group/q cursor-pointer hover:bg-rose-500/10 transition-all p-4">
                      <div className="flex justify-between items-center">
                        <span className="text-rose-400 font-bold whitespace-nowrap">Question {index + 1}</span>
                        <ChevronDown className="w-4 h-4 opacity-50 group-hover/q:rotate-180 transition-transform" />
                      </div>
                      <div className="max-h-0 opacity-0 overflow-hidden group-hover/q:max-h-[500px] group-hover/q:opacity-100 group-hover/q:mt-3 transition-all duration-500 ease-in-out">
                        <p className="text-zinc-300 leading-relaxed text-sm">{questionText}</p>
                      </div>
                    </div>

                    {expectedAnswer && (
                      <div className="px-4 pb-4">
                        <button
                          onClick={() => toggleAnswer(index)}
                          className="mt-4 px-3 py-1.5 text-xs font-mono text-emerald-400 hover:text-emerald-300 bg-emerald-900/20 hover:bg-emerald-900/40 border border-emerald-800/50 rounded transition-colors flex items-center gap-2 w-fit"
                        >
                          {revealedAnswers[index] ? '[-] Hide Expected Answer' : '[+] Reveal Expected Answer'}
                        </button>
                        {revealedAnswers[index] && (
                          <div className="mt-3 p-4 bg-slate-950/80 rounded-md border-l-2 border-emerald-500 text-slate-300 text-sm font-mono whitespace-pre-wrap animate-in fade-in slide-in-from-top-2 duration-200">
                            <span className="text-emerald-500 block text-xs uppercase tracking-wider mb-2">// EXPECTED CRITERIA:</span>
                            {expectedAnswer}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              }) : (
                <div className="text-zinc-500 italic text-sm">No targeted questions available.</div>
              )}
            </div>

            {
              dsaBridge.length > 0 && (
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
              )
            }
          </div>
        </div>
      </div>
    );
  };

  const renderAdvancedAnalytics = () => {
    if (!coreResult) return null;
    const km = coreResult?.keyword_metrics || {};
    const signals = coreResult ? [
      { name: "Semantic Similarity", value: `${(km.semantic_similarity * 100 || 0).toFixed(0)}%`, desc: "Cosine similarity between profile and JD", color: "text-indigo-400", bg: "bg-indigo-500/5", border: "border-indigo-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(99,102,241,0.2)]" },
      { name: "TF-IDF Match", value: `${(km.tfidf_score * 100 || 0).toFixed(0)}%`, desc: "Keyword frequency alignment", color: "text-blue-400", bg: "bg-blue-500/5", border: "border-blue-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(59,130,246,0.2)]" },
      { name: "Skill Alignment", value: `${Object.values(coreResult.contextual_skill_weights || {}).reduce((acc, w) => acc + (w > 1 ? 1 : 0), 0)}`, desc: "Skills verified in context", color: "text-emerald-400", bg: "bg-emerald-500/5", border: "border-emerald-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(52,211,153,0.2)]" },
      { name: "Action Verbs", value: `${coreResult.action_verbs_found?.length || 0}`, desc: "Impact-driven action verbs used", color: "text-pink-400", bg: "bg-pink-500/5", border: "border-pink-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(236,72,153,0.2)]" },
      { name: "Soft Skills", value: `${coreResult.soft_skills_found?.length || 0}`, desc: "Interpersonal attributes identified", color: "text-amber-400", bg: "bg-amber-500/5", border: "border-amber-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(251,191,36,0.2)]" },
      { name: "Keyword Match", value: `${(km.keyword_match || 0).toFixed(0)}%`, desc: "Stop-word filtered keyword match", color: "text-teal-400", bg: "bg-teal-500/5", border: "border-teal-500/30", glow: "group-hover:shadow-[0_0_30px_rgba(20,184,166,0.2)]" },
    ] : [];

    return (
      <div className="bg-zinc-900/20 border border-white/5 rounded-[2.5rem] p-6 backdrop-blur-xl shadow-inner mb-12 animate-in fade-in duration-500">
        <button
          onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
          className="w-full flex items-center justify-between py-2 px-4 hover:bg-white/5 rounded-2xl transition-colors text-left"
        >
          <span className="text-xl font-bold text-white flex items-center gap-3">
            <BrainCircuit className="w-6 h-6 text-purple-400" />
            Advanced Analytics
          </span>
          <ChevronDown className={`w-6 h-6 text-purple-400 transition-transform duration-300 ${isAdvancedOpen ? 'rotate-180' : ''}`} />
        </button>

        {isAdvancedOpen && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 mt-8 animate-in fade-in slide-in-from-top-4 duration-500">
            {signals.map((sig, i) => (
              <div key={i} className={`p-6 rounded-3xl border ${sig.bg} ${sig.border} backdrop-blur-md relative overflow-hidden group transition-all duration-300 hover:-translate-y-1 ${sig.glow}`}>
                <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <div className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-2">{sig.name}</div>
                <div className={`text-4xl font-black ${sig.color} mb-3 drop-shadow-md`}>{sig.value}</div>
                <div className="text-sm text-zinc-500 font-medium leading-relaxed">{sig.desc}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderResultsHeader = () => (
    <div className="flex items-center justify-between mb-8">
      <button onClick={() => setAppState('idle')} className="flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200 transition-colors font-medium">
        <ArrowLeft className="w-4 h-4" /> Back to Upload Screen
      </button>
      {coreResult && (
        <button
          onClick={generatePdfReport}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold transition-all duration-200 shadow-[0_0_20px_rgba(99,102,241,0.4)] hover:shadow-[0_0_30px_rgba(99,102,241,0.6)] hover:-translate-y-0.5"
        >
          <Download className="w-4 h-4" />
          Download Report
        </button>
      )}
    </div>
  );

  const renderResultsHeatmap = () => {
    return (
      <div className="max-w-6xl mx-auto mt-12 px-8 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-20">
        {renderResultsHeader()}
        {renderExecutiveSummary()}
        {renderSkillsAssessment()}
      </div>
    );
  };

  const renderResultsDetailed = () => {
    return (
      <div className="max-w-6xl mx-auto mt-12 px-8 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-20">
        {renderResultsHeader()}
        {renderExecutiveSummary()}
        {renderCareerAndInterviewPrep()}
        {renderAdvancedAnalytics()}
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
            {renderRecentAnalyses()}
          </div>
        )}

        {appState === 'analyzing' && renderAnalyzingView()}
        {appState === 'results_heatmap' && renderResultsHeatmap()}
        {appState === 'results_detailed' && renderResultsDetailed()}
      </div>
    </div>
  );
}
