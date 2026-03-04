import { useState, useEffect, useRef } from 'react';
import { Settings, ChevronRight, MessageSquare, Activity, ShieldAlert, Cpu } from 'lucide-react';

// ─── Types ───────────────────────────────────────────────────────────────────
interface EyePosition { x: number; y: number; }
interface HistoryPoint { t: number; score: number; state: string; }

// ─── State colour map (matches engine.py exactly) ────────────────────────────
const STATE_COLOR: Record<string, string> = {
  'Flow':       '#00ff00',
  'Thinking':   '#3498db',
  'Away':       '#ffa500',
  'Distracted': '#e74c3c',
  'Idle':       '#555555',
  'Offline':    '#333333',
  'Terminated': '#333333',
};
const HISTORY_MAX = 60;
function getColor(s: string) { return STATE_COLOR[s] ?? '#555555'; }

// ─── Focus line graph (pure SVG, zero dependencies) ─────────────────────────
function FocusGraph({ data }: { data: HistoryPoint[] }) {
  if (data.length < 2) {
    return (
      <div style={{ height: 72, display:'flex', alignItems:'center', justifyContent:'center' }}>
        <span style={{ fontFamily:'monospace', fontSize:9, color:'#2a2a2a', letterSpacing:'0.15em' }}>
          AWAITING DATA...
        </span>
      </div>
    );
  }
  const W = 264, H = 72, PX = 4, PY = 6;
  const iW = W - PX * 2, iH = H - PY * 2;
  const pts = data.map((d, i) => ({
    x: PX + (i / (data.length - 1)) * iW,
    y: PY + (1 - d.score / 100) * iH,
    d,
  }));
  const line = pts.map((p,i) => `${i===0?'M':'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  const fill = `${line} L${pts[pts.length-1].x},${H} L${pts[0].x},${H} Z`;
  const latestColor = getColor(data[data.length-1].state);
  const grids = [25,50,75].map(v => ({ y:(PY+(1-v/100)*iH).toFixed(1), v }));

  return (
    <svg width={W} height={H} style={{ overflow:'visible', display:'block' }}>
      <defs>
        <linearGradient id="gfill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={latestColor} stopOpacity="0.22"/>
          <stop offset="100%" stopColor={latestColor} stopOpacity="0"/>
        </linearGradient>
        <filter id="ln-glow">
          <feGaussianBlur stdDeviation="1.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {grids.map(g => (
        <g key={g.v}>
          <line x1={PX} x2={W-PX} y1={g.y} y2={g.y}
            stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="3 5"/>
          <text x={W-PX+3} y={parseFloat(g.y)+3}
            fill="rgba(255,255,255,0.13)" fontSize="7" fontFamily="monospace">{g.v}</text>
        </g>
      ))}
      <path d={fill} fill="url(#gfill)"/>
      <path d={line} fill="none" stroke={latestColor} strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round" filter="url(#ln-glow)"
        style={{ transition:'stroke 0.4s' }}/>
      <circle cx={pts[pts.length-1].x} cy={pts[pts.length-1].y} r="3"
        fill={latestColor}
        style={{ filter:`drop-shadow(0 0 4px ${latestColor})`, transition:'fill 0.4s, cx 0.1s, cy 0.1s' }}/>
    </svg>
  );
}

// ─── Radial arc dial ─────────────────────────────────────────────────────────
function ScoreArc({ score, color }: { score: number; color: string }) {
  const R=28, SW=5, C=34, circ=2*Math.PI*R, dash=(score/100)*circ;
  return (
    <svg width={C*2} height={C*2} style={{ transform:'rotate(-90deg)', flexShrink:0 }}>
      <circle cx={C} cy={C} r={R} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={SW}/>
      <circle cx={C} cy={C} r={R} fill="none" stroke={color} strokeWidth={SW}
        strokeDasharray={`${dash.toFixed(1)} ${(circ-dash).toFixed(1)}`} strokeLinecap="round"
        style={{ transition:'stroke-dasharray 0.8s cubic-bezier(0.4,0,0.2,1), stroke 0.4s' }}/>
    </svg>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────
export function CogniFlowSidebar() {
  const [isExpanded, setIsExpanded]           = useState(true);
  const [currentTime, setCurrentTime]         = useState(new Date());
  const [focusScore, setFocusScore]           = useState(100);
  const [flowState, setFlowState]             = useState('Offline');
  const [recoveryLatency, setRecoveryLatency] = useState<number>(0);
  const [eyePosition, setEyePosition]         = useState<EyePosition>({ x:0.5, y:0.5 });
  const [nudge, setNudge]                     = useState('Monitoring your focus. Stay sharp.');
  const [faceDetected, setFaceDetected]       = useState(false);
  const [gazeOn, setGazeOn]                   = useState(false);
  const [history, setHistory]                 = useState<HistoryPoint[]>([]);
  const [sessionSecs, setSessionSecs]         = useState(0);
  const [sessionActive, setSessionActive]     = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const timerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const startRef  = useRef<number>(0);

  // Clock
  useEffect(() => {
    const t = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Session timer
  useEffect(() => {
    if (sessionActive) {
      startRef.current = Date.now() - sessionSecs * 1000;
      timerRef.current = setInterval(() =>
        setSessionSecs(Math.floor((Date.now() - startRef.current) / 1000)), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [sessionActive]);

  // WebSocket
  useEffect(() => {
    const socket = new WebSocket('ws://127.0.0.1:8000/cognitive-stream');
    socketRef.current = socket;
    socket.onopen = () => { setFlowState('Idle'); setSessionActive(true); setSessionSecs(0); };
    socket.onmessage = (e) => {
      const d = JSON.parse(e.data);
      const score = d.focusScore ?? 0;
      const state = d.currentState ?? 'Idle';
      setFocusScore(score);
      setFlowState(state);
      setRecoveryLatency(d.recoveryLatency ?? 0);
      setEyePosition(d.eyePosition ?? { x:0.5, y:0.5 });
      setFaceDetected(d.faceDetected ?? false);
      setGazeOn(d.gazeOnScreen ?? false);
      if (d.nudge) setNudge(d.nudge);
      setHistory(h => {
        const n = [...h, { t: Date.now(), score, state }];
        return n.length > HISTORY_MAX ? n.slice(-HISTORY_MAX) : n;
      });
    };
    socket.onclose = () => { setFlowState('Offline'); setSessionActive(false); };
    return () => socket.close();
  }, []);

  const handleStop = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ command: 'STOP_SESSION' }));
      setFlowState('Terminated');
      setSessionActive(false);
      alert('Session Ended. Analytics report generated in backend folder.');
    }
  };

  const fmt  = (d: Date) => d.toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false });
  const fmtS = (s: number) => `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  const avg  = history.length ? Math.round(history.reduce((a,p)=>a+p.score,0)/history.length) : 0;
  const color = getColor(flowState);

  return (
    <>
      {/* TRIGGER STRIP */}
      {!isExpanded && (
        <div className="fixed top-0 right-0 h-full w-6 cursor-pointer z-[9999] transition-all hover:w-8"
          style={{ background:'rgba(15,15,15,0.8)', backdropFilter:'blur(10px)' }}
          onClick={() => setIsExpanded(true)}>
          <div className="h-full flex flex-col items-center justify-center gap-4">
            <div className="text-[10px] font-black tracking-widest uppercase rotate-180"
              style={{ writingMode:'vertical-rl', color }}>COGNIFLOW ENGINE</div>
            <Activity size={14} style={{ color }} className="animate-pulse"/>
          </div>
        </div>
      )}

      {/* SIDEBAR */}
      <div
        className={`fixed top-0 right-0 h-full flex flex-col z-[9999] transition-transform duration-500 ease-out border-l border-white/10 ${isExpanded?'translate-x-0':'translate-x-full'}`}
        style={{ width:320, background:'rgba(10,10,10,0.82)', backdropFilter:'blur(30px)',
          borderTopLeftRadius:24, borderBottomLeftRadius:24, overflowY:'auto', scrollbarWidth:'none' }}>

        {/* ── HEADER ── */}
        <div className="p-6 border-b border-white/5 relative bg-white/5 flex-shrink-0">
          <button onClick={() => setIsExpanded(false)}
            className="absolute -left-3 top-1/2 -translate-y-1/2 bg-black border border-white/20 p-1.5 rounded-full hover:scale-110 transition-all shadow-lg"
            style={{ color }}>
            <ChevronRight size={18}/>
          </button>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-2.5 h-2.5 rounded-full animate-pulse"
                style={{ backgroundColor:color, boxShadow:`0 0 12px ${color}` }}/>
              <span className="text-white text-xs font-black tracking-widest uppercase">{flowState}</span>
            </div>
            <div className="text-gray-500 text-xs font-mono">{fmt(currentTime)}</div>
          </div>
        </div>

        {/* ── VIDEO ── */}
        <div className="px-5 pt-5 pb-3 flex-shrink-0">
          <div className="text-gray-500 text-[10px] font-bold tracking-[0.2em] uppercase mb-3 flex items-center gap-2">
            <Activity size={12}/> Neural Vision Link
          </div>
          <div className="relative w-full rounded-2xl overflow-hidden shadow-2xl border border-white/5"
            style={{ height:195, background:'#000' }}>
            <img src="http://127.0.0.1:8000/video_feed" alt="Live AI Analysis"
              className="absolute inset-0 w-full h-full object-cover opacity-80" style={{ filter:'grayscale(20%)' }}/>
            {/* Iris dot */}
            <div className="absolute w-2 h-2 bg-white rounded-full z-20 pointer-events-none"
              style={{ left:`${eyePosition.x*100}%`, top:`${eyePosition.y*100}%`,
                transform:'translate(-50%,-50%)', boxShadow:'0 0 10px white', transition:'left 0.1s,top 0.1s' }}/>
            {/* AI badge */}
            <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 px-2 py-1 rounded-md backdrop-blur-md border border-white/10">
              <div className="w-1.5 h-1.5 bg-red-600 rounded-full animate-pulse"/>
              <span className="text-white text-[9px] font-bold uppercase tracking-tighter">AI Processing</span>
            </div>
            {/* Session timer badge */}
            <div className="absolute top-3 right-3 bg-black/60 px-2 py-1 rounded-md backdrop-blur-md border border-white/10">
              <span className="font-mono text-[10px] font-bold" style={{ color }}>{fmtS(sessionSecs)}</span>
            </div>
            {/* Face / Gaze badges */}
            <div className="absolute bottom-3 left-3 right-3 flex gap-2">
              {[
                { label: faceDetected ? 'Face ✓' : 'No Face', active: faceDetected, c: '#00ff00' },
                { label: gazeOn ? 'Gaze On' : 'Gaze Off', active: gazeOn, c: '#3498db' },
              ].map(b => (
                <div key={b.label} className="flex-1 bg-black/60 backdrop-blur-md border border-white/10 rounded-md px-2 py-1 flex items-center gap-1.5">
                  <div className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                    style={{ background: b.active ? b.c : '#333', boxShadow: b.active ? `0 0 5px ${b.c}` : 'none' }}/>
                  <span className="text-[9px] font-bold uppercase tracking-wider"
                    style={{ color: b.active ? b.c : '#444' }}>{b.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── COGNITIVE METRICS ── */}
        <div className="px-5 py-3 border-b border-white/5 flex-shrink-0">
          <div className="text-gray-500 text-[10px] font-bold tracking-[0.2em] uppercase mb-3 flex items-center gap-2">
            <ShieldAlert size={12}/> Cognitive Metrics
          </div>
          {/* Arc + bar */}
          <div className="flex items-center gap-4 mb-3">
            <div className="relative flex-shrink-0" style={{ width:68, height:68 }}>
              <ScoreArc score={focusScore} color={color}/>
              <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                <span className="font-black text-lg leading-none" style={{ color }}>{focusScore}</span>
                <span className="text-[8px] font-mono text-gray-600 tracking-widest mt-0.5">FOCUS</span>
              </div>
            </div>
            <div className="flex-1">
              <div className="flex justify-between mb-1.5">
                <span className="text-gray-400 text-[11px]">Attentional Persistence</span>
                <span className="text-white font-black text-sm">{focusScore}%</span>
              </div>
              <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all duration-1000 ease-in-out"
                  style={{ width:`${focusScore}%`, backgroundColor:color, boxShadow:`0 0 15px ${color}` }}/>
              </div>
              <div className="flex justify-between mt-2">
                <span className="text-[9px] text-gray-600 font-mono">SESSION AVG</span>
                <span className="text-[9px] font-mono font-bold" style={{ color }}>{avg}%</span>
              </div>
            </div>
          </div>
          {/* FRT + Session */}
          <div className="flex gap-2">
            <div className="flex-1 bg-white/5 border border-white/10 p-3 rounded-xl">
              <div className="text-[10px] text-gray-500 font-bold uppercase mb-1">Recovery (FRT)</div>
              <div className="text-lg font-black font-mono" style={{ color }}>
                {(recoveryLatency ?? 0).toFixed(1)}s
              </div>
            </div>
            <div className="flex-1 bg-white/5 border border-white/10 p-3 rounded-xl">
              <div className="text-[10px] text-gray-500 font-bold uppercase mb-1">Session</div>
              <div className="text-lg font-black font-mono" style={{ color }}>{fmtS(sessionSecs)}</div>
            </div>
          </div>
        </div>

        {/* ── FOCUS TREND GRAPH ── */}
        <div className="px-5 py-3 border-b border-white/5 flex-shrink-0">
          <div className="text-gray-500 text-[10px] font-bold tracking-[0.2em] uppercase mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2"><Cpu size={12}/> Focus Trend</div>
            <span className="font-mono" style={{ color }}>{history.length}/{HISTORY_MAX}pts</span>
          </div>
          {/* Legend */}
          <div className="flex gap-3 mb-2 flex-wrap">
            {(['Flow','Thinking','Away','Distracted'] as const).map(s => (
              <div key={s} className="flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full" style={{ background:getColor(s) }}/>
                <span className="text-[8px] font-mono" style={{ color:getColor(s) }}>{s}</span>
              </div>
            ))}
          </div>
          <div style={{ marginLeft:-4 }}>
            <FocusGraph data={history}/>
          </div>
          {history.length > 2 && (
            <div className="flex justify-between mt-1.5">
              <span className="text-[9px] font-mono text-gray-600">
                MIN <span style={{ color:'#e74c3c' }}>{Math.min(...history.map(h=>h.score))}</span>
              </span>
              <span className="text-[9px] font-mono text-gray-600">
                MAX <span style={{ color:'#00ff00' }}>{Math.max(...history.map(h=>h.score))}</span>
              </span>
            </div>
          )}
        </div>

        {/* ── COGNIBOT ── */}
        <div className="px-5 py-3 border-b border-white/5 flex-shrink-0">
          <div className="text-gray-500 text-[10px] font-bold tracking-[0.2em] uppercase mb-3 flex items-center gap-2">
            <MessageSquare size={12}/> CogniBot
          </div>
          <div className="bg-white/5 border border-white/10 rounded-2xl p-4 text-[12px] text-gray-400 leading-relaxed italic">
            {nudge}
          </div>
        </div>

        {/* ── FOOTER ── */}
        <div className="p-6 border-t border-white/5 bg-black/40 flex-shrink-0 mt-auto">
          <button onClick={handleStop}
            className="w-full py-4 text-white font-black text-[10px] tracking-[0.2em] uppercase rounded-2xl transition-all active:scale-95 shadow-lg shadow-red-900/20 hover:brightness-110"
            style={{ background:'linear-gradient(135deg,#e74c3c,#c0392b)' }}>
            Terminate Session
          </button>
          <div className="flex items-center justify-center mt-5">
            <Settings size={18} className="text-gray-600 cursor-pointer hover:text-white transition-colors"/>
          </div>
        </div>

      </div>

      <style>{`
        div[style*="overflow-y: auto"]::-webkit-scrollbar,
        div[style*="overflowY: auto"]::-webkit-scrollbar { display: none; }
      `}</style>
    </>
  );
}