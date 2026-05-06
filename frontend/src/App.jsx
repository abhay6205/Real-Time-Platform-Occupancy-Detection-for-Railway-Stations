import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Activity, Users, AlertTriangle, ShieldCheck, Settings, Radio, Clock, Cpu, Aperture } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, CartesianGrid } from 'recharts';

const API_BASE = 'http://localhost:8000';

function App() {
  const [record, setRecord] = useState(null);
  const [history, setHistory] = useState([]);
  const [lowThresh, setLowThresh] = useState(15);
  const [highThresh, setHighThresh] = useState(41);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);
  const imgRef = useRef(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const occRes = await axios.get(`${API_BASE}/occupancy`);
        setRecord(occRes.data);
        setApiConnected(true);
        if (!videoLoaded) setVideoLoaded(true);
      } catch (err) {
        setApiConnected(false);
      }
      
      try {
        const histRes = await axios.get(`${API_BASE}/history`);
        const formattedHistory = histRes.data.map(item => ({
          ...item,
          time: new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})
        }));
        setHistory(formattedHistory);
      } catch (err) {
        // silently ignore
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [videoLoaded]);

  const handleApplyThresholds = async () => {
    try {
      await axios.post(`${API_BASE}/thresholds`, {
        low_max: lowThresh,
        high_min: highThresh
      });
    } catch (err) {
      // silently ignore
    }
  };

  const getStatusColor = (density) => {
    if (density === 'Low') return 'text-neon-green';
    if (density === 'Medium') return 'text-neon-orange';
    if (density === 'High') return 'text-neon-red';
    return '';
  };

  const getStatusIcon = (density) => {
    if (density === 'Low') return <ShieldCheck size={32} className="text-neon-green" />;
    if (density === 'Medium') return <Activity size={32} className="text-neon-orange" />;
    if (density === 'High') return <AlertTriangle size={32} className="text-neon-red" />;
    return <Activity size={32} />;
  };

  const getDensityBarWidth = () => {
    if (!record) return '0%';
    const count = record.count;
    return Math.min(count / 200 * 100, 100) + '%';
  };

  const getDensityBarColor = (density) => {
    if (density === 'Low') return 'var(--accent-green)';
    if (density === 'Medium') return 'var(--accent-orange)';
    if (density === 'High') return 'var(--accent-red)';
    return 'rgba(255,255,255,0.3)';
  };

  return (
    <div className="dashboard-layout">
      {/* HEADER */}
      <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '15px', padding: '1.2rem', flexShrink: 0 }}>
        <Aperture size={36} className="text-neon-green" />
        <div style={{ textAlign: 'center' }}>
          <h2 className="title-glow" style={{ fontSize: '1.8rem', margin: 0, letterSpacing: '3px', lineHeight: 1 }}>VisionCore™</h2>
          <div style={{ fontSize: '0.75rem', opacity: 0.5, letterSpacing: '4px', marginTop: '4px' }}>OCCUPANCY INTELLIGENCE</div>
        </div>
      </div>

      <div className="dashboard-grid">
        {/* SIDEBAR */}
        <div className="sidebar">

        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: 'auto' }}>
            <Users size={20} opacity={0.7} />
            <h3 style={{ margin: 0, fontSize: '1rem' }}>Live Status</h3>
            <div style={{ marginLeft: 'auto', width: '8px', height: '8px', borderRadius: '50%', background: apiConnected ? 'var(--accent-green)' : 'var(--accent-red)', animation: 'pulse 2s infinite' }}></div>
          </div>
          
          <div style={{ textAlign: 'center', margin: '2rem 0' }}>
            <div style={{ fontSize: '4.5rem', fontWeight: '800', lineHeight: 1, fontFamily: 'Inter, sans-serif' }} className={getStatusColor(record?.density)}>
              {record ? record.count : '--'}
            </div>
            <div style={{ opacity: 0.5, textTransform: 'uppercase', letterSpacing: '3px', marginTop: '0.5rem', fontSize: '0.7rem' }}>
              Passengers Detected
            </div>
          </div>

          {/* Density progress bar */}
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: getDensityBarWidth(), height: '100%', background: getDensityBarColor(record?.density), borderRadius: '3px', transition: 'width 0.5s ease, background 0.5s ease' }}></div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', padding: '0.75rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>
            {getStatusIcon(record?.density)}
            <div style={{ fontSize: '1.1rem', fontWeight: '700', letterSpacing: '1px' }} className={getStatusColor(record?.density)}>
              {record ? record.density.toUpperCase() : 'WAITING...'}
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.5rem' }}>
            <Settings size={18} opacity={0.7} />
            <h4 style={{ margin: 0, fontSize: '0.95rem' }}>System Thresholds</h4>
          </div>
          
          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ opacity: 0.7 }}>Low → Medium</span>
              <span className="text-neon-orange" style={{ fontWeight: '700' }}>{lowThresh}</span>
            </div>
            <input 
              type="range" 
              min="5" max="50" 
              value={lowThresh} 
              onChange={(e) => setLowThresh(parseInt(e.target.value))} 
            />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ opacity: 0.7 }}>Medium → High</span>
              <span className="text-neon-red" style={{ fontWeight: '700' }}>{highThresh}</span>
            </div>
            <input 
              type="range" 
              min="20" max="150" 
              value={highThresh} 
              onChange={(e) => setHighThresh(parseInt(e.target.value))} 
            />
          </div>

          <button className="btn-primary" onClick={handleApplyThresholds}>
            ⚡ UPDATE PARAMETERS
          </button>

          <div style={{ marginTop: 'auto', padding: '0.75rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', fontSize: '0.75rem', opacity: 0.6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <span>🟢 Low</span><span>0 – {lowThresh - 1}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <span>🟠 Medium</span><span>{lowThresh} – {highThresh - 1}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>🔴 High</span><span>{highThresh}+</span>
            </div>
          </div>
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="main-content">
        
        <div className="glass-panel video-container">
          <div style={{ position: 'relative', width: '100%', height: 'auto', borderRadius: '8px' }}>
            {videoLoaded ? (
              <img 
                ref={imgRef}
                src={`${API_BASE}/video_feed`} 
                alt="Live Camera Feed" 
                className="video-feed"
              />
            ) : (
              <div className="empty-state-bg" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '400px', opacity: 0.4 }}>
                 <Radio size={56} style={{ marginBottom: '1rem' }} />
                 <p style={{ letterSpacing: '3px', fontSize: '0.9rem' }}>AWAITING VIDEO STREAM...</p>
                 <p style={{ fontSize: '0.75rem', opacity: 0.6 }}>Start the backend: python main.py</p>
              </div>
            )}
            
            {/* HUD Overlay */}
            <div style={{ position: 'absolute', top: '10px', left: '10px', display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(0,0,0,0.6)', padding: '4px 12px', borderRadius: '4px', fontSize: '0.75rem', letterSpacing: '1px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: apiConnected ? '#00ff88' : 'red', animation: 'pulse 2s infinite' }}></div>
              LIVE CAM 01
            </div>
            <div style={{ position: 'absolute', bottom: '10px', right: '10px', background: 'rgba(0,0,0,0.6)', padding: '4px 12px', borderRadius: '4px', fontSize: '0.75rem', fontFamily: 'monospace', letterSpacing: '1px' }}>
              {record?.timestamp ? new Date(record.timestamp).toISOString() : '—'}
            </div>
            {record && (
              <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.6)', padding: '4px 12px', borderRadius: '4px', fontSize: '0.75rem' }}>
                <span className={getStatusColor(record.density)}>● {record.density.toUpperCase()}</span>
              </div>
            )}
          </div>
        </div>

        {/* Stats row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', flexShrink: 0 }}>
          <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
            <Clock size={18} opacity={0.5} style={{ marginBottom: '0.5rem' }} />
            <div style={{ fontSize: '0.7rem', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '1px' }}>Last Update</div>
            <div style={{ fontSize: '1rem', fontWeight: '600', marginTop: '0.25rem', fontFamily: 'monospace' }}>
              {record?.timestamp ? new Date(record.timestamp).toLocaleTimeString() : '--:--:--'}
            </div>
          </div>
          <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
            <Cpu size={18} opacity={0.5} style={{ marginBottom: '0.5rem' }} />
            <div style={{ fontSize: '0.7rem', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '1px' }}>Model</div>
            <div style={{ fontSize: '1rem', fontWeight: '600', marginTop: '0.25rem' }}>CSRNet</div>
          </div>
          <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
            <Activity size={18} opacity={0.5} style={{ marginBottom: '0.5rem' }} />
            <div style={{ fontSize: '0.7rem', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '1px' }}>EMA Smooth</div>
            <div style={{ fontSize: '1rem', fontWeight: '600', marginTop: '0.25rem' }}>
              {record?.smoothed ? Math.round(record.smoothed) : '--'}
            </div>
          </div>
        </div>

        <div className="glass-panel chart-container">
          <h4 style={{ margin: '0 0 1rem 0', opacity: 0.8 }}>Occupancy Timeline</h4>
          <ResponsiveContainer width="100%" height="85%">
            <AreaChart data={history} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#88aaff" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#88aaff" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
              <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" fontSize={12} tickMargin={10} />
              <YAxis stroke="rgba(255,255,255,0.5)" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'rgba(10, 15, 25, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
              <Area type="monotone" dataKey="count" stroke="#88aaff" strokeWidth={3} fillOpacity={1} fill="url(#colorCount)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        </div>
      </div>
    </div>
  );
}

export default App;
